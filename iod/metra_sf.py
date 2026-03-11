import sys
from typing import Dict

import torch
from torch.nn import functional as F

from iod.metra import METRA
from iod.sac_utils import _clip_actions


class MetraSf(METRA):
    """Contrastive Successor Features (CSF).

    Learns a trajectory encoder phi(s) and successor feature networks
    psi(s,z,a) such that psi ≈ phi(s) + gamma * psi(s',z,a').
    The policy maximises Q(s,z) = psi(s,z,a)^T z.

    Key differences from base METRA:
      - Q-networks output *vectors* (dim_option) not scalars.
      - Policy loss uses SF inner product with z, not scalar Q.
      - Separate SF TD-learning step (no reward shaping).
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    def _train_components(self, epoch_data: Dict[str, torch.tensor]) -> Dict:
        if self.replay_buffer is not None and self.replay_buffer.n_transitions_stored < self.min_buffer_size:
            return {}

        n_epochs = self._trans_optimization_epochs
        bar_width = 30
        for opt_epoch in range(n_epochs):
            frac = (opt_epoch + 1) / n_epochs
            filled = int(bar_width * frac)
            bar = '█' * filled + '░' * (bar_width - filled)
            sys.stdout.write(f'\r  Optimizing  [{bar}] {opt_epoch + 1}/{n_epochs}  {frac * 100:5.1f}%')
            sys.stdout.flush()

            train_store = {}

            if self.replay_buffer is None:
                mini_batch = self._get_mini_tensors(epoch_data)
            else:
                mini_batch = self._sample_replay_buffer()

            # 1) Update trajectory encoder (phi) — trains CNN if enabled
            self._optimize_te(train_store, mini_batch)

            # 2) Update successor feature networks (psi)
            self._optimize_sf(train_store, mini_batch)

            # 3) Optimise policy (pi) and alpha
            self._optimize_op(train_store, mini_batch)

        sys.stdout.write('\n')
        return train_store

    # ------------------------------------------------------------------
    # SF optimiser
    # ------------------------------------------------------------------
    def _optimize_sf(self, train_store: Dict, mini_batch: Dict) -> None:
        self._update_loss_sf_td(train_store, mini_batch)

        self._gradient_descent(
            train_store['LossQf1'] + train_store['LossQf2'],
            optimizer_keys=['qf'],
        )

        self._update_targets()

    # ------------------------------------------------------------------
    # Policy optimiser  (handles both discrete and continuous actions)
    # ------------------------------------------------------------------
    def _optimize_op(self, train_store: Dict, mini_batch: Dict) -> None:
        # Encode observations through CNN (detached — policy shouldn't train CNN)
        obs_encoded = self._encode_obs(mini_batch['obs'], detach=True)
        states = self._get_concat_obs(obs_encoded, mini_batch['options'])

        with torch.no_grad():
            alpha = self.log_alpha.param.exp()

        if self.use_discrete_sac:
            # --- Discrete action space (Atari) ---
            action_dists = self.option_policy._module(states)
            act_probs = action_dists.probs                     # (B, n_actions)
            logits = action_dists.logits
            log_probs = torch.log_softmax(logits, dim=-1)      # (B, n_actions)

            # SF networks: (B, n_actions, dim_option) — one SF vector per action
            # We need to compute psi for every action, then dot with z
            n_actions = act_probs.shape[-1]
            # Build one-hot action matrix
            eye_actions = torch.eye(n_actions, device=self.device)  # (n_actions, n_actions)
            B = states.shape[0]
            # Expand states: (B, 1, state_dim) -> (B, n_actions, state_dim)
            states_exp = states.unsqueeze(1).expand(B, n_actions, -1).reshape(B * n_actions, -1)
            actions_exp = eye_actions.unsqueeze(0).expand(B, -1, -1).reshape(B * n_actions, -1)

            sf1_all = self.qf1(states_exp, actions_exp).reshape(B, n_actions, -1)  # (B, n_actions, dim_option)
            sf2_all = self.qf2(states_exp, actions_exp).reshape(B, n_actions, -1)

            # Q(s,z,a) = psi(s,z,a)^T z
            z = mini_batch['options']  # (B, dim_option)
            q1_all = (sf1_all * z.unsqueeze(1)).sum(dim=-1)  # (B, n_actions)
            q2_all = (sf2_all * z.unsqueeze(1)).sum(dim=-1)
            q_all = torch.min(q1_all, q2_all)

            # Policy loss: sum_a pi(a|s) [alpha * log pi(a|s) - Q(s,z,a)]
            loss_op = (act_probs * (alpha * log_probs - q_all)).sum(dim=-1).mean()

            train_store.update({
                'LossOp': loss_op,
                'SfQMean': q_all.mean(),
            })

            mini_batch.update({
                'new_action_log_probs': log_probs,
                'act_probs': act_probs,
            })
        else:
            # --- Continuous action space ---
            action_dists = self.option_policy._module(states)
            if hasattr(action_dists, 'rsample_with_pre_tanh_value'):
                new_actions_pre_tanh, new_actions = action_dists.rsample_with_pre_tanh_value()
                new_action_log_probs = action_dists.log_prob(new_actions, pre_tanh_value=new_actions_pre_tanh)
            else:
                new_actions = action_dists.rsample()
                new_actions = _clip_actions(self, new_actions)
                new_action_log_probs = action_dists.log_prob(new_actions)

            sf1 = self.qf1(states, new_actions)
            sf2 = self.qf2(states, new_actions)
            q1 = (sf1 * mini_batch['options']).sum(dim=-1)
            q2 = (sf2 * mini_batch['options']).sum(dim=-1)
            q_values = torch.min(q1, q2)

            loss_op = (alpha * new_action_log_probs - q_values).mean()

            train_store.update({
                'LossOp': loss_op,
                'SfQMean': q_values.mean(),
            })

            mini_batch.update({
                'new_action_log_probs': new_action_log_probs,
                'act_probs': None,
            })

        self._gradient_descent(
            loss_op,
            optimizer_keys=['option_policy'],
        )

        # Tune entropy coefficient
        self._update_loss_alpha(train_store, mini_batch)
        self._gradient_descent(
            train_store['LossAlpha'],
            optimizer_keys=['log_alpha'],
        )

        # Enforce alpha floor
        if self._log_alpha_min is not None:
            with torch.no_grad():
                self.log_alpha.param.data.clamp_(min=self._log_alpha_min)

    # ------------------------------------------------------------------
    # Successor feature TD loss  (handles both discrete and continuous)
    # ------------------------------------------------------------------
    def _update_loss_sf_td(self, train_store: Dict, mini_batch: Dict) -> None:
        obs = mini_batch['obs']
        next_obs = mini_batch['next_obs']
        actions = mini_batch['actions']
        options = mini_batch['options']
        next_options = mini_batch['next_options']
        dones = mini_batch['dones']
        assert torch.allclose(options, next_options)

        # --- Convert discrete action indices to one-hot for SF networks ---
        if self.use_discrete_sac:
            n_actions = self._env_spec.action_space.n
            if actions.dim() == 1 or (actions.dim() == 2 and actions.shape[1] == 1):
                action_ids = actions.long().flatten()
            else:
                action_ids = actions.long().argmax(dim=-1)
            actions_onehot = torch.zeros(actions.shape[0], n_actions, device=self.device)
            actions_onehot.scatter_(1, action_ids.unsqueeze(1), 1.0)
        else:
            actions_onehot = actions

        # --- Encode observations through CNN (detached for SF networks) ---
        obs_encoded = self._encode_obs(obs, detach=True)
        next_obs_encoded = self._encode_obs(next_obs, detach=True)

        processed_cat_obs = self._get_concat_obs(obs_encoded, options)
        next_processed_cat_obs = self._get_concat_obs(next_obs_encoded, next_options)

        # --- Compute phi representations (through CNN, NOT detached so TE loss can train CNN) ---
        obs_for_phi = self._encode_obs(obs, detach=False)
        next_obs_for_phi = self._encode_obs(next_obs, detach=False)
        cur_repr = self.traj_encoder(obs_for_phi).mean
        next_repr = self.traj_encoder(next_obs_for_phi).mean

        # --- Current SF predictions ---
        sf1_pred = self.qf1(processed_cat_obs, actions_onehot)
        sf2_pred = self.qf2(processed_cat_obs, actions_onehot)

        # --- Target SF computation ---
        with torch.no_grad():
            if self.use_discrete_sac:
                # Discrete: expectation over actions weighted by policy probs
                next_action_dists = self.option_policy._module(next_processed_cat_obs)
                act_probs = next_action_dists.probs  # (B, n_actions)
                n_actions = act_probs.shape[-1]
                B = next_processed_cat_obs.shape[0]

                eye_actions = torch.eye(n_actions, device=self.device)
                next_exp = next_processed_cat_obs.unsqueeze(1).expand(B, n_actions, -1).reshape(B * n_actions, -1)
                act_exp = eye_actions.unsqueeze(0).expand(B, -1, -1).reshape(B * n_actions, -1)

                tsf1 = self.target_qf1(next_exp, act_exp).reshape(B, n_actions, -1)  # (B, n_actions, dim_option)
                tsf2 = self.target_qf2(next_exp, act_exp).reshape(B, n_actions, -1)

                # Q values per action for min selection
                q1_per_a = (tsf1 * next_options.unsqueeze(1)).sum(dim=-1)  # (B, n_actions)
                q2_per_a = (tsf2 * next_options.unsqueeze(1)).sum(dim=-1)
                q_min_per_a = torch.min(q1_per_a, q2_per_a)  # (B, n_actions)

                # Select SF from the network that gave the min Q for each action
                # Then take expectation over actions weighted by policy
                use_sf1 = (q1_per_a <= q2_per_a).unsqueeze(-1).float()  # (B, n_actions, 1)
                tsf_min = use_sf1 * tsf1 + (1 - use_sf1) * tsf2          # (B, n_actions, dim_option)

                # Entropy bonus for target (soft Bellman)
                logits = next_action_dists.logits
                log_probs = torch.log_softmax(logits, dim=-1)             # (B, n_actions)
                alpha_val = self.log_alpha.param.exp()

                # Weighted sum: sum_a pi(a|s')[psi(s',z,a) - alpha * log pi(a|s') * z]
                # The entropy term is scalar per action, broadcast to dim_option via z
                entropy_bonus = -alpha_val * log_probs  # (B, n_actions)
                # target_next_sf = sum_a pi(a|s') * [psi - alpha*log_pi * z_direction]
                # But SF target should be: phi(s) + gamma * E_a'[psi(s',z,a')]
                # Entropy is handled in the policy loss, not in SF target.
                # Keep it simple: target = E_a'~pi [psi(s',z,a')]
                target_next_sf = (act_probs.unsqueeze(-1) * tsf_min).sum(dim=1)  # (B, dim_option)
            else:
                # Continuous: sample next actions from policy
                next_action_dists = self.option_policy._module(next_processed_cat_obs)
                if hasattr(next_action_dists, 'rsample_with_pre_tanh_value'):
                    _, new_next_actions = next_action_dists.rsample_with_pre_tanh_value()
                else:
                    new_next_actions = next_action_dists.rsample()
                    new_next_actions = _clip_actions(self, new_next_actions)

                target_next_sf1 = self.target_qf1(next_processed_cat_obs, new_next_actions)
                target_next_sf2 = self.target_qf2(next_processed_cat_obs, new_next_actions)
                # Pick the SF from the network with smaller Q = psi^T z
                q1_vals = (target_next_sf1 * next_options).sum(dim=-1)
                q2_vals = (target_next_sf2 * next_options).sum(dim=-1)
                use_sf1 = (q1_vals <= q2_vals).unsqueeze(-1).float()
                target_next_sf = use_sf1 * target_next_sf1 + (1 - use_sf1) * target_next_sf2

            if self.turn_off_dones:
                dones = torch.zeros_like(dones)

            # SF target: phi(s) + gamma * (1 - done) * psi(s', z, a')
            if self.metra_mlp_rep:
                sf_target = self.f_encoder(obs_encoded, next_obs_encoded) + self.discount * (1. - dones[:, None]) * target_next_sf
            elif self.no_diff_in_rep:
                sf_target = cur_repr.detach() + self.discount * (1. - dones[:, None]) * target_next_sf
            else:
                sf_target = (next_repr.detach() - cur_repr.detach()) + self.discount * (1. - dones[:, None]) * target_next_sf

        # SF losses
        loss_sf1 = F.mse_loss(sf1_pred, sf_target)
        loss_sf2 = F.mse_loss(sf2_pred, sf_target)

        # Representation diagnostics (always logged, regardless of dual_reg)
        phi_x = cur_repr.detach()
        phi_y = next_repr.detach()
        phi_diff = phi_y - phi_x
        phi_diff_l2 = torch.square(phi_diff).sum(dim=1).mean()

        # Per-dimension variance of phi — if any dimension has ~0 variance, it's dead
        phi_dim_var = phi_x.var(dim=0)  # (dim_option,)

        train_store.update({
            'Qf1Mean': sf1_pred.mean(),
            'Qf2Mean': sf2_pred.mean(),
            'QfTargetsMean': sf_target.mean(),
            'QfTdErrsMean': ((sf_target - sf1_pred).mean() + (sf_target - sf2_pred).mean()) / 2,
            'LossQf1': loss_sf1,
            'LossQf2': loss_sf2,
            'PhiMean': phi_x.mean(),
            'PhiStd': phi_x.std(),
            'PhiNorm': phi_x.norm(dim=-1).mean(),
            'phi_diff_l2': phi_diff_l2,
            'phi_l2': torch.square(phi_x).sum(dim=1).mean(),
            'PhiDimVarMin': phi_dim_var.min(),
            'PhiDimVarMax': phi_dim_var.max(),
        })

        mini_batch.update({
            'processed_cat_obs': processed_cat_obs,
            'next_processed_cat_obs': next_processed_cat_obs,
        })

    # ------------------------------------------------------------------
    # Target network update
    # ------------------------------------------------------------------
    def _update_targets(self) -> None:
        target_sfs = [self.target_qf1, self.target_qf2]
        sfs = [self.qf1, self.qf2]
        for target_sf, sf in zip(target_sfs, sfs):
            for t_param, param in zip(target_sf.parameters(), sf.parameters()):
                t_param.data.copy_(t_param.data * (1.0 - self.tau) +
                                   param.data * self.tau)

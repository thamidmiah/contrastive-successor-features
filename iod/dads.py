import numpy as np
import torch

from iod.metra import METRA


class DADS(METRA):
    def __init__(
            self,
            *,
            num_alt_samples: int = 100,
            split_group: int = 65536,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_alt_samples = num_alt_samples
        self.split_group = split_group

    def _train_components(self, epoch_data):
        if self.replay_buffer is not None and self.replay_buffer.n_transitions_stored < self.min_buffer_size:
            return {}

        for _ in range(self._trans_optimization_epochs):
            tensors = {}

            if self.replay_buffer is None:
                v = self._get_mini_tensors(epoch_data)
            else:
                v = self._sample_replay_buffer()

            if self._sd_batch_norm:
                self._sd_input_batch_norm.train()
                self._sd_target_batch_norm.train()

            self._optimize_te(tensors, v)

            if self._sd_batch_norm:
                self._sd_input_batch_norm.eval()
                self._sd_target_batch_norm.eval()

            self._update_rewards(tensors, v)
            self._optimize_op(tensors, v)

        return tensors

    def _optimize_te(self, tensors, internal_vars):
        self._update_loss_sd(tensors, internal_vars)

        # Train skill_dynamics (and CNN if present)
        opt_keys = ['skill_dynamics']
        if self._use_cnn_encoder:
            opt_keys.append('cnn')
        self._gradient_descent(
            tensors['LossSd'],
            optimizer_keys=opt_keys,
        )

    def _process_sd_input(self, sd_input):
        if self._sd_batch_norm:
            sd_input = self._sd_input_batch_norm(sd_input)
        return sd_input

    def _process_sd_target(self, sd_target):
        if self._sd_batch_norm:
            sd_target = self._sd_target_batch_norm(sd_target)
        return sd_target

    def _encode_for_sd(self, obs, detach=False):
        """Encode observations for skill dynamics.

        When CNN is enabled, encode raw pixels → 512-dim vectors.
        Otherwise pass through unchanged.
        """
        return self._encode_obs(obs, detach=detach)

    def _update_loss_sd(self, tensors, v):
        # Encode observations through CNN if enabled (with gradients for training)
        obs_enc = self._encode_for_sd(v['obs'], detach=False)
        next_obs_enc = self._encode_for_sd(v['next_obs'], detach=False)

        next_obs_diff = self._process_sd_target(next_obs_enc - obs_enc)

        sd_input = self._get_concat_obs(self._process_sd_input(obs_enc), v['options'])
        next_obs_dists = self.skill_dynamics(sd_input)
        next_obs_log_probs = next_obs_dists.log_prob(next_obs_diff)

        if self.turn_off_dones:
            v['dones'][...] = 0

        next_obs_log_probs = next_obs_log_probs * (1. - v['dones'])
        next_obs_log_prob_mean = next_obs_log_probs.sum() / ((1. - v['dones']).sum() + 1e-12)

        loss_sd = -next_obs_log_prob_mean

        tensors.update({
            'LossSd': loss_sd,
        })

        if self._sd_batch_norm:
            tensors.update({
                'SdTargetRunningMean': self._sd_target_batch_norm.running_mean.mean(),
                'SdTargetRunningVar': self._sd_target_batch_norm.running_var.mean(),
            })

    def _update_rewards(self, tensors, v):
        with torch.no_grad():
            # Encode observations through CNN if enabled (detached for reward computation)
            obs_enc = self._encode_for_sd(v['obs'], detach=True)
            next_obs_enc = self._encode_for_sd(v['next_obs'], detach=True)

            next_obs_diff = self._process_sd_target(next_obs_enc - obs_enc)

            obs_repeated = torch.cat([obs_enc] * self.num_alt_samples, dim=0)
            next_obs_repeated = torch.cat([next_obs_diff] * self.num_alt_samples, dim=0)

            alt_options_shape = (obs_repeated.size(0), self.dim_option)
            if self.discrete:
                alt_options = torch.randint(self.dim_option, (obs_repeated.size(0),)).to(self.device)
                alt_options = torch.eye(self.dim_option, device=self.device)[alt_options]
            else:
                if self.uniform_z:
                    alt_options = torch.rand(alt_options_shape).to(self.device) * 2 - 1
                else:
                    alt_options = torch.normal(mean=torch.zeros(alt_options_shape), std=torch.ones(alt_options_shape)).to(self.device)

            sd_input = self._get_concat_obs(self._process_sd_input(obs_enc), v['options'])
            next_obs_log_probs = self.skill_dynamics(sd_input).log_prob(next_obs_diff)

            split_group = self.split_group
            next_obs_alt_log_probs = []
            for i in range((alt_options_shape[0] + split_group - 1) // split_group):
                start_idx = i * split_group
                end_idx = min((i + 1) * split_group, alt_options_shape[0])
                sd_input = self._get_concat_obs(self._process_sd_input(obs_repeated[start_idx:end_idx]), alt_options[start_idx:end_idx])
                next_obs_alt_log_probs.append(
                    self.skill_dynamics(sd_input).log_prob(next_obs_repeated[start_idx:end_idx])
                )
            next_obs_alt_log_probs = torch.cat(next_obs_alt_log_probs, dim=0).view(self.num_alt_samples, -1)

            rewards = (np.log(self.num_alt_samples + 1) - torch.log(1 + torch.exp(torch.clip(
                next_obs_alt_log_probs - next_obs_log_probs.view(1, -1), -50, 50)).sum(dim=0)))

            tensors.update({
                'DadsSdLogProbMean': next_obs_log_probs.mean(),
                'DadsSdAltLogProbMean': next_obs_alt_log_probs.mean(),
                'DadsRewardMean': rewards.mean(),
            })

        v['rewards'] = rewards

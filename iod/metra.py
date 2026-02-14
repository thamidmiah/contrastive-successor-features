import copy
from typing import Dict, List, Any
from collections import defaultdict

import numpy as np
import torch

import global_context
from garage import TrajectoryBatch
from garagei import log_performance_ex
from iod import sac_utils
from iod.iod import IOD
from garagei.torch.q_functions.continuous_mlp_q_function_ex import ContinuousMLPQFunctionEx
from garagei.replay_buffer.path_buffer_ex import PathBufferEx
from garagei.experiment.option_local_runner import OptionLocalRunner

from iod.utils import get_torch_concat_obs, FigManager, get_option_colors, record_video, draw_2d_gaussians
from iod.cnn_encoder import NatureCNN, ImpalaCNN

class METRA(IOD):
    """This class implements the original METRA algorithm from the paper "METRA: Scalable Unsupervised RL with Metric-Aware Abstraction" (https://arxiv.org/abs/2310.08887).
    """
    def __init__(
            self,
            *,
            qf1: ContinuousMLPQFunctionEx,
            qf2: ContinuousMLPQFunctionEx,
            log_alpha: float,
            tau: float,
            scale_reward: float,
            target_coef: float,
            replay_buffer: PathBufferEx,
            min_buffer_size: int,
            inner: bool,
            dual_reg: int,
            dual_slack: float,
            dual_dist: str,
            pixel_shape: tuple=None,
            self_normalizing: bool = False,
            log_sum_exp: bool = False,
            add_log_sum_exp_to_rewards: bool = False,
            fixed_lam: float = None,
            add_penalty_to_rewards: bool = False,
            no_diff_in_rep: bool = False,
            use_discrete_sac: bool = False,
            turn_off_dones: bool = False,
            f_encoder: torch.nn.Module = None,
            metra_mlp_rep: bool = False,
            eval_goal_metrics: bool = False,
            goal_range: float = None,
            frame_stack: int = None,
            sample_new_z: bool = False,
            num_negative_z: int = 256,
            infonce_lam: float = 1.0,
            diayn_include_baseline: bool = False,
            uniform_z: bool = False,
            num_zero_shot_goals: int = 50,
            use_cnn_encoder: bool = False,
            cnn_type: str = 'nature',
            alpha_intrinsic: float = 0.0,
            cnn_learning_rate: float = 1e-4,
            cnn_encoder: torch.nn.Module = None,  # Pre-created CNN encoder (optional, shared with policy)
            **kwargs,
    ):
        super().__init__(**kwargs)

        # CNN Encoder setup
        self._use_cnn_encoder = use_cnn_encoder
        self._cnn_type = cnn_type
        self._alpha_intrinsic = alpha_intrinsic
        
        # Get observation dimension from env_spec
        obs_dim = self._env_spec.observation_space.flat_dim
        
        if self._use_cnn_encoder:
            # Use pre-created CNN encoder if provided (shared with policy), otherwise create new one
            if cnn_encoder is not None:
                self.cnn_encoder = cnn_encoder
                print(f"[METRA] Using SHARED CNN encoder (already created for policy)")
                print(f"[METRA] Intrinsic reward weight: α={alpha_intrinsic}")
            else:
                # Create new CNN encoder (fallback if not provided)
                print(f"[METRA] Creating NEW {cnn_type.upper()} CNN encoder")
                print(f"[METRA] Intrinsic reward weight: α={alpha_intrinsic}")
                
                if cnn_type == 'nature':
                    self.cnn_encoder = NatureCNN(
                        in_channels=4,  # Frame stack
                        output_dim=512
                    ).to(self.device)
                elif cnn_type == 'impala':
                    self.cnn_encoder = ImpalaCNN(
                        in_channels=4,
                        output_dim=512
                    ).to(self.device)
                else:
                    raise ValueError(f"Unknown CNN type: {cnn_type}")
            
            # CNN optimizer will be managed by OptimizerGroupWrapper in run/train.py
            self.cnn_optimizer = None
            
            # Encoded observation dimension
            self._encoded_obs_dim = 512
            print(f"[METRA] Observation encoding: {obs_dim} -> {self._encoded_obs_dim}")
            print(f"[METRA] CNN will be optimized by OptimizerGroupWrapper")
        else:
            self._encoded_obs_dim = obs_dim
            print(f"[METRA] Using MLP encoder (no CNN)")
            print(f"[METRA] Observation dim: {obs_dim}")

        # Q networks
        self.qf1 = qf1.to(self.device)
        self.qf2 = qf2.to(self.device)

        # Target Q networks
        self.target_qf1 = copy.deepcopy(self.qf1)
        self.target_qf2 = copy.deepcopy(self.qf2)

        self.target_te = copy.deepcopy(self.traj_encoder)

        self.log_alpha = log_alpha.to(self.device)

        self.param_modules.update(
            qf1=self.qf1,
            qf2=self.qf2,
            log_alpha=self.log_alpha,
        )

        self.tau = tau

        self.replay_buffer = replay_buffer
        self.min_buffer_size = min_buffer_size
        self.inner = inner

        self.dual_reg = dual_reg
        self.dual_slack = dual_slack
        self.dual_dist = dual_dist

        self.use_discrete_sac = use_discrete_sac    
        self._reward_scale_factor = scale_reward
        if self.use_discrete_sac:
            self._target_entropy = np.log(self._env_spec.action_space.n) * target_coef
        else:
            self._target_entropy = -np.prod(self._env_spec.action_space.shape).item() / 2. * target_coef

        self.pixel_shape = pixel_shape
        self.self_normalizing = self_normalizing
        self.log_sum_exp = log_sum_exp
        self.add_log_sum_exp_to_rewards = add_log_sum_exp_to_rewards
        self.fixed_lam = fixed_lam
        self.add_penalty_to_rewards = add_penalty_to_rewards
        self.no_diff_in_rep = no_diff_in_rep
        self.turn_off_dones = turn_off_dones
        self.metra_mlp_rep = metra_mlp_rep
        if self.metra_mlp_rep:
            self.f_encoder = f_encoder.to(self.device)
        self.eval_goal_metrics = eval_goal_metrics
        self.goal_range = goal_range
        self.frame_stack = frame_stack
        self.sample_new_z = sample_new_z
        self.num_negative_z = num_negative_z
        self.infonce_lam = infonce_lam
        self.diayn_include_baseline = diayn_include_baseline
        self.uniform_z = uniform_z
        self.num_zero_shot_goals = num_zero_shot_goals

        assert self._trans_optimization_epochs is not None

    @property
    def policy(self):
        return {
            'option_policy': self.option_policy,
        }

    def _get_concat_obs(self, obs: torch.Tensor, option: torch.Tensor) -> torch.Tensor:
        """Straight call to the helper function get_torch_concat_obs.

        Args:
            obs (torch.Tensor): the observation tensor
            option (torch.Tensor): the option tensor

        Returns:
            torch.Tensor: concatenated observation tensor
        """
        return get_torch_concat_obs(obs, option)
    
    def _encode_obs(self, obs: torch.Tensor) -> torch.Tensor:
        """Encode observations through CNN if enabled.
        
        Args:
            obs: Raw observations (batch, obs_dim) or (batch, channels, H, W)
            
        Returns:
            encoded: Encoded observations (batch, encoded_dim)
        """
        if self._use_cnn_encoder:
            return self.cnn_encoder(obs)
        return obs
    
    def _compute_intrinsic_reward(self, obs: torch.Tensor, option: torch.Tensor) -> torch.Tensor:
        """Compute intrinsic reward from skill-state mutual information.
        
        Intrinsic reward = how well the state discriminates the skill.
        Higher when state clearly indicates which skill is being executed.
        
        Args:
            obs: Encoded observations (batch, encoded_dim)
            option: Skill indices (batch, dim_option)
            
        Returns:
            intrinsic_reward: (batch,) intrinsic reward values
        """
        if self._alpha_intrinsic == 0.0:
            return torch.zeros(obs.shape[0], device=self.device)
        
        # Get trajectory encoding
        with torch.no_grad():
            te_output = self.traj_encoder(obs)
            
            if self.discrete:
                # For discrete skills: get skill probabilities
                option_idx = torch.argmax(option, dim=-1)
                log_probs = torch.log_softmax(te_output, dim=-1)
                intrinsic = log_probs[torch.arange(len(option_idx)), option_idx]
            else:
                # For continuous skills: use similarity
                intrinsic = torch.sum(te_output * option, dim=-1)
        
        return intrinsic.detach()

    def _get_train_trajectories_kwargs(self, runner: OptionLocalRunner) -> Dict:
        """Generate train trajectory arguments which are basically just the options (skills).

        Args:
            runner (OptionLocalRunner): the runner object

        Returns:
            Dict: dictionary containing the options
        """
        if self.discrete:
            extras = self._generate_option_extras(np.eye(self.dim_option)[np.random.randint(0, self.dim_option, runner._train_args.batch_size)])
        else:
            if self.uniform_z:
                random_options = np.random.uniform(low=-1.0, high=1.0, size=(runner._train_args.batch_size, self.dim_option))
            else:
                random_options = np.random.randn(runner._train_args.batch_size, self.dim_option)
                if self.unit_length:
                    random_options /= np.linalg.norm(random_options, axis=-1, keepdims=True)
            extras = self._generate_option_extras(random_options)

        return dict(
            extras=extras,
            sampler_key='option_policy',
        )

    def _flatten_data(self, data: Dict[str, List[np.ndarray]]) -> Dict[str, torch.tensor]:
        """
        Joins all trajectories together per key.

        Args:
            data (Dict[str, List[np.ndarray]]): Dict where each key has a list of paths / trajectories.

        Returns:
            Dict[str, torch.tensor]: Dict where each key is a torch tensor of the joined paths.
        """
        epoch_data = {}
        for key, value in data.items():
            epoch_data[key] = torch.tensor(np.concatenate(value, axis=0), dtype=torch.float32, device=self.device)
        return epoch_data

    def _update_replay_buffer(self, data: Dict[str, List[np.ndarray]]) -> None:
        """Update the replay buffer with newly collected data.

        Args:
            data (Dict[str, List[np.ndarray]]): data to add to replay buffer
        """
        if self.replay_buffer is not None:
            # Add paths to the replay buffer
            for i in range(len(data['actions'])):
                # Every i iteration extracts one path (trajectory) from the data
                path: Dict[str, np.ndarray] = {}
                for key in data.keys():
                    cur_list = data[key][i]
                    if cur_list.ndim == 1:
                        cur_list = cur_list[..., np.newaxis]
                    # Flatten observations if they're multi-dimensional (e.g., images)
                    elif key in ['obs', 'next_obs'] and cur_list.ndim > 2:
                        # Reshape from (timesteps, ...) to (timesteps, flat_dim)
                        cur_list = cur_list.reshape(cur_list.shape[0], -1)
                    path[key] = cur_list
                self.replay_buffer.add_path(path)

    def _sample_replay_buffer(self, batch_size: int = None) -> Dict[str, torch.tensor]:
        """Sample batch of transitions from the replay buffer.

        Args:
            batch_size (int, optional): specifies the batch size. Defaults to None.

        Returns:
            Dict[str, torch.tensor]: str keys that map to various data items
        """
        if batch_size is None:
            batch_size = self._trans_minibatch_size

        samples = self.replay_buffer.sample_transitions(batch_size)
        data = {}
        for key, value in samples.items():
            if value.shape[1] == 1 and 'option' not in key:
                value = np.squeeze(value, axis=1)
            
            # Flatten observations if they're multi-dimensional (e.g., images)
            if key in ['obs', 'next_obs'] and value.ndim > 2:
                # Reshape from (batch, ...) to (batch, flat_dim)
                value = value.reshape(value.shape[0], -1)
            
            data[key] = torch.from_numpy(value).float().to(self.device)

        return data

    def _train_once_inner(self, path_data: Dict[str, List[np.ndarray]]) -> Dict:
        """Add trajectories to replay buffer, concatenate all trajectories together into one tensor, and train all components.

        Args:
            path_data (Dict[str, List[np.ndarray]]): dictionary of trajectories

        Returns:
            Dict: dictionary containing the training losses etc. for each component.
        """
        # Add trajectories to replay buffer
        print(f"[DEBUG] _train_once_inner: Adding trajectories to replay buffer")
        self._update_replay_buffer(path_data)
        print(f"[DEBUG] _train_once_inner: Replay buffer size = {self.replay_buffer.n_transitions_stored if self.replay_buffer else 'N/A'}")

        # Concatenate all trajectories together into one tensor
        epoch_data = self._flatten_data(path_data)
        print(f"[DEBUG] _train_once_inner: Data flattened, starting component training")

        # Train all components
        tensors = self._train_components(epoch_data)
        print(f"[DEBUG] _train_once_inner: Component training completed")

        return tensors

    def _train_components(self, epoch_data: Dict[str, torch.tensor]) -> Dict:
        """If replay buffer is full enough, train all components sequentially for _trans_optimization_epochs number of times.

        Args:
            epoch_data (Dict[str, torch.tensor]): Dict containing the data for the current epoch.

        Returns:
            Dict: Dict containing the training losses etc. for each component.
        """
        # Make sure replay buffer is used and has enough transitions
        if self.replay_buffer is not None and self.replay_buffer.n_transitions_stored < self.min_buffer_size:
            print(f"[DEBUG] _train_components: Skipping - buffer not full enough ({self.replay_buffer.n_transitions_stored}/{self.min_buffer_size})")
            return {}
        
        print(f"[DEBUG] _train_components: Starting {self._trans_optimization_epochs} optimization epochs")
        for opt_epoch in range(self._trans_optimization_epochs):
            if opt_epoch % 50 == 0 or opt_epoch == self._trans_optimization_epochs - 1:
                print(f"[DEBUG] _train_components: Optimization epoch {opt_epoch}/{self._trans_optimization_epochs}")
            
            train_store = {}

            # Sample mini batch
            if self.replay_buffer is None:
                mini_batch = self._get_mini_tensors(epoch_data)
            else:
                mini_batch = self._sample_replay_buffer()

            # Update trajectory encoder
            self._optimize_te(train_store, mini_batch)

            # Recompute the rewards since the trajectory encoder has changed
            self._update_rewards(train_store, mini_batch)

            # Optimize the policy
            self._optimize_op(train_store, mini_batch)

        print(f"[DEBUG] _train_components: All {self._trans_optimization_epochs} optimization epochs completed")
        return train_store

    def _optimize_te(self, train_store: Dict, mini_batch: Dict) -> None:
        """Optimizes the METRA representation loss, which consists of updating the trajectory encoder as well as performing dual gradient descent on lambda.

        Args:
            train_store (Dict): dictionary to store training losses
            mini_batch (Dict): dictionary containing the mini batch data
        """
        # Compute trajectory encoder loss
        self._update_loss_te(train_store, mini_batch)

        # Perform one step of gradient descent
        if self._use_cnn_encoder:
            # Update trajectory encoder and CNN encoder together
            self._gradient_descent(
                train_store['LossTe'],
                optimizer_keys=(['traj_encoder', 'cnn'] if not self.metra_mlp_rep else ['f_encoder', 'cnn']),
            )
        else:
            # Update trajectory encoder only
            self._gradient_descent(
                train_store['LossTe'],
                optimizer_keys=(['traj_encoder'] if not self.metra_mlp_rep else ['f_encoder']),
            )

        # If using, perform dual gradient descent on lambda
        if self.dual_reg and self.fixed_lam is None and not self.log_sum_exp:
            # Compute lambda loss
            self._update_loss_dual_lam(train_store, mini_batch)

            # Perform one step of gradient descent
            self._gradient_descent(
                train_store['LossDualLam'],
                optimizer_keys=['dual_lam'],
            )

    def _optimize_op(self, train_store: Dict, mini_batch: Dict) -> None:
        """Computes Q, policy, and alpha loss and updates the corresponding networks.

        Args:
            train_store (Dict): dictionary to store training losses
            mini_batch (Dict): dictionary containing the mini batch data
        """
        # Note: We do NOT encode observations here anymore.
        # Each component (_update_loss_qf, _update_loss_op) will encode observations
        # as needed, to avoid issues with computation graph being freed after first backward.
        
        # Compute hybrid rewards if using intrinsic component
        if self._alpha_intrinsic > 0:
            # For intrinsic rewards, we need encoded observations
            obs_encoded = self._encode_obs(mini_batch['obs']) if self._use_cnn_encoder else mini_batch['obs']
            intrinsic_rewards = self._compute_intrinsic_reward(obs_encoded, mini_batch['options'])
            # Add to training store for logging
            train_store['IntrinsicReward'] = intrinsic_rewards.mean().item()
            train_store['TaskReward'] = mini_batch['rewards'].mean().item()
            # Combine task and intrinsic rewards
            mini_batch['rewards'] = mini_batch['rewards'] + self._alpha_intrinsic * intrinsic_rewards
            train_store['TotalReward'] = mini_batch['rewards'].mean().item()
        
        # Compute Q function loss
        self._update_loss_qf(train_store, mini_batch)

        # Update Q networks only (not CNN - CNN will be updated from policy loss)
        self._gradient_descent(
            train_store['LossQf1'] + train_store['LossQf2'],
            optimizer_keys=['qf'],
        )

        # Compute policy loss
        self._update_loss_op(train_store, mini_batch)

        # Update policy only (not CNN - CNN is updated from trajectory encoder loss)
        self._gradient_descent(
            train_store['LossSacp'],
            optimizer_keys=['option_policy'],
        )

        # Compute alpha entropy loss
        self._update_loss_alpha(train_store, mini_batch)

        # Update alpha
        self._gradient_descent(
            train_store['LossAlpha'],
            optimizer_keys=['log_alpha'],
        )

        # Update target networks
        sac_utils.update_targets(self)

    def _update_rewards(self, train_store: Dict, mini_batch: Dict) -> None:
        """Compute the rewards for the current mini batch using the learned representations.

        Args:
            train_store (Dict): training store
            mini_batch (Dict): mini batch data
        """
        obs = mini_batch['obs']
        next_obs = mini_batch['next_obs']
        
        # CRITICAL: Encode observations through CNN if enabled
        obs = self._encode_obs(obs)
        next_obs = self._encode_obs(next_obs)

        if self.inner:
            cur_z = self.traj_encoder(obs).mean
            next_z = self.traj_encoder(next_obs).mean
            
            # CRITICAL: Always normalize phi representations to prevent norm explosion
            # This breaks the positive feedback loop: phi grows -> Q grows -> gradients explode -> phi grows more
            cur_z = cur_z / (cur_z.norm(dim=-1, keepdim=True) + 1e-8)
            next_z = next_z / (next_z.norm(dim=-1, keepdim=True) + 1e-8)

            target_z = next_z - cur_z

            if self.no_diff_in_rep:
                target_z = cur_z

            if self.self_normalizing:
                target_z = target_z / target_z.norm(dim=-1, keepdim=True)

            if self.log_sum_exp:
                if self.sample_new_z:
                    new_z = torch.randn(self.num_negative_z, self.dim_option, device=mini_batch['options'].device)
                    if self.unit_length:
                        new_z /= torch.norm(new_z, dim=-1, keepdim=True)
                    pairwise_scores = target_z @ new_z.t()
                else:
                    pairwise_scores = target_z @ mini_batch['options'].t()
                log_sum_exp = torch.logsumexp(pairwise_scores, dim=-1)

            if self.discrete:
                masks = (mini_batch['options'] - mini_batch['options'].mean(dim=1, keepdim=True)) * self.dim_option / (self.dim_option - 1 if self.dim_option != 1 else 1)
                rewards = (target_z * masks).sum(dim=1)
                
                # CRITICAL: Scale intrinsic reward to prevent collapse
                # After normalization, intrinsic rewards become ~1e-5 which is too small
                # Scale by 100× to make them comparable to typical reward magnitudes
                intrinsic_scale = 100.0
                rewards = rewards * intrinsic_scale
            else:
                inner = (target_z * mini_batch['options']).sum(dim=1)
                rewards = inner
                
                # CRITICAL: Scale intrinsic reward for continuous case too
                intrinsic_scale = 100.0
                rewards = rewards * intrinsic_scale

            # CRITICAL: Store TWO versions of phi representations:
            # 1. Non-detached for METRA loss (shapes representation via skill discovery)
            # 2. Detached for Q-functions (prevents critic bootstrapping from collapsing representation)
            # This is crucial: let METRA train the encoder, but don't let critic interfere
            mini_batch.update({
                'cur_z': cur_z,  # For METRA loss
                'next_z': next_z,  # For METRA loss
                'cur_z_detached': cur_z.detach(),  # For Q-functions
                'next_z_detached': next_z.detach(),  # For Q-functions
            })

        elif self.metra_mlp_rep:
            # unneccessary but avoids key errors for now
            cur_z = self.traj_encoder(obs).mean
            next_z = self.traj_encoder(next_obs).mean
            
            # CRITICAL: Always normalize phi representations to prevent norm explosion
            cur_z = cur_z / (cur_z.norm(dim=-1, keepdim=True) + 1e-8)
            next_z = next_z / (next_z.norm(dim=-1, keepdim=True) + 1e-8)
            
            # CRITICAL: Store detached versions for Q-functions (prevent collapse)
            mini_batch.update({
                'cur_z': cur_z,
                'next_z': next_z,
                'cur_z_detached': cur_z.detach(),
                'next_z_detached': next_z.detach(),
            })

            rep = self.f_encoder(obs, next_obs)
            rewards = (rep * mini_batch['options']).sum(dim=1)

            if self.log_sum_exp:
                if self.sample_new_z:
                    new_z = torch.randn(self.num_negative_z, self.dim_option, device=mini_batch['options'].device)
                    if self.unit_length:
                        new_z /= torch.norm(new_z, dim=-1, keepdim=True)
                    pairwise_scores = rep @ new_z.t()
                else:
                    pairwise_scores = rep @ mini_batch['options'].t()
                log_sum_exp = torch.logsumexp(pairwise_scores, dim=-1)

        else:
            target_dists = self.traj_encoder(next_obs)

            if self.discrete:
                logits = target_dists.mean
                rewards = -torch.nn.functional.cross_entropy(logits, mini_batch['options'].argmax(dim=1), reduction='none')
            else:
                rewards = target_dists.log_prob(mini_batch['options'])

            if self.diayn_include_baseline:
                rewards -= torch.log(torch.tensor(1/self.dim_option))

        train_store.update({
            'PureRewardMean': rewards.mean(),
            'PureRewardStd': rewards.std(),
        })

        mini_batch['rewards'] = rewards
        if self.log_sum_exp:
            mini_batch['log_sum_exp'] = log_sum_exp

    def _update_loss_te(self, train_store: Dict[str, Any], mini_batch: Dict[str, Any]) -> None:
        """Compute trajectory encoder loss.

        Args:
            train_store (Dict[str, Any]): training store
            mini_batch (Dict[str, Any]): mini batch data
        """
        # First compute the rewards for the current mini batch
        self._update_rewards(train_store, mini_batch)
        rewards = mini_batch['rewards']

        obs = mini_batch['obs']
        next_obs = mini_batch['next_obs']

        # Add constraint if using
        if self.dual_reg:
            dual_lam = self.dual_lam.param.exp()
            x = obs
            y = next_obs
            phi_x = mini_batch['cur_z']
            phi_y = mini_batch['next_z']

            if self.dual_dist == 'one':
                cst_dist = torch.ones_like(x[:, 0])
            else:
                raise NotImplementedError

            inside_l2 = phi_y - phi_x

            cst_penalty = cst_dist - torch.square(inside_l2).sum(dim=1)
            cst_penalty = torch.clamp(cst_penalty, max=self.dual_slack)

            if self.self_normalizing:
                te_obj = rewards
            elif self.log_sum_exp:
                te_obj = rewards - self.infonce_lam * mini_batch['log_sum_exp']
            elif self.fixed_lam is not None:
                te_obj = rewards + self.fixed_lam * cst_penalty
            else:
                te_obj = rewards + dual_lam.detach() * cst_penalty

            mini_batch.update({
                'cst_penalty': cst_penalty
            })
            train_store.update({
                'DualCstPenalty': cst_penalty.mean(),
                'phi_diff_l2': torch.square(phi_y - phi_x).sum(dim=1).mean(),
                'phi_l2': torch.square(phi_x).sum(dim=1).mean(),
            })
        else:
            te_obj = rewards

        loss_te = -te_obj.mean()

        train_store.update({
            'TeObjMean': te_obj.mean(),
            'LossTe': loss_te,
        })

    def _update_loss_dual_lam(self, train_store: Dict[str, Any], mini_batch: Dict[str, Any]) -> None:
        """Compute dual lambda loss.

        Args:
            train_store (Dict[str, Any]): train store
            mini_batch (Dict[str, Any]): mini batch data
        """
        log_dual_lam = self.dual_lam.param
        dual_lam = log_dual_lam.exp()
        loss_dual_lam = log_dual_lam * (mini_batch['cst_penalty'].detach()).mean()

        train_store.update({
            'DualLam': dual_lam,
            'LossDualLam': loss_dual_lam,
        })

    def _update_loss_qf(self, train_store: Dict[str, Any], mini_batch: Dict[str, Any]) -> None:
        """Compute Q function losses.

        Args:
            train_store (Dict[str, Any]): train store
            mini_batch (Dict[str, Any]): mini batch data
        """
        # Encode observations if using CNN
        if self._use_cnn_encoder:
            obs = self._encode_obs(mini_batch['obs'])
            next_obs = self._encode_obs(mini_batch['next_obs'])
        else:
            obs = mini_batch['obs']
            next_obs = mini_batch['next_obs']
        
        # Concatenate with options
        processed_cat_obs = self._get_concat_obs(obs, mini_batch['options'])
        next_processed_cat_obs = self._get_concat_obs(next_obs, mini_batch['next_options'])

        rewards = mini_batch['rewards'] * self._reward_scale_factor

        # Add the log sum exp term to the rewards if using
        if self.add_log_sum_exp_to_rewards:
            # CRITICAL: Use DETACHED phi for Q-function computation
            # This prevents critic from affecting encoder gradients
            target_z = mini_batch['next_z_detached'] - mini_batch['cur_z_detached']
            if self.sample_new_z:
                new_z = torch.randn(self.num_negative_z, self.dim_option, device=mini_batch['options'].device)
                if self.unit_length:
                    new_z /= torch.norm(new_z, dim=-1, keepdim=True)
                pairwise_scores = target_z @ new_z.t()
            else:
                pairwise_scores = target_z @ mini_batch['options'].t()
            log_sum_exp = torch.logsumexp(pairwise_scores, dim=-1)

            rewards -= self.infonce_lam * log_sum_exp

        # Add the METRA penalty term to the rewards if using
        if self.add_penalty_to_rewards:
            # CRITICAL: Use DETACHED phi for Q-function computation
            x = mini_batch['obs']
            phi_x = mini_batch['cur_z_detached']
            phi_y = mini_batch['next_z_detached']
            cst_dist = torch.ones_like(x[:, 0])
            cst_penalty = cst_dist - torch.square(phi_y - phi_x).sum(dim=1)
            cst_penalty = torch.clamp(cst_penalty, max=self.dual_slack)
            rewards += (self.dual_lam.param.exp() * cst_penalty).detach()

        # Compute the Q function losses
        sac_utils.update_loss_qf(
            self, train_store, mini_batch,
            obs=processed_cat_obs,
            actions=mini_batch['actions'],
            next_obs=next_processed_cat_obs,
            dones=mini_batch['dones'],
            rewards=rewards,
            policy=self.option_policy,
            turn_off_dones=self.turn_off_dones,
            use_discrete_sac=self.use_discrete_sac
        )

        mini_batch.update({
            'processed_cat_obs': processed_cat_obs,
            'next_processed_cat_obs': next_processed_cat_obs,
        })

    def _update_loss_op(self, train_store: Dict[str, Any], mini_batch: Dict[str, Any]) -> None:
        """Compute the policy loss.

        Args:
            train_store (Dict[str, Any]): train store
            mini_batch (Dict[str, Any]): mini batch data
        """
        # Encode observations if using CNN
        if self._use_cnn_encoder:
            obs = self._encode_obs(mini_batch['obs'])
        else:
            obs = mini_batch['obs']
        
        # Concatenate with options
        processed_cat_obs = self._get_concat_obs(obs, mini_batch['options'])

        # Compute policy loss
        sac_utils.update_loss_sacp(
            self, train_store, mini_batch,
            obs=processed_cat_obs,
            policy=self.option_policy,
            use_discrete_sac=self.use_discrete_sac
        )

    def _update_loss_alpha(self, train_store: Dict[str, Any], mini_batch: Dict[str, Any]) -> None:
        """Compute alpha entropy coefficient loss.

        Args:
            train_store (Dict[str, Any]): train store
            mini_batch (Dict[str, Any]): mini batch data
        """
        sac_utils.update_loss_alpha(
            self, train_store, mini_batch,
            use_discrete_sac=self.use_discrete_sac
        )

    def _update_target_te(self) -> None:
        """Update target network weights.
        """
        for t_param, param in zip(self.target_te.parameters(), self.traj_encoder.parameters()):
            t_param.data.copy_(t_param.data * (1.0 - self.tau) +
                                param.data * self.tau)

    @torch.no_grad()
    def _evaluate_policy(self, runner: OptionLocalRunner) -> None:
        """Evaluation has many functions, it can:
            (1) Visualize the trajectories in the xy plane.
            (2) Visualize the distribution of learned representations from the last observation of a trajectory.
            (3) Evaluate zero-shot goal reaching.
            (4) Generate and save videos.
            (5) Log performance and debugging metrics.

        Args:
            runner (OptionLocalRunner): the runner object
        """
        # Generate random options
        if self.discrete:
            eye_options = np.eye(self.dim_option)
            random_options = []
            colors = []
            for i in range(self.dim_option):
                num_trajs_per_option = self.num_random_trajectories // self.dim_option + (i < self.num_random_trajectories % self.dim_option)
                for _ in range(num_trajs_per_option):
                    random_options.append(eye_options[i])
                    colors.append(i)
            random_options = np.array(random_options)
            colors = np.array(colors)
            num_evals = len(random_options)
            from matplotlib import cm
            cmap = 'tab10' if self.dim_option <= 10 else 'tab20'
            random_option_colors = []
            for i in range(num_evals):
                random_option_colors.extend([cm.get_cmap(cmap)(colors[i])[:3]])
            random_option_colors = np.array(random_option_colors)
        else:
            if self.uniform_z:
                random_options = np.random.uniform(low=-1.0, high=1.0, size=(self.num_random_trajectories, self.dim_option))
            else:
                random_options = np.random.randn(self.num_random_trajectories, self.dim_option)
                if self.unit_length:
                    random_options = random_options / np.linalg.norm(random_options, axis=1, keepdims=True)
            random_option_colors = get_option_colors(random_options * 4)

        # Generate random trajectories based on random options
        random_trajectories = self._get_trajectories(
            runner,
            sampler_key='option_policy',
            extras=self._generate_option_extras(random_options),
            worker_update=dict(
                _render=False,
                _deterministic_policy=True,
            ),
            env_update=dict(_action_noise_std=None),
        )

        # Visualize trajectories (skip if environment doesn't support rendering)
        if hasattr(runner._env, 'render_trajectories'):
            with FigManager(runner, 'TrajPlot_RandomZ') as fm:
                runner._env.render_trajectories(
                    random_trajectories, random_option_colors, self.eval_plot_axis, fm.ax
                )

        data = self.process_samples(random_trajectories)
        last_obs = torch.stack([torch.from_numpy(ob[-1]).to(self.device) for ob in data['obs']])
        
        # Flatten observations if they're multi-dimensional (e.g., images)
        if last_obs.ndim > 2:
            last_obs = last_obs.flatten(start_dim=1)
        
        # CRITICAL: Encode observations through CNN if enabled
        last_obs = self._encode_obs(last_obs)
        
        option_dists = self.traj_encoder(last_obs)

        option_means = option_dists.mean.detach().cpu().numpy()
        if self.inner:
            option_stddevs = torch.ones_like(option_dists.stddev.detach().cpu()).numpy()
        else:
            option_stddevs = option_dists.stddev.detach().cpu().numpy()
        option_samples = option_dists.mean.detach().cpu().numpy()

        option_colors = random_option_colors

        # Visualize last observation representation distribution
        with FigManager(runner, f'PhiPlot') as fm:
            draw_2d_gaussians(option_means, option_stddevs, option_colors, fm.ax)
            draw_2d_gaussians(
                option_samples,
                [[0.03, 0.03]] * len(option_samples),
                option_colors,
                fm.ax,
                fill=True,
                use_adaptive_axis=True,
            )

        # Evaluate zero-shot goal reaching
        eval_option_metrics = {}
        if self.eval_goal_metrics:
            env = runner._env
            goals = []  # list of (goal_obs, goal_info)
            goal_metrics = defaultdict(list)

            if self.env_name == 'kitchen':
                goal_names = ['BottomBurner', 'LightSwitch', 'SlideCabinet', 'HingeCabinet', 'Microwave', 'Kettle']
                for i in range(self.num_zero_shot_goals):
                    goal_idx = np.random.randint(len(goal_names))
                    goal_name = goal_names[goal_idx]
                    goal_obs = env.render_goal(goal_idx=goal_idx).copy().astype(np.float32)
                    goal_obs = np.tile(goal_obs, self.frame_stack or 1).flatten()
                    goals.append((goal_obs, {'goal_idx': goal_idx, 'goal_name': goal_name}))

            elif self.env_name == 'robobin_image':
                goal_names = ['ReachLeft', 'ReachRight', 'PushFront', 'PushBack']
                for i in range(self.num_zero_shot_goals):
                    goal_idx = np.random.randint(len(goal_names))
                    goal_name = goal_names[goal_idx]
                    goal_obs = env.render_goal(goal_idx=goal_idx).copy().astype(np.float32)
                    goal_obs = np.tile(goal_obs, self.frame_stack or 1).flatten()
                    goals.append((goal_obs, {'goal_idx': goal_idx, 'goal_name': goal_name}))
            
            elif self.env_name in ['dmc_cheetah', 'dmc_quadruped', 'dmc_humanoid']:
                for i in range(self.num_zero_shot_goals):
                    env.reset()
                    state = env.physics.get_state().copy()
                    if self.env_name == 'dmc_cheetah':
                        goal_loc = (np.random.rand(1) * 2 - 1) * self.goal_range
                        state[:1] = goal_loc
                    else:
                        goal_loc = (np.random.rand(2) * 2 - 1) * self.goal_range
                        state[:2] = goal_loc
                    env.physics.set_state(state)
                    if self.env_name == 'dmc_humanoid':
                        for _ in range(50):
                            env.step(np.zeros_like(env.action_space.sample()))
                    else:
                        env.step(np.zeros_like(env.action_space.sample()))
                    goal_obs = env.render(mode='rgb_array', width=64, height=64).copy().astype(np.float32)
                    goal_obs = np.tile(goal_obs, self.frame_stack or 1).flatten()
                    goals.append((goal_obs, {'goal_loc': goal_loc}))

            elif self.env_name in ['ant', 'ant_pixel', 'half_cheetah']:
                for i in range(self.num_zero_shot_goals):
                    env.reset()
                    state = env.unwrapped._get_obs().copy()
                    if self.env_name in ['half_cheetah']:
                        goal_loc = (np.random.rand(1) * 2 - 1) * self.goal_range
                        state[:1] = goal_loc
                        env.set_state(state[:9], state[9:])
                    else:
                        goal_loc = (np.random.rand(2) * 2 - 1) * self.goal_range
                        state[:2] = goal_loc
                        env.set_state(state[:15], state[15:])
                    for _ in range(5):
                        env.step(np.zeros_like(env.action_space.sample()))
                    if self.env_name == 'ant_pixel':
                        goal_obs = env.render(mode='rgb_array', width=64, height=64).copy().astype(np.float32)
                        goal_obs = np.tile(goal_obs, self.frame_stack or 1).flatten()
                    else:
                        goal_obs = env._apply_normalize_obs(state).astype(np.float32)
                    goals.append((goal_obs, {'goal_loc': goal_loc}))

            if self.unit_length:
                mean_length = 1.
            else:
                mean_length = np.linalg.norm(np.random.randn(1000000, self.dim_option), axis=1).mean()

            for method in ['Single', 'Adaptive'] if (self.discrete and self.inner) else ['']:
                for goal_obs, goal_info in goals:
                    obs = env.reset()
                    step = 0
                    done = False
                    success = 0
                    staying_time = 0

                    hit_success_3 = 0
                    end_success_3 = 0
                    at_success_3 = 0

                    hit_success_1 = 0
                    end_success_1 = 0
                    at_success_1 = 0

                    option = None
                    while step < self.max_path_length and not done:
                        if self.inner:
                            if self.no_diff_in_rep: 
                                te_input = torch.from_numpy(goal_obs[None, ...]).to(self.device)
                                te_input = self._encode_obs(te_input)  # Encode through CNN
                                phi = self.traj_encoder(te_input).mean[0]

                                if self.self_normalizing:
                                    phi = phi / phi.norm(dim=-1, keepdim=True)

                                phi = phi.detach().cpu().numpy()
                                if self.discrete:
                                    option = np.eye(self.dim_option)[phi.argmax()]
                                else:
                                    option = phi
                            else:
                                te_input = torch.from_numpy(np.stack([obs, goal_obs])).to(self.device)
                                te_input = self._encode_obs(te_input)  # Encode through CNN
                                phi_s, phi_g = self.traj_encoder(te_input).mean
                                phi_s, phi_g = phi_s.detach().cpu().numpy(), phi_g.detach().cpu().numpy()
                                if self.discrete:
                                    if method == 'Adaptive':
                                        option = np.eye(self.dim_option)[(phi_g - phi_s).argmax()]
                                    else:
                                        if option is None:
                                            option = np.eye(self.dim_option)[(phi_g - phi_s).argmax()]
                                else:
                                    option = (phi_g - phi_s) / np.linalg.norm(phi_g - phi_s) * mean_length
                        else:
                            te_input = torch.from_numpy(goal_obs[None, ...]).to(self.device)
                            te_input = self._encode_obs(te_input)  # Encode through CNN
                            phi = self.traj_encoder(te_input).mean[0]
                            phi = phi.detach().cpu().numpy()
                            if self.discrete:
                                option = np.eye(self.dim_option)[phi.argmax()]
                            else:
                                option = phi
                        action, agent_info = self.option_policy.get_action(np.concatenate([obs, option]))
                        next_obs, _, done, info = env.step(action)
                        obs = next_obs

                        if self.env_name == 'kitchen':
                            _success = env.compute_success(goal_info['goal_idx'])[0]
                            success = max(success, _success)
                            staying_time += _success

                        if self.env_name == 'robobin_image':
                            success = max(success, info['success'])
                            staying_time += info['success']

                        if self.env_name in ['dmc_cheetah', 'dmc_quadruped', 'dmc_humanoid', 'ant', 'ant_pixel', 'half_cheetah']:
                            if self.env_name in ['dmc_cheetah']:
                                cur_loc = env.physics.get_state()[:1]
                            elif self.env_name in ['dmc_quadruped', 'dmc_humanoid']:
                                cur_loc = env.physics.get_state()[:2]
                            elif self.env_name in ['half_cheetah']:
                                cur_loc = env.unwrapped._get_obs()[:1]
                            else:
                                cur_loc = env.unwrapped._get_obs()[:2] 

                            if np.linalg.norm(cur_loc - goal_info['goal_loc']) < 3:
                                hit_success_3 = 1.
                                at_success_3 += 1.

                            if np.linalg.norm(cur_loc - goal_info['goal_loc']) < 1:
                                hit_success_1 = 1.
                                at_success_1 += 1.

                        step += 1

                    if self.env_name == 'kitchen':
                        goal_metrics[f'Kitchen{method}Goal{goal_info["goal_name"]}'].append(success)
                        goal_metrics[f'Kitchen{method}GoalOverall'].append(success * len(goal_names))
                        goal_metrics[f'Kitchen{method}GoalStayingTime{goal_info["goal_name"]}'].append(staying_time)
                        goal_metrics[f'Kitchen{method}GoalStayingTimeOverall'].append(staying_time)

                    elif self.env_name == 'robobin_image':
                        goal_metrics[f'Robobin{method}Goal{goal_info["goal_name"]}'].append(success)
                        goal_metrics[f'Robobin{method}GoalOverall'].append(success * len(goal_names))
                        goal_metrics[f'Robobin{method}GoalStayingTime{goal_info["goal_name"]}'].append(staying_time)
                        goal_metrics[f'Robobin{method}GoalStayingTimeOverall'].append(staying_time)

                    elif self.env_name in ['dmc_cheetah', 'dmc_quadruped', 'dmc_humanoid', 'ant', 'ant_pixel', 'half_cheetah']:
                        if self.env_name in ['dmc_cheetah']:
                            cur_loc = env.physics.get_state()[:1]
                        elif self.env_name in ['dmc_quadruped', 'dmc_humanoid']:
                            cur_loc = env.physics.get_state()[:2]
                        elif self.env_name in ['half_cheetah']:
                            cur_loc = env.unwrapped._get_obs()[:1]
                        else:
                            cur_loc = env.unwrapped._get_obs()[:2]
                        distance = np.linalg.norm(cur_loc - goal_info['goal_loc'])
                        squared_distance = distance ** 2
                        if distance < 3:
                            end_success_3 = 1.
                        if distance < 1:
                            end_success_1 = 1.
                    
                        goal_metrics[f'HitSuccess3{method}'].append(hit_success_3)
                        goal_metrics[f'EndSuccess3{method}'].append(end_success_3)
                        goal_metrics[f'AtSuccess3{method}'].append(at_success_3 / step)

                        goal_metrics[f'HitSuccess1{method}'].append(hit_success_1)
                        goal_metrics[f'EndSuccess1{method}'].append(end_success_1)
                        goal_metrics[f'AtSuccess1{method}'].append(at_success_1 / step)

                        goal_metrics[f'Goal{method}Distance'].append(distance)
                        goal_metrics[f'Goal{method}SquaredDistance'].append(squared_distance)

            goal_metrics = {key: np.mean(value) for key, value in goal_metrics.items()}
            eval_option_metrics.update(goal_metrics)

        # Generate and save videos
        if self.eval_record_video:
            if self.discrete:
                video_options = np.eye(self.dim_option)
                video_options = video_options.repeat(self.num_video_repeats, axis=0)
            else:
                if self.dim_option == 2:
                    radius = 1. if self.unit_length else 1.5
                    video_options = []
                    for angle in [3, 2, 1, 4]:
                        video_options.append([radius * np.cos(angle * np.pi / 4), radius * np.sin(angle * np.pi / 4)])
                    video_options.append([0, 0])
                    for angle in [0, 5, 6, 7]:
                        video_options.append([radius * np.cos(angle * np.pi / 4), radius * np.sin(angle * np.pi / 4)])
                    video_options = np.array(video_options)
                else:
                    video_options = np.random.randn(9, self.dim_option)
                    if self.unit_length:
                        video_options = video_options / np.linalg.norm(video_options, axis=1, keepdims=True)
                video_options = video_options.repeat(self.num_video_repeats, axis=0)
            video_trajectories = self._get_trajectories(
                runner,
                sampler_key='local_option_policy',
                extras=self._generate_option_extras(video_options),
                worker_update=dict(
                    _render=True,
                    _deterministic_policy=True,
                ),
            )
            record_video(runner, 'Video_RandomZ', video_trajectories, skip_frames=self.video_skip_frames)

        # Logging
        if hasattr(runner._env, 'calc_eval_metrics'):
            eval_option_metrics.update(runner._env.calc_eval_metrics(random_trajectories, is_option_trajectories=True))
        with global_context.GlobalContext({'phase': 'eval', 'policy': 'option'}):
            log_performance_ex(
                runner.step_itr,
                TrajectoryBatch.from_trajectory_list(self._env_spec, random_trajectories),
                discount=self.discount,
                additional_records=eval_option_metrics,
            )
        self._log_eval_metrics(runner)
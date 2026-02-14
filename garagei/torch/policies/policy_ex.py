import numpy as np
import torch

from garage.torch.distributions import TanhNormal
from garage.torch.policies.stochastic_policy import StochasticPolicy


class PolicyEx(StochasticPolicy):
    def __init__(self,
                 name,
                 *,
                 module,
                 clip_action=False,
                 omit_obs_idxs=None,
                 option_info=None,
                 force_use_mode_actions=False,
                 ):
        super().__init__(env_spec=None, name=name)

        self._clip_action = clip_action
        self._omit_obs_idxs = omit_obs_idxs

        self._option_info = option_info
        self._force_use_mode_actions = force_use_mode_actions

        self._module = module
        self._obs_preprocessor = None  # Can be set to a CNN encoder later
        self._option_dim = 0  # Set this when using options with CNN encoder

    def process_observations(self, observations):
        # If CNN preprocessor is set, encode observations first
        if self._obs_preprocessor is not None:
            if not isinstance(observations, torch.Tensor):
                observations = torch.as_tensor(observations).float()
            # Ensure observations are on the same device as the preprocessor
            observations = observations.to(next(self._obs_preprocessor.parameters()).device)

            # Encoded obs = 512, encoded obs + option = 520
            # Raw obs = 28224, raw obs + option = 28232
            expected_raw_size = self._obs_preprocessor.expected_obs_size
            expected_encoded_size = self._obs_preprocessor.output_dim
            
            # If observations are already encoded (match encoded size or encoded+option size), skip encoding
            if self._option_dim > 0:
                if observations.shape[-1] == expected_encoded_size + self._option_dim:
                    # Already encoded with option - pass through
                    return observations
            elif observations.shape[-1] == expected_encoded_size:
                # Already encoded without option - pass through
                return observations
            
            if self._option_dim > 0 and observations.shape[-1] > expected_raw_size:
                # Split observation and option
                obs_only = observations[..., :-self._option_dim]
                option = observations[..., -self._option_dim:]
                
                # Encode observation through CNN
                encoded_obs = self._obs_preprocessor(obs_only)
                
                # Re-concatenate encoded observation with option
                observations = torch.cat([encoded_obs, option], dim=-1)
            else:
                # No option concatenated, just encode
                observations = self._obs_preprocessor(observations)
        
        if self._omit_obs_idxs is not None:
            observations = observations.clone()
            observations[:, self._omit_obs_idxs] = 0
        return observations

    def forward(self, observations):
        observations = self.process_observations(observations)
        dist = self._module(observations)
        try:
            ret_mean = dist.mean
            ret_log_std = (dist.variance.sqrt()).log()
            info = dict(mean=ret_mean, log_std=ret_log_std)
        except NotImplementedError:
            info = dict()
        if hasattr(dist, '_normal'):
            info.update(dict(
                normal_mean=dist._normal.mean,
                normal_std=dist._normal.variance.sqrt(),
            ))

        return dist, info

    def forward_mode(self, observations):
        observations = self.process_observations(observations)
        samples = self._module.forward_mode(observations)
        return samples, dict()

    def forward_with_transform(self, observations, *, transform):
        observations = self.process_observations(observations)
        dist, dist_transformed = self._module.forward_with_transform(observations, transform=transform)
        try:
            ret_mean = dist.mean
            ret_log_std = (dist.variance.sqrt()).log()
            ret_mean_transformed = dist_transformed.mean.cpu()
            ret_log_std_transformed = (dist_transformed.variance.sqrt()).log().cpu()
            info = (dict(mean=ret_mean, log_std=ret_log_std),
                    dict(mean=ret_mean_transformed, log_std=ret_log_std_transformed))
        except NotImplementedError:
            info = (dict(),
                    dict())
        return (dist, dist_transformed), info

    def forward_with_chunks(self, observations, *, merge):
        observations = [self.process_observations(o) for o in observations]
        dist = self._module.forward_with_chunks(observations,
                                                merge=merge)
        try:
            ret_mean = dist.mean
            ret_log_std = (dist.variance.sqrt()).log()
            info = dict(mean=ret_mean, log_std=ret_log_std)
        except NotImplementedError:
            info = dict()

        return dist, info

    def get_mode_actions(self, observations):
        with torch.no_grad():
            if not isinstance(observations, torch.Tensor):
                observations = torch.as_tensor(observations).float().to(next(self.parameters()).device)
            samples, info = self.forward_mode(observations)
            return samples.cpu().numpy(), {
                k: v.detach().cpu().numpy()
                for (k, v) in info.items()
            }

    def get_sample_actions(self, observations):
        with torch.no_grad():
            if not isinstance(observations, torch.Tensor):
                observations = torch.as_tensor(observations).float().to(next(self.parameters()).device)
            dist, info = self.forward(observations)
            if isinstance(dist, TanhNormal):
                pre_tanh_values, actions = dist.rsample_with_pre_tanh_value()
                log_probs = dist.log_prob(actions, pre_tanh_values)
                actions = actions.detach().cpu().numpy()
                infos = {
                    k: v.detach().cpu().numpy()
                    for (k, v) in info.items()
                }
                infos['pre_tanh_value'] = pre_tanh_values.detach().cpu().numpy()
                infos['log_prob'] = log_probs.detach().cpu().numpy()
            else:
                actions = dist.sample()
                log_probs = dist.log_prob(actions)
                actions = actions.detach().cpu().numpy()
                infos = {
                    k: v.detach().cpu().numpy()
                    for (k, v) in info.items()
                }
                log_probs = log_probs.detach().cpu().numpy()
                if len(log_probs.shape) == 0:
                    log_probs = log_probs[None]
                infos['log_prob'] = log_probs
            return actions, infos

    def get_actions(self, observations):
        assert isinstance(observations, np.ndarray) or isinstance(observations, torch.Tensor)
        if self._force_use_mode_actions:
            actions, info = self.get_mode_actions(observations)
        else:
            actions, info = self.get_sample_actions(observations)
        if self._clip_action:
            epsilon = 1e-6
            actions = np.clip(
                actions,
                self.env_spec.action_space.low + epsilon,
                self.env_spec.action_space.high - epsilon,
            )
        return actions, info

    def get_action(self, observation):
        with torch.no_grad():

            if not isinstance(observation, torch.Tensor):
                observation = torch.as_tensor(observation).float().to(next(self.parameters()).device)
            observation = observation.unsqueeze(0)
            action, agent_infos = self.get_actions(observation)
            return action[0], {k: v[0] for k, v in agent_infos.items()}

import akro

from garage.envs import EnvSpec

class AkroWrapperTrait:
    @property
    def spec(self):
        # If wrapped env already has a spec, use it (for gymnasium compatibility)
        if hasattr(self.env, 'spec') and hasattr(self.env.spec, 'action_space'):
            return self.env.spec
        # Otherwise create spec from gym spaces
        return EnvSpec(action_space=akro.from_gym(self.action_space),
                       observation_space=akro.from_gym(self.observation_space))


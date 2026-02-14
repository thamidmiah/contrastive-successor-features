"""
Atari environment wrapper with preprocessing.

Phase 1: Basic environment integration
Phase 2: Standard Atari preprocessing (grayscale, resize, frame stack, normalize)
"""

import gymnasium as gym
import ale_py
import numpy as np
from gymnasium.spaces import Box
from gymnasium.wrappers import (
    GrayscaleObservation,
    ResizeObservation,
)

# Register ALE environments with gymnasium
gym.register_envs(ale_py)


class NormalizePixels(gym.ObservationWrapper):
    """Normalize pixel values to [0, 1] range."""
    
    def observation(self, obs):
        """Normalize observation."""
        return obs.astype(np.float32) / 255.0


class FrameStackWrapper(gym.Wrapper):
    """
    Stack frames manually to avoid issues with squeezed dimensions.
    
    This is a simpler implementation that handles 2D or 3D observations.
    """
    
    def __init__(self, env, num_frames=4):
        """Initialize frame stack."""
        super().__init__(env)
        self.num_frames = num_frames
        self.frames = None
        
        # Update observation space
        obs_shape = env.observation_space.shape
        if len(obs_shape) == 2:
            # 2D observation (H, W) -> stack along new axis
            new_shape = (num_frames, obs_shape[0], obs_shape[1])
        elif len(obs_shape) == 3:
            # 3D observation (H, W, C) -> stack frames in channel dimension
            new_shape = (obs_shape[0], obs_shape[1], obs_shape[2] * num_frames)
        else:
            raise ValueError(f"Unexpected observation shape: {obs_shape}")
        
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255 if env.observation_space.dtype == np.uint8 else 1.0,
            shape=new_shape,
            dtype=env.observation_space.dtype
        )
    
    def reset(self, **kwargs):
        """Reset and initialize frame stack."""
        obs = self.env.reset(**kwargs)
        if isinstance(obs, tuple):
            obs = obs[0] if len(obs) > 0 else obs
        
        # Initialize frames with the first observation repeated
        self.frames = [obs.copy() for _ in range(self.num_frames)]
        return self._get_observation()
    
    def step(self, action):
        """Step and update frame stack."""
        step_result = self.env.step(action)
        
        # Handle both 4-tuple and 5-tuple returns
        if len(step_result) == 5:
            obs, reward, terminated, truncated, info = step_result
            done = terminated or truncated
        else:
            obs, reward, done, info = step_result
        
        # Add new frame and remove oldest
        self.frames.append(obs)
        self.frames = self.frames[-self.num_frames:]
        
        return self._get_observation(), reward, done, info
    
    def _get_observation(self):
        """Stack frames into single observation."""
        if len(self.frames[0].shape) == 2:
            # 2D frames: stack along first axis
            return np.stack(self.frames, axis=0)
        else:
            # 3D frames: concatenate along channel axis
            return np.concatenate(self.frames, axis=-1)


class EnvSpec:
    """Custom spec class compatible with garage's interface."""
    def __init__(self, observation_space, action_space):
        self.observation_space = observation_space
        self.action_space = action_space


class AtariEnv(gym.Wrapper):
    """
    Atari environment wrapper with standard preprocessing.
    
    Applies standard Atari preprocessing:
    - Grayscale conversion (optional)
    - Resize to target resolution
    - Frame stacking for temporal information
    - Pixel normalization
    
    Args:
        game: Name of the Atari game (e.g., 'Breakout', 'Pong')
        screen_size: Target screen size (default: 84)
        grayscale: Whether to convert to grayscale (default: True)
        frame_stack: Number of frames to stack (default: 1, set to 4 for temporal info)
        normalize_pixels: Whether to normalize pixels to [0, 1] (default: False)
    """
    
    def __init__(
        self,
        game='Breakout',
        screen_size=84,
        grayscale=True,
        frame_stack=1,
        normalize_pixels=False,
    ):
        """Initialize the Atari environment with preprocessing."""
        # Create base environment
        env = gym.make(f'ALE/{game}-v5', render_mode='rgb_array')
        
        print(f"[AtariEnv] Creating {game} environment with preprocessing:")
        print(f"  Original shape: {env.observation_space.shape}")
        
        # Apply preprocessing wrappers in order
        if grayscale:
            env = GrayscaleObservation(env, keep_dim=True)
            print(f"  After grayscale: {env.observation_space.shape}")
        
        if screen_size != env.observation_space.shape[0]:
            env = ResizeObservation(env, shape=(screen_size, screen_size))
            print(f"  After resize: {env.observation_space.shape}")
        
        if normalize_pixels:
            env = NormalizePixels(env)
            print(f"  After normalize: pixels in [0, 1]")
        
        if frame_stack > 1:
            env = FrameStackWrapper(env, num_frames=frame_stack)
            print(f"  After frame stack: {env.observation_space.shape}")
        
        super().__init__(env)
        
        # Get a dummy observation to determine final shape and dtype
        reset_result = env.reset()
        if isinstance(reset_result, tuple):
            dummy_obs = reset_result[0]
        else:
            dummy_obs = reset_result
        
        if hasattr(dummy_obs, '__array__'):
            dummy_obs = np.array(dummy_obs)
        
        # Create observation space that matches actual observations
        # (normalize_pixels changes dtype from uint8 to float32 and range from [0,255] to [0,1])
        actual_obs_space = Box(
            low=0.0 if normalize_pixels else 0,
            high=1.0 if normalize_pixels else 255,
            shape=dummy_obs.shape,
            dtype=np.float32 if normalize_pixels else np.uint8
        )
        
        # Create custom spec for compatibility with garage
        self._custom_spec = EnvSpec(
            observation_space=actual_obs_space,
            action_space=env.action_space
        )
        
        # Add flat_dim attributes that the codebase expects
        self._custom_spec.observation_space.flat_dim = int(np.prod(dummy_obs.shape))
        self._custom_spec.action_space.flat_dim = env.action_space.n
        
        print(f"  Final observation flat_dim: {self._custom_spec.observation_space.flat_dim}")
        print(f"  Action space: {env.action_space} (flat_dim: {self._custom_spec.action_space.flat_dim})")
        
        self._frame_stack = frame_stack
    
    @property
    def spec(self):
        """Return custom spec for compatibility."""
        return self._custom_spec
    
    def reset(self, **kwargs):
        """Reset the environment."""
        reset_result = self.env.reset(**kwargs)
        # Handle both tuple (obs, info) and single obs returns
        if isinstance(reset_result, tuple):
            obs = reset_result[0]
        else:
            obs = reset_result
        
        # Convert LazyFrames to numpy array if needed
        if hasattr(obs, '__array__'):
            obs = np.array(obs)
        return obs
    
    def step(self, action, **kwargs):
        """Step the environment.
        
        Args:
            action: The action to take
            **kwargs: Additional keyword arguments (e.g., render) - ignored for Atari
        """
        # Convert action to integer for discrete action spaces
        if isinstance(action, np.ndarray):
            # Handle both single element and multi-element arrays
            if action.size == 1:
                action = int(action.item())
            else:
                # For multi-element, take argmax (for one-hot or probability distributions)
                action = int(np.argmax(action))
        elif not isinstance(action, int):
            action = int(action)
        
        step_result = self.env.step(action)
        
        # Handle both 4-tuple and 5-tuple returns
        if len(step_result) == 5:
            obs, reward, terminated, truncated, info = step_result
            done = terminated or truncated
        else:
            obs, reward, done, info = step_result
        
        # Convert LazyFrames to numpy array if needed
        if hasattr(obs, '__array__'):
            obs = np.array(obs)
        return obs, reward, done, info

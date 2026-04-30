"""
Montezuma's Revenge Room 1 Wrapper.

Terminates the episode when the player leaves Room 1 (the starting throne room).
Tracks player x/y position and key/torch/item pickup events via ALE RAM.

Features:
    - No-op reset randomisation: takes 0–30 NOOP actions at episode start
      to create micro-state variety and help skill differentiation.
    - Life-loss detection: terminates on death (not just room exit).

RAM addresses (verified for ALE/MontezumaRevenge-v5):
    RAM[3]  (0x03) : Room number. Starts at 1. Changes when player exits.
    RAM[42] (0x2A) : Player Y position (screen pixel row, decreases = lower on screen).
                     77 = starting platform height, 63 = lower platform.
    RAM[34] (0x22) : Player X position (sprite pixel column).
                     100/101 = starting x, changes as player moves left/right.
    RAM[100](0x64) : Item / key pickup flag. Non-zero when key has been collected.
    RAM[58] (0x3A) : Lives remaining (starts at 5, decreases on death).

Usage:
    from envs.atari.montezuma_room1_wrapper import MontezumaRoom1Wrapper
    from envs.atari.atari_env import AtariEnv

    base_env = AtariEnv(game='MontezumaRevenge', frame_stack=4, normalize_pixels=True)
    env = MontezumaRoom1Wrapper(base_env)
"""

import gymnasium as gym
import numpy as np


class MontezumaRoom1Wrapper(gym.Wrapper):
    """
    Gymnasium wrapper that constrains Montezuma's Revenge to Room 1 only.

    Episode is forcibly terminated (truncated=True) when:
      - The room number (RAM[3]) changes from its initial value.
      - The player loses a life (lives counter drops).

    Each step's info dict is augmented with:
      - 'player_x'   : int, player x pixel position
      - 'player_y'   : int, player y pixel position
      - 'room'       : int, current room number
      - 'has_key'    : bool, whether the key has been collected this episode
      - 'lives'      : int, remaining lives
      - 'left_room'  : bool, whether this step caused a room exit
    """

    # --- Verified RAM addresses for ALE/MontezumaRevenge-v5 ---
    RAM_ROOM   = 3    # 0x03 : Room number (1 = starting room)
    RAM_PLAYER_Y = 42 # 0x2A : Player Y position
    RAM_PLAYER_X = 34 # 0x22 : Player X position
    RAM_KEY    = 100  # 0x64 : Key pickup flag (non-zero = key collected)
    RAM_LIVES  = 58   # 0x3A : Lives remaining

    # Room 1 number as reported by ALE RAM at game start
    ROOM_1_NUMBER = 1

    # NOOP action index in ALE (action 0 = NOOP)
    NOOP_ACTION = 0

    def __init__(self, env, noop_max=30):
        """
        Args:
            env: A wrapped AtariEnv (or any gym env) for MontezumaRevenge.
                 Must expose ALE RAM via env.unwrapped.ale.getRAM().
            noop_max: Maximum number of no-op actions at episode start (0 to disable).
                      Randomly samples between 0 and noop_max NOOPs each reset.
        """
        super().__init__(env)
        self._initial_room = self.ROOM_1_NUMBER
        self._initial_lives = None
        self._has_key = False
        self._noop_max = noop_max

    # ------------------------------------------------------------------
    # Pass through the AtariEnv custom spec so garage / akro are happy
    # ------------------------------------------------------------------

    @property
    def spec(self):
        """Delegate to inner AtariEnv spec (has flat_dim attributes)."""
        return self.env.spec

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_ale(self):
        """Walk the wrapper stack to find the ALE object."""
        e = self.env
        while hasattr(e, 'env'):
            if hasattr(e, 'unwrapped'):
                break
            e = e.env
        return e.unwrapped.ale

    def _read_ram(self):
        """Return the 128-byte RAM array."""
        return self._get_ale().getRAM()

    def _parse_state(self, ram):
        """Extract game state fields from RAM array."""
        return {
            'room':     int(ram[self.RAM_ROOM]),
            'player_x': int(ram[self.RAM_PLAYER_X]),
            'player_y': int(ram[self.RAM_PLAYER_Y]),
            'key_flag': int(ram[self.RAM_KEY]),
            'lives':    int(ram[self.RAM_LIVES]),
        }

    # ------------------------------------------------------------------
    # Gymnasium interface
    # ------------------------------------------------------------------

    def reset(self, **kwargs):
        """Reset env, apply no-op randomisation, record initial room and lives."""
        result = self.env.reset(**kwargs)
        # Handle both (obs, info) tuple and bare obs
        if isinstance(result, tuple):
            obs, info = result[0], result[1] if len(result) > 1 else {}
        else:
            obs, info = result, {}

        # --- No-op reset randomisation ---
        # Take a random number of NOOP actions to create micro-state variety.
        # This changes timing and sprite animations without altering the task,
        # giving the policy slightly different initial conditions each episode.
        if self._noop_max > 0:
            n_noops = np.random.randint(0, self._noop_max + 1)
            for _ in range(n_noops):
                step_result = self.env.step(self.NOOP_ACTION)
                if len(step_result) == 5:
                    obs, _, terminated, truncated, info = step_result
                    if terminated or truncated:
                        # Died during noops (very unlikely but be safe)
                        result = self.env.reset(**kwargs)
                        if isinstance(result, tuple):
                            obs = result[0]
                        else:
                            obs = result
                        break
                else:
                    obs, _, done, info = step_result
                    if done:
                        result = self.env.reset(**kwargs)
                        if isinstance(result, tuple):
                            obs = result[0]
                        else:
                            obs = result
                        break

        ram = self._read_ram()
        state = self._parse_state(ram)

        self._initial_room  = state['room']
        self._initial_lives = state['lives']
        self._has_key       = False

        info.update({
            'player_x':  state['player_x'],
            'player_y':  state['player_y'],
            'room':      state['room'],
            'has_key':   self._has_key,
            'lives':     state['lives'],
            'left_room': False,
        })
        return obs

    def step(self, action, **kwargs):
        """Step env and check room-exit / death conditions."""
        result = self.env.step(action, **kwargs)

        # Unpack — inner AtariEnv returns 4-tuple (obs, rew, done, info)
        if len(result) == 5:
            obs, reward, terminated, truncated, info = result
        else:
            obs, reward, terminated_or_done, info = result
            terminated = bool(terminated_or_done)
            truncated  = False

        ram   = self._read_ram()
        state = self._parse_state(ram)

        # Track key pickup
        if state['key_flag'] > 0:
            self._has_key = True

        # --- Room-exit detection ---
        left_room = (state['room'] != self._initial_room)

        # --- Life-loss detection (player died inside room) ---
        lost_life = (
            self._initial_lives is not None
            and state['lives'] < self._initial_lives
        )

        # Truncate episode if player left room 1
        if left_room or lost_life:
            truncated = True
            terminated = False  # not a natural terminal; we imposed it

        # Merge into info
        info.update({
            'player_x':  state['player_x'],
            'player_y':  state['player_y'],
            'room':      state['room'],
            'has_key':   self._has_key,
            'lives':     state['lives'],
            'left_room': left_room,
        })

        # Return 4-tuple to match AtariEnv / rest of the codebase
        done = terminated or truncated
        return obs, reward, done, info

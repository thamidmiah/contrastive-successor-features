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

    RAM_ROOM   = 3    
    RAM_PLAYER_Y = 42 
    RAM_PLAYER_X = 34 
    RAM_KEY    = 100  
    RAM_LIVES  = 58

    ROOM_1_NUMBER = 1

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

    @property
    def spec(self):
        return self.env.spec

    def _get_ale(self):
        e = self.env
        while hasattr(e, 'env'):
            if hasattr(e, 'unwrapped'):
                break
            e = e.env
        return e.unwrapped.ale

    def _read_ram(self):
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

    def reset(self, **kwargs):
        """Reset env, apply no-op randomisation, record initial room and lives."""
        result = self.env.reset(**kwargs)
        # Handle both (obs, info) tuple and bare obs
        if isinstance(result, tuple):
            obs, info = result[0], result[1] if len(result) > 1 else {}
        else:
            obs, info = result, {}

        # --- No-op reset randomisation ---
        if self._noop_max > 0:
            n_noops = np.random.randint(0, self._noop_max + 1)
            for _ in range(n_noops):
                step_result = self.env.step(self.NOOP_ACTION)
                if len(step_result) == 5:
                    obs, _, terminated, truncated, info = step_result
                    if terminated or truncated:
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

        done = terminated or truncated
        return obs, reward, done, info

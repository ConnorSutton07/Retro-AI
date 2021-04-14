import gym
import numpy as np 
import cv2
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import RIGHT_ONLY
import collections 

class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env = None, n: int = 4) -> None:
        """
        Return only every nth frame

        """
        super(MaxAndSkipEnv, self).__init__(env)
        self._obs_buffer = collections.deque(maxlen = 2) # most recent raw observations for max pooling across time steps
        self._skip = n

    def step(self, action):
        total_reward = 0.0
        done = None 
        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            self._obs_buffer.append(obs)
            total_reward += reward 
            if done:
                break 
        max_frame = np.max(np.stack(self._obs_buffer), axis=0)

    def reset(self):
        """ Clear past frame buffer, init to firs obs """
        self._obs_buffer.clear()
        obs = self.env.reset()
        self._obs_buffer.append(obs)
        return obs

class ProcessFrame84(gym.ObservationWrapper):
    """
    Downsamples image to 84x84
    Converts to grayscale

    """
    def __init__(self, env = None, shape: tuple = (84, 84, 1), dtype=np.uint8):
        super(ProcessFrame84, self).__init__(env)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=shape, dtype=dtype)

    def observation(self, obs):
        return ProcessFrame84.process(obs)

    @staticmethod
    def process(frame):
        if frame.size == 240 * 256 * 3:
            img = np.reshape(frame, [240, 256, 3]).astype(np.float32)
        else:
            assert False, "Unknown resolution."
        img = img[:, :, 0] * 0.299 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.114
        resized_screen = cv2.resize(img, (84, 110), interpolation=cv2.INTER_AREA)
        x_t = resized_screen[18:102, :]
        x_t = np.reshape(x_t, [84, 84, 1]) # see if can change to self.shape
        return x_t.astype(np.uint8) # change to self.dtype

class ImageToPytorch(gym.ObservationWrapper):
    def __init__(self, env):
        super(ImageToPytorch, self).__init__(env)
        old_shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0.0, 
            high=1.0,
            shape=(old_shape[-1], old_shape[0], old_shape[1]),
            dtype=np.float32
        )
    def observation(self, observation):
        return np.moveaxis(observation, 2, 0)

class ScaledFloatFrame(gym.ObservationWrapper):
    """ 
    Normalize pixel values in frame
    [0, 255] -> [0, 1]

    """
    def observation(self, obs):
        return np.array(obs).astype(np.float32) / 255.0

class BufferWrapper(gym.ObservationWrapper):
    def __init__(self, env, n_steps, dtype=np.float32):
        super(BufferWrapper, self).__init__(env)
        self.dtype = dtype
        old_space = env.observation_space
        self.observation_space = gym.spaces.Box(
            old_space.low.repeat(n_steps, axis=0),
            old_space.high.repeat(n_steps, axis=0),
            dtype=dtype
        )

    def reset(self):
        self.buffer = np.zeros_like(self.observation_space.low, dtype=self.dtype)
        return self.observation(self.env.reset())

    def observation(self, observation):
        self.buffer[:-1] = self.buffer[1:]
        self.buffer[-1] = observation
        return self.buffer

def make_env(env):
    env = MaxAndSkipEnv(env) # repeat actions over 4 frames
    env = ProcessFrame84(env) # reduce frame size to 84x84
    env = ImageToPytorch(env) # convert frame to PyTorch tensor
    env = BufferWrapper(env, 4) # buffer only collects every 4th frame
    env = ScaledFloatFrame(env) # normalize pixel values between 0 & 1
    return JoypadSpace(env, RIGHT_ONLY) # reduce action-space to 5 (agent cannot move left)


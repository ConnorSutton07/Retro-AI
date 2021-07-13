
"""
Wrapper for stable baselines PPO

"""
from __future__ import annotations
from stable_baselines3 import PPO

class PPOAgent(PPO):
    def __init__(self,
                 env:           Gym.Env,
                 save_path:     str,
                 policy:        str = 'MlpPolicy',
                 gamma:         float = 0.99,
                 n_steps:       int   = 2048,
                 ent_coef:      float = 0.01,
                 learning_rate: float = 0.00025,
                 vf_coef:       float = 0.5,
                 max_grad_norm: float = 0.5,
                 lam:           float = 0.95,
                 batch_size:    int   = 64,
                 noptepochs:    int   = 4,
                 verbose:       int   = 1,
                 timesteps:     int   = 5000):

        super(PPOAgent, self).__init__(policy = policy,
                                       env = env)
        #self.policy = policy
        #self.env = env
        self.n_steps = n_steps
        self.ent_coef = ent_coef
        self.learning_rate = learning_rate
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.lam = 0.95
        self.batch_size = batch_size
        self.noptepochs = noptepochs
        self.verbose = verbose
        self.save_path = save_path
        self.timesteps = timesteps
        
    def learn(self):
        super(PPOAgent, self).learn(self.timesteps)

    def save(self):
        super(PPOAgent, self).save(self.save_path)

    @staticmethod
    def load(load_path, env):
        return PPO.load(load_path, env = env)
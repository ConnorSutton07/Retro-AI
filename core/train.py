from __future__ import annotations
from core.training.dqn import DQNAgent
from core.training.ppo import PPOAgent
from core.environment import make_env, make_envs
from core.config import * 
from core.reward import Rewards, default_reward_function
from stable_baselines3.common.vec_env import DummyVecEnv
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
import pickle
import numpy as np
import retro
import cv2
import keyboard
import sys

def run(training_mode: bool, 
        pretrained:    bool,
        num_episodes:  int,
        save_path:     str,
        load_path:     str = None,
        game:          str = 'DonkeyKongCountry-Snes',
        alg:           str = "DQN"
        ) -> None:

    print("Creating environment...")
    if pretrained:
        env = make_env(game)
        PPO_Evaluation(load_path, env)
    elif alg == "DQN":
        env = make_env(game)
        DQN_Training(num_episodes, save_path, load_path, env, pretrained, training_mode)
    elif alg == "PPO":
        env = make_envs(game)
        PPO_Training(num_episodes, save_path, env)

def PPO_Training(num_episodes: int,
                 save_path:    str,
                 env:          Gym.Env):
    """
    Begins training session using Principal Policy Optimization
    Uses Stable-Baselines implementation
    Multiprocessing enabled

    """
    agent = PPOAgent(env = env, save_path = save_path, timesteps = num_episodes)
    agent.learn()
    agent.save()

def PPO_Evaluation(load_path: str,
                   env:       Gym.Env):

    agent = PPOAgent.load(load_path, env)
    env.reset()

    for i in range(10):
        obs = env.reset()
        done = False
        total_reward = 0
        while not done:
            env.render()
            action, state = agent.predict(obs)
            obs, reward, done, info = env.step(action)
            total_reward += reward
        print("Total Reward: ", total_reward)
    env.close()

def DQN_Training(num_episodes:  int,
                 save_path:     str,
                 load_path:     str,
                 env:           Gym.Env,
                 pretrained:    bool,
                 training_mode: bool):
    """
    Begins training seession using Double Deep-Q Learning algorithm.
    Custom implementation

    """
    print("Initializing agent...")
    observation_space = env.observation_space.shape
    action_space = env.action_space.n
    agent = DQNAgent(
        state_space = observation_space,
        action_space = action_space,
        pretrained = pretrained,
        max_memory_size = MAX_MEMORY_SIZE,
        batch_size = BATCH_SIZE,
        gamma = GAMMA,
        learning_rate = LEARNING_RATE,
        dropout = DROPOUT,
        exploration_max = EXPLORATION_MAX,
        exploration_min = EXPLORATION_MIN,
        exploration_decay = EXPLORATION_DECAY,
        double_dq = True,
        load_path = load_path,
        save_path = save_path)

    env.reset()
    total_rewards = []
    force_quit = False
    iterator = tqdm(range(num_episodes))
    print("Starting...")
    
    for ep_num in iterator:
        state = env.reset()
        state = torch.Tensor([state])
        total_reward = 0
        steps = 0
        counter = 0
        terminal = False
        prev_reward = 0
        prev_info = None
        while not terminal:
            #if not training_mode:
            env.render()

            action = agent.act(state)
            steps += 1
            state_next, reward, terminal, info = env.step(int(action[0]))
            #reward, terminal = reward_function(info, prev_info)
            total_reward += reward

            counter = counter + 1 if prev_reward >= total_reward else 0
            if counter > 200:
                terminal = True

            state_next = torch.Tensor([state_next])
            reward = torch.tensor([reward]).unsqueeze(0)
            terminal_t = torch.tensor([int(terminal)]).unsqueeze(0)

            if training_mode:
                agent.remember(state, action, reward, state_next, terminal_t)
                agent.experience_replay()

            prev_reward = total_reward
            state = state_next
            prev_info = info.copy()

            if keyboard.is_pressed('esc'):
                print("Exiting...")
                force_quit = True
                iterator.close()
                sys.exit()
                break

            
        total_rewards.append(total_reward)
        print("Total reward after episode {} is {}".format(ep_num + 1, total_rewards[-1]))
        num_episodes += 1      

    if not training_mode:
        env.render(close=True)

    if not force_quit:  
        if training_mode:
            print("Saving model...")
            agent.save(total_rewards)
        if num_episodes > 500:
            plt.title("Episodes trained vs. Average Rewards (per 500 eps)")
            plt.plot([0 for _ in range(500)] + 
                    np.convolve(total_rewards, np.ones((500,))/500, mode="valid").tolist())
            plt.show()


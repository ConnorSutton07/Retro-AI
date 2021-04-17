import gym_super_mario_bros
from core.agent import DQNAgent
from core.environment import make_env
from core.config import * 
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
import pickle
import numpy as np


def run(training_mode: bool, 
        pretrained: bool,
        session_path: str, 
        session_name: str,
        num_episodes: int,
        game: str = 'SuperMarioBros-2-1-v0' 
        ) -> None:

    env = gym_super_mario_bros.make(game)
    env = make_env(env)
    observation_space = env.observation_space.shape
    action_space = env.action_space.n 
    agent = DQNAgent(
        state_space = observation_space,
        action_space = action_space,
        max_memory_size = MAX_MEMORY_SIZE,
        batch_size = BATCH_SIZE,
        gamma = GAMMA,
        learning_rate = LEARNING_RATE,
        dropout = DROPOUT,
        exploration_max = EXPLORATION_MAX,
        exploration_min = EXPLORATION_MIN,
        exploration_decay = EXPLORATION_DECAY,
        double_dq = True,
        pretrained = pretrained)

    env.reset()
    total_rewards = []

    for ep_num in tqdm(range(num_episodes)):
        state = env.reset()
        state = torch.Tensor([state])
        total_reward = 0
        steps = 0
        terminal = False
        while not terminal:
            if not training_mode:
                #show_state(env, ep_num)
                env.render()
            action = agent.act(state)
            steps += 1

            state_next, reward, terminal, info = env.step(int(action[0]))
            total_reward += reward
            state_next = torch.Tensor([state_next])
            reward = torch.tensor([reward]).unsqueeze(0)
            terminal = torch.tensor([int(terminal)]).unsqueeze(0)

            if training_mode:
                agent.remember(state, action, reward, state_next, terminal)
                agent.experience_replay()

            state = state_next
            
        total_rewards.append(total_reward)
        print("Total reward after episode {} is {}".format(ep_num + 1, total_rewards[-1]))
        num_episodes += 1      

    if training_mode:
        with open("ending_position.pkl", "wb") as f:
            pickle.dump(agent.ending_position, f)
        with open("num_in_queue.pkl", "wb") as f:
            pickle.dump(agent.num_in_queue, f)
        with open("total_rewards.pkl", "wb") as f:
            pickle.dump(total_rewards, f)
        if agent.double_dq:
            torch.save(agent.local_net.state_dict(), "dq1.pt")
            torch.save(agent.target_net.state_dict(), "dq2.pt")
        else:
            torch.save(agent.dqn.state_dict(), "dq.pt")  
        torch.save(agent.STATE_MEM,  "STATE_MEM.pt")
        torch.save(agent.ACTION_MEM, "ACTION_MEM.pt")
        torch.save(agent.REWARD_MEM, "REWARD_MEM.pt")
        torch.save(agent.STATE2_MEM, "STATE2_MEM.pt")
        torch.save(agent.DONE_MEM,   "DONE_MEM.pt")
    
    env.close()  

    if num_episodes > 500:
        plt.title("Episodes trained vs. Average Rewards (per 500 eps)")
        plt.plot([0 for _ in range(500)] + 
                 np.convolve(total_rewards, np.ones((500,))/500, mode="valid").tolist())
        plt.show()
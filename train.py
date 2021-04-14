import gym_super_mario_bros
from agent import DQNAgent
from environment import make_env
from tqdm import tqdm
import torch


def run(training_mode, num_episodes: int, pretrained: bool):
    env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')
    env = make_env(env)
    observation_space = env.observation_space.shape
    action_space = env.action_space.n 
    agent = DQNAgent(
        state_space = observation_space,
        action_space = action_space,
        max_memory_size = 30000,
        batch_size = 32,
        gamma = 0.90,
        learning_rate = 0.00025,
        dropout = 0.0,
        exploration_max = 1.0,
        exploration_min = 0.02,
        exploration_decay = 0.99,
        double_dq = True,
        pretrained = pretrained
        )

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
                show_state(env, ep_num)
            action = agent.act(state)
            steps += 1

            state_next, reward, terminal, info = env.step(int(action[0]))
            total_reward += reward
            state_next = torch.tensor([state_next])
            reward = torch.tensor([reward]).unsqueeze(0)
            terminal = torch.tensor([int(terminal)]).unsqueeze(0)

            if training_mode:
                agent.remember(state, action, reward, state_next, terminal)
                agent.experience_replay()
            
        total_rewards.append(total_reward)

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


def show_state(env, ep=0, info=""):
    plt.figure(3)
    plt.clf()
    plt.imshow(env.render(mode='rgb_array'))
    plt.title("Episode: %d %s" % (ep, info))
    plt.axis('off')

    display.clear_output(wait=True)
    display.display(plt.gcf())

def vectorize_action(action, action_space):
    # Given a scalar action, return a one-hot encoded action
    
    return [0 for _ in range(action)] + [1] + [0 for _ in range(action + 1, action_space)]
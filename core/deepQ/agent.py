from core.deepQ.solver import DQNSolver
import torch 
import torch.nn as nn
import pickle
import random
import os

class DQNAgent:

    def __init__(self, 
                 state_space, 
                 action_space, 
                 max_memory_size: int, 
                 batch_size: int, 
                 gamma: float, 
                 learning_rate: float,
                 dropout: float, 
                 exploration_max: float, 
                 exploration_min: float, 
                 exploration_decay: float,
                 double_dq: bool = True, 
                 pretrained: bool = False,
                 load_path: str = None,
                 save_path: str = None
                 ) -> None:

        # Define DQN Layers
        self.state_space = state_space
        self.action_space = action_space
        self.double_dq = double_dq
        self.pretrained = pretrained
        self.load_path = load_path
    
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if self.double_dq:  
            self.local_net = DQNSolver(state_space, action_space).to(self.device)
            self.target_net = DQNSolver(state_space, action_space).to(self.device)
            
            if self.pretrained:
                self.local_net.load_state_dict(torch.load(os.path.join(self.load_path, "dq1.pt"), map_location=torch.device(self.device)))
                self.target_net.load_state_dict(torch.load(os.path.join(self.load_path, "dq2.pt"), map_location=torch.device(self.device)))
                    
            self.optimizer = torch.optim.Adam(self.local_net.parameters(), lr=learning_rate)
            self.copy = 5000  # Copy the local model weights into the target network every 5000 steps
            self.step = 0
        else:  
            self.dqn = DQNSolver(state_space, action_space).to(self.device)
            
            if self.pretrained:
                self.dqn.load_state_dict(torch.load(os.path.join(self.load_path, "dq.pt"), map_location=torch.device(self.device)))
            self.optimizer = torch.optim.Adam(self.dqn.parameters(), lr=learning_rate)

        # Create memory
        self.max_memory_size = max_memory_size
        if self.pretrained:
            self.STATE_MEM = torch.load(os.path.join(load_path, "STATE_MEM.pt"))
            self.ACTION_MEM = torch.load(os.path.join(load_path, "ACTION_MEM.pt"))
            self.REWARD_MEM = torch.load(os.path.join(load_path, "REWARD_MEM.pt"))
            self.STATE2_MEM = torch.load(os.path.join(load_path, "STATE2_MEM.pt"))
            self.DONE_MEM = torch.load(os.path.join(load_path, "DONE_MEM.pt"))
            with open(os.path.join(load_path, "ending_position.pkl"), 'rb') as f:
                self.ending_position = pickle.load(f)
            with open(os.path.join(load_path, "num_in_queue.pkl"), 'rb') as f:
                self.num_in_queue = pickle.load(f)
        else:
            self.STATE_MEM = torch.zeros(max_memory_size, *self.state_space)
            self.ACTION_MEM = torch.zeros(max_memory_size, 1)
            self.REWARD_MEM = torch.zeros(max_memory_size, 1)
            self.STATE2_MEM = torch.zeros(max_memory_size, *self.state_space)
            self.DONE_MEM = torch.zeros(max_memory_size, 1)
            self.ending_position = 0
            self.num_in_queue = 0
        
        self.memory_sample_size = batch_size
        
        # Learning parameters
        self.gamma = gamma
        self.l1 = nn.SmoothL1Loss().to(self.device) # Also known as Huber loss
        self.exploration_max = exploration_max
        self.exploration_rate = exploration_max
        if self.pretrained:
            self.exploration_rate = 0.05
        self.exploration_min = exploration_min
        self.exploration_decay = exploration_decay

    def remember(self, state, action, reward, state2, done):
        self.STATE_MEM[self.ending_position] = state.float()
        self.ACTION_MEM[self.ending_position] = action.float()
        self.REWARD_MEM[self.ending_position] = reward.float()
        self.STATE2_MEM[self.ending_position] = state2.float()
        self.DONE_MEM[self.ending_position] = done.float()
        self.ending_position = (self.ending_position + 1) % self.max_memory_size  # FIFO tensor
        self.num_in_queue = min(self.num_in_queue + 1, self.max_memory_size)
        
    def recall(self):
        # Randomly sample 'batch size' experiences
        idx = random.choices(range(self.num_in_queue), k=self.memory_sample_size)
        
        STATE = self.STATE_MEM[idx]
        ACTION = self.ACTION_MEM[idx]
        REWARD = self.REWARD_MEM[idx]
        STATE2 = self.STATE2_MEM[idx]
        DONE = self.DONE_MEM[idx]
        
        return STATE, ACTION, REWARD, STATE2, DONE

    def act(self, state):
        # Epsilon-greedy action

        if random.random() < self.exploration_rate:  
            return torch.tensor([[random.randrange(self.action_space)]])
        if self.double_dq:
            # Local net is used for the policy
            self.step +=1
            return torch.argmax(self.local_net(state.to(self.device))).unsqueeze(0).unsqueeze(0).cpu()
        else:
            return torch.argmax(self.dqn(state.to(self.device))).unsqueeze(0).unsqueeze(0).cpu()

    def copy_model(self):
        # Copy local net weights into target net
        
        self.target_net.load_state_dict(self.local_net.state_dict())
    
    def experience_replay(self):
        
        if self.double_dq and self.step % self.copy == 0:
            self.copy_model()

        if self.memory_sample_size > self.num_in_queue:
            return

        STATE, ACTION, REWARD, STATE2, DONE = self.recall()
        STATE = STATE.to(self.device)
        ACTION = ACTION.to(self.device)
        REWARD = REWARD.to(self.device)
        STATE2 = STATE2.to(self.device)
        DONE = DONE.to(self.device)
        
        self.optimizer.zero_grad()
        if self.double_dq:
            # Double Q-Learning target is Q*(S, A) <- r + γ max_a Q_target(S', a)
            target = REWARD + torch.mul((self.gamma * 
                                        self.target_net(STATE2).max(1).values.unsqueeze(1)), 
                                        1 - DONE)

            current = self.local_net(STATE).gather(1, ACTION.long()) # Local net approximation of Q-value
        else:
            # Q-Learning target is Q*(S, A) <- r + γ max_a Q(S', a) 
            target = REWARD + torch.mul((self.gamma * 
                                        self.dqn(STATE2).max(1).values.unsqueeze(1)), 
                                        1 - DONE)
                
            current = self.dqn(STATE).gather(1, ACTION.long())
        
        loss = self.l1(current, target)
        loss.backward() # Compute gradients
        self.optimizer.step() # Backpropagate error

        self.exploration_rate *= self.exploration_decay
        
        # Makes sure that exploration rate is always at least 'exploration min'
        self.exploration_rate = max(self.exploration_rate, self.exploration_min)

    def save(self) -> None:

        with open(os.path.join(self.save_path, "ending_position.pkl"), "wb") as f:
            pickle.dump(agent.ending_position, f)
        with open(os.path.join(self.save_path, "num_in_queue.pkl"), "wb") as f:
            pickle.dump(agent.num_in_queue, f)
        with open(os.path.join(self.save_path, "total_rewards.pkl"), "wb") as f:
            pickle.dump(total_rewards, f)
        if agent.double_dq:
            torch.save(agent.local_net.state_dict(), os.path.join(self.save_path, "dq1.pt"))
            torch.save(agent.target_net.state_dict(), os.path.join(self.save_path, "dq2.pt"))
        else:
            torch.save(agent.dqn.state_dict(), os.path.join(self.save_path, "dq.pt"))
        torch.save(agent.STATE_MEM, os.path.join(self.save_path, "STATE_MEM.pt"))
        torch.save(agent.ACTION_MEM, os.path.join(self.save_path, "ACTION_MEM.pt"))
        torch.save(agent.REWARD_MEM, os.path.join(self.save_path, "REWARD_MEM.pt"))
        torch.save(agent.STATE2_MEM, os.path.join(self.save_path, "STATE2_MEM.pt"))
        torch.save(agent.DONE_MEM, os.path.join(self.save_path, "DONE_MEM.pt"))

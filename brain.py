import numpy as np
from snake_rl.envs.snake_env import SnakeEnv
import random
from Game.experience import Experience
import time
import pygame
from net import Net
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

class Brain:
    def __init__(self, learning_rate, discount_rate, eps_start, eps_end, eps_decay, memory_size, batch_size, max_episodes, max_steps, target_update):
        self.memory = []
        self.push_count = 0
        self.learning_rate = learning_rate
        self.discount_rate = discount_rate
        self.eps_start = eps_start
        self.current_eps = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.max_steps = max_steps
        self.max_episodes = max_episodes
        self.current_episode = 1
        self.policy_model = None
        self.replay_model = None
        self.target_update = target_update
        pygame.init()
        self.screen = pygame.display.set_mode((600, 600))
        pygame.display.set_caption("Snake")       
    
    def build_model(self):
        self.policy_model = Net()
        self.replay_model = self.policy_model
    
    def decay_epsilon(self, episode):
        self.current_eps = self.eps_end + (self.eps_start - self.eps_end) * np.exp(-self.eps_decay * episode)

    def push_memory(self, new_memory):
        if(len(self.memory) < self.memory_size):
            self.memory.append(new_memory)
        else:
            self.memory[self.push_count % self.memory_size] = new_memory
        self.push_count += 1
    
    def sample_memory(self):
        return random.sample(self.memory, self.batch_size)

    def can_sample_memory(self):
        return len(self.memory) >= self.batch_size

    #1 forward pass and backprop
    def fit(self, x, y):
        self.policy_model.zero_grad()
        optimizer = optim.Adam(self.policy_model.parameters(), lr = self.learning_rate)
        criterion = nn.MSELoss()
        loss = criterion(x, y)
        loss.backward()
        optimizer.step()

    def train(self):
        self.build_model()
        for episode in range(self.max_episodes):
            self.current_episode = episode
            env = SnakeEnv(self.screen)
            episode_reward = 0
            for timestep in range(self.max_steps):
                env.render(self.screen)
                state = env.get_state()
                action = None
                epsilon = self.current_eps
                if epsilon > random.random():
                    action = np.random.choice(env.action_space) #explore
                else:
                    with torch.no_grad():
                        values = self.policy_model(Variable(torch.from_numpy(env.get_state()).float())) #exploit
                    action = np.argmax(values)
                experience = env.step(action)
                if(experience['done'] == True):
                    episode_reward += experience['reward']
                    break
                episode_reward += experience['reward']
                self.push_memory(Experience(experience['state'], experience['action'], experience['reward'], experience['next_state']))
                self.decay_epsilon(episode)
                if self.can_sample_memory():
                    memory_sample = self.sample_memory()
                    for memory in memory_sample:
                        memstate = memory.state
                        realq = self.policy_model(Variable(torch.from_numpy(memstate).float()))
                        action = memory.action
                        next_state = memory.next_state
                        reward = memory.reward
                        max_q = reward + (self.discount_rate * self.replay_model(Variable(torch.from_numpy(next_state).float()))) #bellman equation
                        self.fit(realq, max_q)
            print("Episode: ", episode, " Total Reward: ", episode_reward)
            if episode % self.target_update == 0:
                self.replay_model.set_weights(self.policy_model)
        self.policy_model.save_weights('weights.hdf5')
        pygame.quit()

    def render(self):
        self.env.render(self.screen)

    def choose_action(self, state):
        q_values = self.policy_model.predict(state)
        action = np.amax(q_values)
        return action

    def load(self):
        self.build_model()
        self.policy_model.load_weights("weights.hdf5")

    def play(self):
        for episode in range(100):
            env = SnakeEnv(self.screen)
            for timestep in range(1000):
                env.render(self.screen)
                pred = self.policy_model.predict(env.get_state())
                print(np.array(pred))
                action = np.amax(pred)
                d = env.step(action)
                if(d['done'] == True):
                    break
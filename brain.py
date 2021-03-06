import numpy as np
import tensorflow as tf
from snake_rl.envs.snake_env import SnakeEnv
import random
from Game.experience import Experience
import time
import pygame
from PIL import Image
from keras import Sequential
from keras.layers import Conv2D, Dense, MaxPool2D, Activation, Flatten
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


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
        self.policy_model = Sequential()
        self.policy_model.add(Conv2D(32, (3, 3), padding = 'same', activation = 'relu', data_format = "channels_last", input_shape = (600, 600, 3)))
        self.policy_model.add(MaxPool2D(pool_size = (2, 2)))
        self.policy_model.add(Conv2D(64, (3, 3), padding="same", activation="relu"))
        self.policy_model.add(MaxPool2D(pool_size=(2, 2)))
        self.policy_model.add(Conv2D(128, (3, 3), padding="same", activation="relu"))
        self.policy_model.add(MaxPool2D(pool_size=(2, 2)))
        self.policy_model.add(Flatten())
        self.policy_model.add(Dense(32, activation = "relu"))
        self.policy_model.add(Dense(5, activation = "softmax"))
        self.policy_model.compile(optimizer = 'rmsprop', loss = 'mean_squared_error')

        self.replay_model = Sequential()
        self.replay_model.add(Conv2D(32, (3, 3), padding = 'same', activation = 'relu', data_format = "channels_last", input_shape = (600, 600, 3)))
        self.replay_model.add(MaxPool2D(pool_size = (2, 2)))
        self.replay_model.add(Conv2D(64, (3, 3), padding="same", activation="relu"))
        self.replay_model.add(MaxPool2D(pool_size=(2, 2)))
        self.replay_model.add(Conv2D(128, (3, 3), padding="same", activation="relu"))
        self.replay_model.add(MaxPool2D(pool_size=(2, 2)))
        self.replay_model.add(Flatten())
        self.replay_model.add(Dense(32, activation = "relu"))
        self.replay_model.add(Dense(5, activation = "softmax"))
        self.replay_model.compile(optimizer = 'rmsprop', loss = 'mean_squared_error')


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

    def screenshot(self):
        data = pygame.image.tostring(self.screen, 'RGB')
        image = Image.frombytes('RGB', (600, 600), data)
        matrix = np.asarray(image.getdata(), dtype=np.uint8)
        print(matrix.shape)
        matrix = (matrix - 128)/(128 - 1)
        matrix = np.reshape(matrix, (1, 600, 600, 2))
        return matrix

    def train(self):
        tf.logging.set_verbosity(tf.logging.ERROR)
        self.build_model()
        for episode in range(self.max_episodes):
            self.current_episode = episode
            env = SnakeEnv(self.screen)
            episode_reward = 0
            for timestep in range(self.max_steps):
                env.render(self.screen)
                state = self.screenshot()
                #state = env.get_state()
                action = None
                epsilon = self.current_eps
                if epsilon > random.random():
                    action = np.random.choice(env.action_space) #explore
                else:
                    values = self.policy_model.predict(state) #exploit
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
                        realq = self.policy_model.predict(memstate)
                        action = memory.action
                        next_state = memory.next_state
                        reward = memory.reward
                        max_q = reward + (self.discount_rate * self.replay_model.predict(next_state)) #bellman equation
                        self.policy_model.fit(memstate, max_q, verbose = 0)
            print("Episode: ", episode, " Total Reward: ", episode_reward)
            if episode % self.target_update == 0:
                self.replay_model.set_weights(self.policy_model.get_weights())
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

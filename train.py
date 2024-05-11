from brain import Brain
from game import Game
import numpy as np
import torch
import random
import os
from collections import deque

epsilon_start = 0.99
epsilon_final = 0.01
epsilon_decay = 0.0001
lr = 1e-4
batch_size = 32
max_timesteps = int(1e8)
learn_every = 4
update_every = 2
episode_reward = 0
curr_episode = 0
save_every = 10
moving_average_length = 25

game = Game()
brain = Brain(dim = game.dim, action_size = 4)
prev_frame = torch.tensor(game.get_frame())
game.step(game.DOWN)
curr_frame = torch.tensor(game.get_frame())

def get_epsilon(step):
	return epsilon_final + (epsilon_start - epsilon_final) * np.exp(-epsilon_decay * step)

moving_average = deque(maxlen = moving_average_length)

for timestep in range(max_timesteps):
	state = torch.stack([prev_frame, curr_frame])
	action = 0
	epsilon = get_epsilon(timestep)
	if(epsilon < random.uniform(0, 1)):
		action = brain.pick_action(state)
	else:
		action = random.randint(0, 3)

	(reward, terminal) = game.step(action)
	game.render()
	episode_reward += reward

	next_frame = torch.tensor(game.get_frame())
	next_state = torch.stack([curr_frame, next_frame])
	brain.add_to_memory(state, action, reward, next_state, terminal)
	prev_frame = curr_frame
	curr_frame = next_frame

	if(timestep % learn_every == 0):
		brain.learn()
	if(timestep % update_every == 0):
		brain.update()
	
	if(terminal):
		moving_average.append(len(game.snake))
		debug = 'Episode ' + str(curr_episode) + ' snake length ' + str(len(game.snake))
		if(len(moving_average) >= moving_average_length):
			debug += ' moving average ' + str(float(sum(moving_average)) / float(len(moving_average)))
		print(debug)
		curr_episode += 1
		if(curr_episode % save_every == 0):
			brain.save_model(os.path.join(os.path.dirname(__file__), 'model'))
		episode_reward = 0
		game = Game()
		prev_frame = torch.tensor(game.get_frame())
		game.step(game.DOWN)
		curr_frame = torch.tensor(game.get_frame())


brain.save_model(os.path.join(os.path.dirname(__file__), 'model'))

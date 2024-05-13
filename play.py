from brain import Brain
from game import Game
import torch
import os
import time

g = Game()
b = Brain(dim = g.dim, action_size = 4)
b.load_model(os.path.join(os.path.dirname(__file__), 'model'))
tau = 0.01

prev_frame = torch.tensor(g.get_frame())
g.step(g.DOWN)
curr_frame = torch.tensor(g.get_frame())

while True:
	state = torch.stack([prev_frame, curr_frame])
	action = b.pick_action_boltzmann(state, tau)
	(reward, terminal) = g.step(action)
	g.render()
	prev_frame = curr_frame	
	curr_frame = torch.tensor(g.get_frame())

	if(terminal):
		g = Game()
		prev_frame = torch.tensor(g.get_frame())
		g.step(g.DOWN)
		curr_frame = torch.tensor(g.get_frame())

	time.sleep(0.1)

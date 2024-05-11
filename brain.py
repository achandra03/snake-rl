import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from memory import Memory

class Net(nn.Module):
	
	def __init__(self, dim, action_size):
		super(Net, self).__init__()
		self.conv1 = nn.Conv2d(in_channels=2, out_channels=128, kernel_size=3, stride=1) 
		self.norm1 = nn.BatchNorm2d(128)
		self.conv2 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1) 
		self.norm2 = nn.BatchNorm2d(256)
		self.conv3 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1)
		self.flatten = 256 * 14 * 14
		self.fc1 = nn.Linear(self.flatten, 100)
		self.drop1 = nn.Dropout(0.25)
		self.fc2 = nn.Linear(100, action_size)

	def forward(self, x):
		x = x.type(torch.FloatTensor)
		x = self.conv1(x)
		x = F.relu(x)
		x = self.conv2(x)
		x = F.relu(x)
		x = self.conv3(x)
		x = F.relu(x)
		x = x.view(-1, self.flatten)
		x = self.fc1(x)
		x = F.relu(x)
		x = self.drop1(x)
		x = self.fc2(x)
		return x

class Brain():

	def __init__(self, dim, action_size, batch_size = 32, lr = 1e-4, gamma = 0.99):
		self.q_net = Net(dim, action_size)
		self.target_net = Net(dim, action_size)
		self.memory = Memory(batch_size)
		self.lr = lr
		self.batch_size = batch_size
		self.gamma = gamma
		self.optimizer = optim.RMSprop(self.q_net.parameters(), lr=self.lr)

	def pick_action(self, state):
		q_values = self.q_net.forward(state)
		maxes = torch.max(q_values.view(-1), 0)
		return maxes[1].item()

	def add_to_memory(self, state, action, reward, next_state, terminal):
		self.memory.add_sample(state, action, reward, next_state, terminal)

	def can_learn(self):
		return self.memory.can_sample()

	def learn(self):
		if(not self.can_learn()):
			return
		samples = self.memory.sample()
	
		states = torch.stack([samples[i][0] for i in range(len(samples))])
		actions = torch.tensor([samples[i][1] for i in range(len(samples))])
		rewards = torch.tensor([samples[i][2] for i in range(len(samples))])
		next_states = torch.stack([samples[i][3] for i in range(len(samples))])
		terminals = torch.tensor([samples[i][4] for i in range(len(samples))])

		with torch.no_grad():
			target_action_values = self.target_net(next_states).detach()
			max_action_values = target_action_values.amax(1, keepdim=True)
			terminal_y = (terminals * rewards)
			nonterminal_y = (1 - terminals) * (rewards + (self.gamma * max_action_values).squeeze(1))
			target_q_values = (terminal_y + nonterminal_y).unsqueeze(1)

		q_predictions = self.q_net(states)
		actions = actions.unsqueeze(1)
		q_expected = q_predictions.gather(1, actions)
		loss = F.huber_loss(target_q_values, q_expected)
		self.optimizer.zero_grad()
		loss.backward()
		self.optimizer.step()
		


	def update(self):
		for name, param in self.q_net.named_parameters():
			if(name in self.target_net.state_dict()):
				try:
					self.target_net.state_dict()[name].copy_(param.data)
				except Exception as e:
					print('could not copy from qnet to target net')
					
	def save_model(self, path):
		torch.save(self.q_net.state_dict(), path)

	def load_model(self, path):
		self.q_net.load_state_dict(torch.load(path))
		self.q_net.eval()


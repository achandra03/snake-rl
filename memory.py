import numpy as np
import random
from collections import deque

class Memory():

	def __init__(self, batch_size, capacity=1e7):
		self.batch_size = batch_size
		self.capacity = capacity
		self.samples = deque(maxlen = int(capacity))

	def can_sample(self):
		return len(self.samples) >= self.batch_size

	def sample(self):
		return random.sample(self.samples, self.batch_size)

	def add_sample(self, state, action, reward, next_state, terminal):
		self.samples.append((state, action, reward, next_state, terminal))

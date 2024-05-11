import pygame
import random
import time

class Game():
	
	def __init__(self, dim = 20, scale_factor = 10):
		self.snake = [{'x': 0, 'y': 0}]
		self.UP = 0
		self.RIGHT = 1
		self.DOWN = 2
		self.LEFT = 3
		self.direction = self.DOWN #0 - up  1 - right  2 - down  3 - left
		self.dirs = [{'x': 0, 'y': -1}, {'x': 1, 'y': 0}, {'x': 0, 'y': 1}, {'x': -1, 'y': 0}]
		self.dim = dim
		self.food = {'x': random.randint(5, dim - 1), 'y': random.randint(5, dim - 1)}
		self.scale_factor = scale_factor
		self.screen = pygame.display.set_mode((dim * scale_factor, dim * scale_factor))

	def render(self):
		self.screen.fill((0, 0, 0))
		for segment in self.snake:
			pygame.draw.rect(self.screen, (255, 255, 255), (segment['x'] * self.scale_factor, segment['y'] * self.scale_factor, self.scale_factor, self.scale_factor))
		pygame.draw.rect(self.screen, (255, 0, 0), (self.food['x'] * self.scale_factor, self.food['y'] * self.scale_factor, self.scale_factor, self.scale_factor))
		pygame.display.flip()

	#returns (reward, terminal)
	def step(self, action):
		self.change_direction(action)
		prev_position = self.snake[0].copy()
		self.snake[0]['x'] += self.dirs[self.direction]['x']
		self.snake[0]['y'] += self.dirs[self.direction]['y']

		if(self.snake[0]['x'] < 0 or self.snake[0]['y'] < 0 or self.snake[0]['x'] >= self.dim or self.snake[0]['y'] >= self.dim): #crashes into wall
			return (-10, 1)
	
		reward = 0
		terminal = 0

		old_dist = abs(prev_position['x'] - self.food['x']) + abs(prev_position['y'] - self.food['y'])
		new_dist = abs(self.snake[0]['x'] - self.food['x']) + abs(self.snake[0]['y'] - self.food['y'])
		
		if(self.snake[0]['x'] == self.food['x'] and self.snake[0]['y'] == self.food['y']): #eats food
			reward = 10
			self.snake.append(prev_position)
			self.food = {'x': random.randint(0, self.dim - 1), 'y': random.randint(0, self.dim - 1)}
			while(self.food in self.snake):
				self.food = {'x': random.randint(0, self.dim - 1), 'y': random.randint(0, self.dim - 1)}

		else:
			reward = -0.2

		'''
		elif(new_dist < old_dist): #snake gets closer to food
			reward += 0.1
		elif(new_dist > old_dist): #snake gets farther from food
			reward -= 0.1
		'''
		for i in range(1, len(self.snake)):
			tmp = self.snake[i]
			self.snake[i] = prev_position
			prev_position = tmp

		if(self.snake[0] in self.snake[1:]): #crashes into itself
			reward = -10
			terminal = 1

		return (reward, terminal)

	def change_direction(self, new_direction):
		if(new_direction == self.UP and self.direction != self.DOWN):
			self.direction = self.UP
		if(new_direction == self.RIGHT and self.direction != self.LEFT):
			self.direction = self.RIGHT
		if(new_direction == self.DOWN and self.direction != self.UP):
			self.direction = self.DOWN
		if(new_direction == self.LEFT and self.direction != self.RIGHT):
			self.direction = self.LEFT

	def get_frame(self):
		grid = [[0 for i in range(self.dim)] for j in range(self.dim)]
		if(self.snake[0]['y'] >= 0 and self.snake[0]['y'] < self.dim and self.snake[0]['x'] >= 0 and self.snake[0]['x'] < self.dim):
			grid[self.snake[0]['y']][self.snake[0]['x']] = -1
		for segment in self.snake[1:]:
			grid[segment['y']][segment['x']] = -0.5
		grid[self.food['y']][self.food['x']] = 1
		return grid

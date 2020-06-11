from Game.Snake import Snake
from Game.Food import Food

import pygame
import random
import numpy as np
import sys
from PIL import Image


class SnakeEnv():

    def create_food(self):
        x = random.randint(0, 19)
        while x == self.snake.head.x:
            x = random.randint(0, 19)
        y = random.randint(0, 19)
        while y == self.snake.head.y:
            y = random.randint(0, 19)
        self.food = Food(self.screen, x * 30, y * 30)
        if(self.snake.board[x, y] == 1 or self.snake.board[x, y] == 2):
            self.create_food()
        else:
            self.snake.update_food(x, y)
        self.snake.board[x, y] = 5

    def __init__(self, screen):
        self.action_space = np.array([0, 1, 2, 3, 4])
        self.state = None
        pygame.init()
        self.screen = screen
        self.snake = Snake(self.screen)
        self.create_food()
        self.state = self.snake.board
        self.total_reward = 0

    def reset(self):
        self.__init__()
    
    def screenshot(self):
        data = pygame.image.tostring(self.screen, 'RGB')
        image = Image.frombytes('RGB', (600, 600), data)
        matrix = np.asarray(image.getdata(), dtype=np.uint8)
        matrix = (matrix - 128)/(128 - 1)
        matrix = np.reshape(matrix, (1, 600, 600, 3))
        return matrix
    
    def step(self, action):
        print("step")
        d = dict()
        d['state'] = self.screenshot()
        self.snake.move(action)
        reward = 0
        done = False
        if(self.snake.head.x == self.food.x and self.snake.head.y == self.food.y):
            self.create_food()
            self.snake.add_body()
            reward += 1
        else:
            lost = self.snake.check_loss()
            if lost == 1:
                reward = -1
                done = True
            else:
                #reward = -0.5 
                pass
        self.total_reward += reward
        d['action'] = action
        d['reward'] = reward
        self.state = self.snake.board
        d['next_state'] = self.screenshot()
        d['done'] = done
        return d
    
    
    def get_state(self):
        return np.reshape(self.snake.board, (400, 1)).T / 5

    def render(self):
        self.screen.fill((0, 0, 0))
        self.food.render()
        self.snake.render()
        pygame.display.flip()

    def close(self):
        pygame.quit()
        
    def move(self, key):
        self.snake.move(key)
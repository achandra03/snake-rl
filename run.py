import snake_rl
import pygame
import random
from brain import Brain
from snake_rl.envs.snake_env import SnakeEnv
import keras
from keras import models
from keras.layers import Dense
from keras import losses
import os
"""
env = SnakeEnv()

done = False
while not done:
  env.render()
  pygame.event.pump()
  for event in pygame.event.get():
    if event.type == pygame.QUIT:
        done = True
    
    if event.type == pygame.KEYDOWN:
      done = False
      key = event.key
      if key == pygame.K_UP:
        done = env.step(0)
      elif key == pygame.K_RIGHT:
        done = env.step(1)
      elif key == pygame.K_DOWN:
        done = env.step(2)
      elif key == pygame.K_LEFT:
        done = env.step(3)
      elif key == pygame.K_a:
        done = env.step(pygame.K_a)
      done = done['done']
      if done == True:
        pygame.quit()
"""

os.environ["SDL_VIDEODRIVER"] = "dummy"
learning_rate = 0.5
discount_rate = 0.99
eps_start = 1
eps_end = .01
eps_decay = .001
memory_size = 100000
batch_size = 256
max_episodes = 1000
max_steps = 5000
target_update = 10
b = Brain(learning_rate, discount_rate, eps_start, eps_end, eps_decay, memory_size, batch_size, max_episodes, max_steps, target_update)
b.train()



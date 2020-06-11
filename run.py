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

(width, height) = (600, 600)
screen = pygame.display.set_mode((width, height))
env = SnakeEnv(screen)

done = False
while not done:
  env.render()
  for event in pygame.event.get():
    if event.type == pygame.QUIT:
      break
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
      if done['done'] == True:
        pygame.quit()
    env.render()




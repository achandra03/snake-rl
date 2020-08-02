import snake_rl
import pygame
from pygame.locals import *
import random
from brain import Brain
from snake_rl.envs.snake_env import SnakeEnv
import keras
from keras import models
from keras.layers import Dense
from keras import losses
import os
import neat

(width, height) = (600, 600)
screen = pygame.display.set_mode((width, height))
if __name__ == '__main__':
    env = SnakeEnv(screen)
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'conf.txt')
    env.run(config_path)

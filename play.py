import pickle
from snake_rl.envs.snake_env import SnakeEnv
from Game.Snake import Snake
import pygame

(width, height) = (600, 600)
screen = pygame.display.set_mode((width, height))
env = SnakeEnv(screen)
snake = Snake(screen)
net = pickle.load(open("best.pkl", "rb"))
while not snake.done:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            break
    inputs = snake.get_input()
    snake_x = inputs["snake_x"]
    snake_y = inputs["snake_y"]
    food_x = inputs["food_x"]
    food_y = inputs["food_y"]
    food_vert = inputs["food_vert"]
    food_horz = inputs["food_horz"]
    wall_vert = inputs["wall_vert"]
    wall_horz = inputs["wall_horz"]
    body_front = inputs["body_front"]

    output = round(3 * net.activate((food_vert, food_horz, wall_vert, wall_horz, body_front))[0], 0)
    env.step(snake, output)
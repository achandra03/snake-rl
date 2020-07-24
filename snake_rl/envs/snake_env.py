from Game.Snake import Snake
from Game.Food import Food
import neat
import os
import pygame
import random
import numpy as np
import sys
from PIL import Image
import pickle


class SnakeEnv():

    def __init__(self, screen):
        self.action_space = np.array([0, 1, 2, 3])
        self.state = None
        pygame.init()
        self.screen = screen
        self.snakes = [] 
        self.total_reward = 0

    def reset(self):
        self.__init__()
    
        
    def get_state(self):
        return np.reshape(self.snake.board, (400, 1)).T / 5

    def render(self, snake):
        self.screen.fill((0, 0, 0))
        snake.food.render()
        snake.render()
        pygame.display.flip()
    
    def step(self, snake, action):
        snake.move(action)
        self.render(snake)

    def close(self):
        pygame.quit()


    def eval_genomes(self, genomes, config):
        global nets_g
        nets_g = []
        nets = []
        snakes = []
        global ge_g
        ge_g = []
        ge = []
        for genome_id, genome in genomes:
            genome.fitness = 0
            net = neat.nn.FeedForwardNetwork.create(genome, config)
            nets.append(net)
            snakes.append(Snake(self.screen))
            ge.append(genome)
        
        ge_g = ge.copy()
        nets_g = nets.copy()
        run = True
        #Main loop
        while run and len(snakes) > 0:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    run = False
                    pygame.quit()
                    quit()
                    break

            for x, snake in enumerate(snakes):
                ge[x].fitness += 0.1

                """
                Inputs to the neural net:
                Vertical distance from food to head
                Horizontal distance from food to head
                Vertical distance to nearest wall from head
                Horizontal distance to nearest wall from head
                Distance from head to body segment (default -1)
                """

                snake_x = snake.head.x
                snake_y = snake.head.y
                food_x = snake.food.x 
                food_y = snake.food.y 

                food_vert = snake_y - food_y
                food_horz = snake_x - food_x
                wall_vert = min(snake_y, 600 - snake_y)
                wall_horz = min(snake_x, 600 - snake_x)
                body_front = snake.body_front()
                output = round(3 * nets[snakes.index(snake)].activate((food_vert, food_horz, wall_vert, wall_horz, body_front))[0], 0)
                state = snake.move(output)
                if state["Food"] == True:
                    ge[snakes.index(snake)].fitness += 1

                if state["Died"] == True:
                    ge[snakes.index(snake)].fitness -= 1
                    nets.pop(snakes.index(snake))
                    ge.pop(snakes.index(snake))
                    snakes.pop(snakes.index(snake))


    def run(self, config_file):
        config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config_file)
        population = neat.Population(config)
        population.add_reporter(neat.StdOutReporter(True))
        stats = neat.StatisticsReporter()
        population.add_reporter(stats)
        best = population.run(self.eval_genomes, 200)
        print('\nBest genome:\n{!s}'.format(best))
        best_net = nets_g[ge_g.index(best)]
        pickle.dump(best_net, open('best.pkl', 'wb'))



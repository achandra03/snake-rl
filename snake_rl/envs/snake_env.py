from Game.Snake import Snake
from Game.Food import Food
import neat
import pygame
import random
import numpy as np
import sys
from PIL import Image


class SnakeEnv():

    def create_food(self):
        x = random.randint(0, 19)
        y = random.randint(0, 19)
        while(self.snake.board[x, y] == 1 or self.snake.board[x, y] == 2):
            x = random.randint(0, 19)
            y = random.randint(0, 19)
        self.food = Food(self.screen, x * 30, y * 30)
        self.snake.update_food(x, y)
        self.snake.board[x, y] = 5

    def __init__(self, screen):
        self.action_space = np.array([0, 1, 2, 3])
        self.state = None
        pygame.init()
        self.screen = screen
        self.snake = Snake(self.screen)
        self.snakes = []
        self.create_food()
        self.state = self.snake.board
        self.total_reward = 0

    def reset(self):
        self.__init__()
    
    
    
    def move(self, snake, action):
        snake.move(action)
        done = False
        if(snake.head.x == self.food.x and snake.head.y == self.food.y):
            self.create_food()
            snake.add_body()
        else:
            lost = snake.check_loss()
            if lost == 1:
                done = True
    
    
    
    def get_state(self):
        return np.reshape(self.snake.board, (400, 1)).T / 5

    def render(self):
        self.screen.fill((0, 0, 0))
        self.food.render()
        self.snake.render()
        for r in self.snake.List:
            pygame.display.update(r.rect)

    def close(self):
        pygame.quit()


    def eval_genomes(genomes, config):
        nets = []
        snakes = []
        ge = []

        for g in genomes:
            net = neat.nn.FeedForwardNetwork(g, config)
            nets.append(net)
            snakes.append(Snake(self.screen))
            g.fitness = 0
            ge.append(g)
        
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

                snake_x = snake.x
                snake_y = snake.y
                food_x = snake.food.x 
                food_y = snake.food.y 

                food_vert = snake_y - food_y
                food_horz = snake_x - food_x
                wall_vert = min(snake_y, 600 - snake_y)
                wall_horz = min(snake_x, 600 - snake_x)
                body_front = snake.body_front()

                output = round(4 * nets[snakes.index(snake)].activate((food_vert, food_horz, wall_vert, wall_horz, body_front)), 0)
                move(snake, output)

                



    def run():
        config_file_path = os.path.dirname("/Users/arnav/Desktop/snake_rl/conf.txt")
        config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation,config_file)
        population = neat.Population(config)
        population.add_reporter(neat.StdOutReporter(True))
        stats = neat.StatisticsReporter()
        population.add_reporter(stats)
        best = population.run(eval_genomes, 50)
        print('\nBest genome:\n{!s}'.format(best))

import pygame
from Game.Body import Body
import numpy as np

class Snake:

    def __init__(self, window):
        self.head = Body(window, 0, 0, None)
        self.List = [self.head]
        self.window = window
        self.board = np.zeros((20, 20))
        self.board.fill(0.1)
        self.board[0, 0]= 2
        self.food_x = 0
        self.food_y = 0

    def render(self):
        for body in self.List:
            body.render()
    
    def update_board(self):
        self.board.fill(0.1)
        self.board[self.food_x, self.food_y] = 5
        for body in self.List:
            body.move()
            x = body.y // 30
            y = body.x // 30
            if(body.ahead is None):
                try:
                    self.board[x, y] = 2
                except:
                    pass
            else:
                self.board[x, y] = 1

    def move(self, key):
        if key == 3 and self.List[0].direction != 1:
            self.List[0].direction = 3
        elif key == 0 and self.List[0].direction != 2:
            self.List[0].direction = 0
        elif key == 1 and self.List[0].direction != 3:
            self.List[0].direction = 1
        elif key == 2 and self.List[0].direction != 0:
            self.List[0].direction = 2
        elif key == pygame.K_a:
            print(self.head.old_x)
            print(self.head.old_y)
            return
        self.update_board()

    def add_body(self):
        x = self.List[len(self.List) - 1].old_y
        y = self.List[len(self.List) - 1].old_x
        b = Body(self.window, x, y, self.List[len(self.List) - 1])
        self.List.append(b)
        direction = self.List[len(self.List) - 2].direction
        b.set_direction(direction)

    def check_loss(self):
        if(self.head.x >= 600 or self.head.x < 0 or self.head.y >= 600 or self.head.y < 0):
            #pygame.quit()
            return 1
        else:
            for body in self.List:
                if body is not self.head and body.x == self.head.x and body.y == self.head.y:
                    #pygame.quit()
                    return 1
        return 0

    def update_food(self, x, y):
        self.food_x = y
        self.food_y = x

    def take_action(self, action):
        if action == 0:
            self.move(pygame.K_UP)
        elif action == 1:
            self.move(pygame.K_RIGHT)
        elif action == 2:
            self.move(pygame.K_DOWN)
        elif action == 3:
            self.move(pygame.K_LEFT)


        
        

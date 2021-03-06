import pygame
import random
from Game.Body import Body
from Game.Food import Food
import numpy as np

class Snake:


    def create_food(self):
        x = random.randint(0, 19)
        y = random.randint(0, 19)
        while(self.board[x, y] == 1 or self.board[x, y] == 2):
            x = random.randint(0, 19)
            y = random.randint(0, 19)
        food = Food(self.window, x * 30, y * 30)
        self.update_food(x, y)
        self.board[x, y] = 5
        return food

    def __init__(self, window):
        self.head = Body(window, 0, 0, None)
        self.List = [self.head]
        self.window = window
        self.board = np.zeros((20, 20))
        self.board.fill(0.1)
        self.board[0, 0]= 2
        self.food = self.create_food()
        self.done = False

    def render(self):
        self.food.render()
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
        self.update_board()
        ate_food = False
        if(self.head.x == self.food.x and self.head.y == self.food.y):
            self.food = self.create_food()
            self.add_body()
            ate_food = True
        else:
            lost = self.check_loss()
            if lost == 1:
                self.done = True
        return {"Died": self.done, "Food": ate_food}

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

    def body_front(self):
        direction = self.head.direction
        if direction == 0:
            x = self.head.x // 30
            head_y = (self.head.y // 30)
            y = head_y - 1
            while(y >= 0):
                if(self.board[x][y] == 1):
                    return (head_y - y) * 30
                y = y - 1

        elif direction == 1:
            head_x = self.head.x // 30
            x = head_x + 1
            y = self.head.y // 30
            while(x < 20):
                if(self.board[x][y] == 1):
                    return (x - head_x) * 30
                x = x + 1

        elif direction == 2:
            x = self.head.x // 30
            head_y = (self.head.y // 30)
            y = head_y + 1
            while(y < 20):
                if(self.board[x][y] == 1):
                    return (y - head_y) * 30
                y = y + 1

        elif direction == 3:
            head_x = self.head.x // 30
            x = head_x - 1
            y = self.head.y // 30
            while(x >= 0):
                if(self.board[x][y] == 1):
                    return (head_x - x) * 30
                x = x - 1

        return -1

    def get_input(self):
        dict = {}
        dict["snake_x"] = self.head.x
        dict["snake_y"] = self.head.y
        dict["food_x"] = self.food.x 
        dict["food_y"] = self.food.y 

        dict["food_vert"] = self.head.y - self.food.y
        dict["food_horz"] = self.head.x - self.food.x
        dict["wall_vert"] = min(self.head.y, 600 - self.head.y)
        dict["wall_horz"] = min(self.head.x, 600 - self.head.x)
        dict["body_front"] = self.body_front()
        return dict





        
        

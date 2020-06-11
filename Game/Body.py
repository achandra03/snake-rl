import pygame
from pygame import Rect

class Body:
    """
    0 - up
    1 - right
    2 - down
    3 - left
    """
    def __init__(self, window, x, y, ahead):
        self.direction = 2
        self.window = window
        self.x = x
        self.y = y
        self.old_x = x
        self.old_y = y
        self.rect = Rect(x, y, 30, 30)
        self.ahead = ahead

    def render(self):
        pygame.draw.rect(self.window, (255, 255, 255), (self.x, self.y, 30, 30))

    def set_direction(self, direction):
        self.direction = direction

    def move(self):
        if(self.ahead is None):
            self.old_x = self.y
            self.old_y = self.x
            if self.direction == 0:
                self.y -= 30
            elif self.direction == 1:
                self.x += 30
            elif self.direction == 2:
                self.y += 30
            elif self.direction == 3:
                self.x -= 30
        else:
            self.old_x = self.y
            self.old_y = self.x
            self.direction = self.ahead.direction
            self.x = self.ahead.old_y
            self.y = self.ahead.old_x

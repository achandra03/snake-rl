import pygame

class Food:

    def __init__(self, window, x, y):
        self.window = window
        self.x = x
        self.y = y

    def render(self):
        pygame.draw.rect(self.window, (255, 0, 0), (self.x, self.y, 30, 30))
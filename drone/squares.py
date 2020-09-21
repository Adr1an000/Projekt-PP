import threading
import time

import pygame


class Squares:
    def __init__(self, screen, freq, color, filling, place):
        self.screen = screen
        self.freq = freq
        self.color = color
        self.filling = filling
        self.place = place
    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self.__display)
        self.thread.start()

    def __display(self):
        onTime = (1/self.freq)*self.filling
        offTime = (1/self.freq) - onTime
        self.visable = False
        while self.running:
            self.visable=True
            time.sleep(onTime)
            self.visable=False
            time.sleep(offTime)

    def draw(self):
        if self.visable:
            pygame.draw.rect(self.screen, self.color, self.place)


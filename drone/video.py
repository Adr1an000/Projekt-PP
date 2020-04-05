import pygame
import cv2
from pyardrone import ARDrone
from pyardrone.video import VideoClient

import logging


def main():
   # pygame.init()
    W, H = 640, 360
    #screen = pygame.display.set_mode((W, H))
    #drone = ARDrone()
    logging.basicConfig(level=logging.DEBUG)
    client = VideoClient('192.168.1.1', 5555)
    client.connect()
    client.video_ready.wait()

    try:
       while True:
           cv2.imshow('im', client.frame)
           if cv2.waitKey(10) == ord(' '):
               break
    finally:
        print("Shutting down...")
        client.close()
        drone.close()
    print("Ok.")


if __name__ == '__main__':
    main()
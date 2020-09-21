
import pygame
import ardrone

from drone import gui
from drone.video import Video
from drone.brain import load_model
import drone.dataSource as dataSource
import numpy as np


def main(win):
    pygame.init()
    datasource = dataSource.DataSource()
    model = load_model()
    drone = ardrone.ARDrone()
    if drone == None:
        print("not connect to drone")
        return -1
        gui.main()
    drone.speed = 0.7
    v = Video(screen = win)
    running = True
    while running:
        temp = datasource.readData()
        pre = np.argmax(model.predict(temp))
        print(pre)
        k = pygame.key.get_pressed()
        v.video()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False 
            elif event.type == pygame.KEYUP:
                drone.hover()
            elif event.type == pygame.key.get_pressed():
                if event.type == pygame.K_t:
                    if pre == 1:
                        drone.move_forward()
                    elif pre == 2:
                        drone.turn_left()
                    elif pre == 3:
                        drone.turn_right()
        if k[pygame.K_DELETE]:
            running = False
        elif k[pygame.K_ESCAPE]:
            drone.reset()
            running = False
                # takeoff / land
        elif k[pygame.K_q]:
             drone.takeoff()
        elif k[pygame.K_SPACE]:
             drone.land()
        elif k[pygame.K_BACKSPACE]:
             drone.reset()
             # forward / backward
        elif k[pygame.K_w] :
             drone.move_forward()

        elif k[pygame.K_s]:
             drone.move_backward()
             # left / right
        elif k[pygame.K_a]:
             drone.move_left()
        elif k[pygame.K_d]:
             drone.move_right()
        # up / down
        elif k[pygame.K_UP]:
             drone.move_up()
        elif k[pygame.K_DOWN]:
             drone.move_down()
             # turn left / turn right
        elif k[pygame.K_LEFT]:
             drone.turn_left()
        elif k[pygame.K_RIGHT]:
             drone.turn_right()


    print("Shutting down...")
    drone.halt()
    print("Ok.")

if __name__ == '__main__':
    main(win=pygame.display.set_mode((800, 600)) )
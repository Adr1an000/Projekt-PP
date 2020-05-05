
import pygame
import ardrone

from drone.video import Video


def main():
    pygame.init()

    drone = ardrone.ARDrone()

    drone.speed = 0.7
    v = Video();
  #  drone.set_cam(screen)

    running = True
    while running:
        v.video()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False 
            elif event.type == pygame.KEYUP:
                drone.hover()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_DELETE:
                    running = False
                    pygame.quit();
                    quit(0)
                elif event.key == pygame.K_ESCAPE:
                    drone.reset()
                    running = False
                # takeoff / land
                elif event.key == pygame.K_q:
                    drone.takeoff()
                elif event.key == pygame.K_SPACE:
                    drone.land()
                # emergency
                elif event.key == pygame.K_BACKSPACE:
                    drone.reset()
                # forward / backward
                elif event.key == pygame.K_w:
                    drone.move_forward()
                elif event.key == pygame.K_s:
                    drone.move_backward()
                # left / right
                elif event.key == pygame.K_a:
                    drone.move_left()
                elif event.key == pygame.K_d:
                    drone.move_right()
                # up / down
                elif event.key == pygame.K_UP:
                    drone.move_up()
                elif event.key == pygame.K_DOWN:
                    drone.move_down()
                # turn left / turn right
                elif event.key == pygame.K_LEFT:
                    drone.turn_left()
                elif event.key == pygame.K_RIGHT:
                    drone.turn_right()






    print("Shutting down...")
    drone.halt()
    print("Ok.")

if __name__ == '__main__':
    main()

import pygame
import pyardrone
import logging

from pyardrone.video import VideoClient


def main():
    logging.basicConfig(level=logging.DEBUG)
    pygame.init()
    W, H = 640, 480
    screen = pygame.display.set_mode((W, H))
    drone = pyardrone.ARDrone()
    clock = pygame.time.Clock()
    client = VideoClient('192.168.1.1', 5555)
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False 
            elif event.type == pygame.KEYUP:
                drone.hover()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    #reset nie dziala
                    running = False
                # takeoff / land
                elif event.key == pygame.K_RETURN:
                    drone.takeoff()
                elif event.key == pygame.K_SPACE:
                    drone.land()
                # emergency
                elif event.key == pygame.K_BACKSPACE:
                    drone.emergency()
                # forward / backward
                elif event.key == pygame.K_w:
                    drone.move(forward=1)
                elif event.key == pygame.K_s:
                    drone.move(backward=1)
                # left / right
                elif event.key == pygame.K_a:
                    drone.move(left=1)
                elif event.key == pygame.K_d:
                    drone.move(right=1)
                # up / down
                elif event.key == pygame.K_UP:
                    drone.move(up=1)
                elif event.key == pygame.K_DOWN:
                    drone.move(down=1)
                # turn left / turn right
                elif event.key == pygame.K_LEFT:
                    drone.move(cw=1)
                elif event.key == pygame.K_RIGHT:
                    drone.move(ccw=1)

        try:
            #pygame.image.frombuffer(image.tostring(), image.shape[:2],  "RGB")  // from stack
            surface = pygame.image.frombuffer(client.frame, (W, H), 'RGB')
            # battery status
            hud_color = (255, 0, 0) if drone.navdata.get('drone_state', dict()).get('emergency_mask', 1) else (10, 10, 255)
            bat = drone.navdata.get(0, dict()).get('battery', 0)
            f = pygame.font.Font(None, 20)
            hud = f.render('Battery: %i%%' % bat, True, hud_color)
            screen.blit(surface, (0, 0))
            screen.blit(hud, (10, 10))
        except:
            pass

        pygame.display.flip()
        clock.tick(50)
        pygame.display.set_caption("FPS: %.2f" % clock.get_fps())

    print("Shutting down...")
    drone.close()
    print("Ok.")

if __name__ == '__main__':
    main()
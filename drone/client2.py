
import pygame

from pyardrone import ARDrone
from pyardrone.at import base, parameters

from pyardrone.video import VideoClient

import logging

def main():
    pygame.init()
    W, H = 640, 360
    screen = pygame.display.set_mode((W, H))
    drone = ARDrone()
    logging.basicConfig(level=logging.DEBUG)
    client = VideoClient('192.168.1.1', 5555)
    client.connect()
    client.video_ready.wait()
    speed = parameters.Float()

    drone.navdata_ready.wait()
  #  drone.set_cam(screen)
    clock = pygame.time.Clock()
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYUP:
                drone.hover()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:

                    running = False
                # takeoff / land
                elif event.key == pygame.K_q:
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

            im = client.frame
            surface = pygame.image.fromstring(im.tobytes(), (360, 360),  im.mode)
            screen.blit(surface, (0, 0))
            # battery status
            hud_color = (255, 0, 0) if drone.navdata.get('drone_state', dict()).get('emergency_mask', 1) else (10, 10, 255)
            bat = drone.navdata.get('demo', dict()).get('battery', 0)
            alt = drone.navdata.get('demo', dict()).get('altitude', 1)*1000
            f = pygame.font.Font(None, 20)
            hud = f.render('Battery: %i%%' % bat, True, hud_color)
            screen.blit(hud, (10, 10))
            altitude = f.render( 'altitude: %5.3f' %alt, True, hud_color)
            screen.blit(altitude, (10, 30))
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
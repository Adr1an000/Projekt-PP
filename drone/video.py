import pygame
from ardrone import drone

from drone.temp import Temp


class Video:
    def __init__(self):
        self.f = pygame.font.Font(None, 20)

        self.W, self.H = 640, 360
        self.clock = pygame.time.Clock()
        self.screen = pygame.display.set_mode((self.W, self.H))
        self.up = Temp(self.screen, 1, (255, 255, 255), 0.5, (self.W/2-30, 20, 60, 60))
        self.down = Temp(self.screen, 0.25, (255, 255, 255), 0.5, (self.W/2-30, self.H-80, 60, 60))
        self.left = Temp(self.screen, 0.125, (255, 255, 255), 0.5, (20, self.H/2-30, 60, 60))
        self.right = Temp(self.screen, 0.5, (255, 255, 255), 0.5, (self.W-20/2-80, self.H/2-60, 60, 60))
        self.up.start()
        self.left.start()
        self.down.start()
        self.right.start()


    def video(self):

        try:
            self.screen.fill((0, 0, 0))
            im = drone.image
            surface = pygame.image.fromstring(im.tobytes(), im.size, im.mode)
            self.screen.blit(surface, (0, 0))
            # battery status
            hud_color = (255, 0, 0) if drone.navdata.get('drone_state', dict()).get('emergency_mask', 1) else (
            10, 10, 255)
            bat = drone.navdata.get('demo', dict()).get('battery', 0)
            alt = drone.navdata.get('demo', dict()).get('altitude', 1) * 1000
            hud = self.f.render('Battery: %i%%' % bat, True, hud_color)
            self.screen.blit(hud, (10, 10))
            altitude = self.f.render('altitude: %5.3f' % alt, True, hud_color)
            self.screen.blit(altitude, (10, 30))
        except:
            pass

        self.up.draw()
        self.left.draw()
        self.right.draw()
        self.down.draw()

        pygame.display.flip()
        self.clock.tick(60)
        pygame.display.set_caption("FPS: %.2f" % self.clock.get_fps())

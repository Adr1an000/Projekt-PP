import pygame
from ardrone import drone

from drone.temp import Temp


class Video:
    def __init__(self, screen = pygame.display.set_mode((640, 360)) ):
        self.f = pygame.font.Font(None, 20)
        self.W_drone, self.H_drone = 640, 360
        self.W, self.H = pygame.display.get_surface().get_size()
        self.clock = pygame.time.Clock()
        self.screen = screen
        self.begin_screen_W = (self.W- self.W_drone) / 2
        self.begin_screen_H = (self.H - self.H_drone) / 2
        self.rectangle_H = 60
        self.rectangle_W = 60
        self.up = Temp(self.screen, 8, (100, 100, 100), 0.5, ((self.W-self.rectangle_W)/2, self.rectangle_H/2, self.rectangle_W, self.rectangle_H))
        self.down = Temp(self.screen, 10, (100, 100, 100), 0.5, ((self.W-self.rectangle_W)/2, self.H-1.5*self.rectangle_H, self.rectangle_W, self.rectangle_H))
        self.left = Temp(self.screen, 12, (100, 100, 100), 0.5, (self.rectangle_W/2, (self.H-self.rectangle_H)/2, self.rectangle_W, self.rectangle_H))
        self.right = Temp(self.screen, 15, (100, 100, 100), 0.5, (self.W-1.5*self.rectangle_W, (self.H-self.rectangle_H)/2, self.rectangle_W, self.rectangle_H))
        self.up.start()
        self.left.start()
        self.down.start()
        self.right.start()
        self.direction =0


    def video(self):

        try:
            self.screen.fill((0, 0, 0))
            im = drone.image
            surface = pygame.image.fromstring(im.tobytes(), im.size, im.mode)
            self.screen.blit(surface, (self.begin_screen_H, self.begin_screen_W))
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

    def video_to_learn(self):
        self.screen.fill((0, 0, 0))
        if self.direction != 0:
            if self.direction == 1:
                self.up.draw()
            elif self.direction ==2:
               self.right.draw()
            elif self.direction ==3:
                self.down.draw()
            else:
                self.left.draw()

        pygame.display.flip()
        self.clock.tick(60)
        pygame.display.set_caption("FPS: %.2f" % self.clock.get_fps())




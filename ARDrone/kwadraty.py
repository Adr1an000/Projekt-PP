
import pygame
def main():
    pygame.init()
    W, H = 1200, 720
    screen = pygame.display.set_mode((W, H))

    clock = pygame.time.Clock()


    running = True
    while running:
        escape = True
        screen.fill((0, 0, 0))
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    i = 60
                    for k in range (0, 40):
                        pygame.draw.rect(screen, (200,20, 20), ( W/2-20, 20, 40, 40))
                        pygame.display.flip()
                        pygame.time.delay(i)
                        screen.fill((0,0,0))
                        pygame.display.flip()
                        pygame.time.delay(i*2)
                elif event.key == pygame.K_DOWN:
                    i = 70
                    for k in range(0, 40):
                        pygame.draw.rect(screen, (200, 20, 20), (W/2-20, H-60, 40, 40))
                        pygame.display.flip()
                        pygame.time.delay(i)
                        screen.fill((0, 0, 0))
                        pygame.display.flip()
                        pygame.time.delay(i*2)
                elif event.key == pygame.K_LEFT:
                    i = 80
                    for k in range(0, 40):
                        pygame.draw.rect(screen, (200, 20, 20), (20, H/2-20, 40, 40))
                        pygame.display.flip()
                        pygame.time.delay(i)
                        screen.fill((0, 0, 0))
                        pygame.display.flip()
                        pygame.time.delay(i*2)
                elif event.key == pygame.K_RIGHT:
                    i = 90
                    for k in range(0, 40):
                        pygame.draw.rect(screen, (200, 20, 20), (W-60, H/2-20, 40, 40))
                        pygame.display.flip()
                        pygame.time.delay(i)
                        screen.fill((0, 0, 0))
                        pygame.display.flip()
                        pygame.time.delay(i*2)
        clock.tick(50)
        pygame.display.set_caption("FPS: %.2f" % clock.get_fps())
        # screen.blit(surface, (0, 0))

    print("Shutting down...")

if __name__ == '__main__':
    main()
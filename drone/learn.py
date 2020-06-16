import pygame
from drone.video import Video


def main(win):
    v = Video(win)
    running = True
    while running:
        v.video_to_learn()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_DELETE:
                    running = False
                elif event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_UP:
                    v.direction = 1

                elif event.key == pygame.K_RIGHT:
                    v.direction = 2
                # turn left / turn right
                elif event.key == pygame.K_DOWN:
                    v.direction = 3
                elif event.key == pygame.K_LEFT:
                    v.direction = 4
                elif event.key == pygame.K_SPACE:
                    v.direction = 0
if __name__ == '__main__':
    main(win=pygame.display.set_mode((800, 600)) )
import pygame as pg

from drone import client, learn
from drone.button import Button

def main():
    pg.init()

    pg.display.set_caption('menu')
    win = pg.display.set_mode((800, 600))

    background = pg.Surface((800, 600))
    background.fill(pg.Color('#000000'))
    b_start = Button((0, 200, 20), 300, 200, 200, 100, "START")
    b_learn = Button((0, 200, 200), 300, 400, 200, 100, "LEARN")
    font_head = pg.font.SysFont('comicsans', 100)
    font_error = pg.font.SysFont('comicsans', 30)
    text_head=font_head.render('DRONE', False, (100, 100, 10))
    text_error=font_error.render('no connect to drone', False, (255, 10, 10))
    isConnect = True
    is_running = True

    while is_running:
        win.blit(background, (0, 0))
        pos = pg.mouse.get_pos()
        if b_start.isOver(pos):
            b_start.draw(win, outline=(255, 255, 255))
        elif b_learn.isOver(pos):
            b_learn.draw(win, outline=(255, 255, 255))
        for event in pg.event.get():
            if event.type == pg.QUIT:
                 is_running = False
            if event.type == pg.MOUSEBUTTONDOWN:
                if b_start.isOver(pos):
                    if(client.main(win)==-1):
                        isConnect = False
                    else:
                        isConnect = True
                if b_learn.isOver(pos):
                   learn.main(win)
        win.blit(text_head, (50, 50))
        if isConnect == False :
            win.blit(text_error, (450, 550))
        b_start.draw(win)
        b_learn.draw(win)
        pg.display.flip()
        pg.display.update()

if __name__ == '__main__':
    main()
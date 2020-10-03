import pygame


pygame.init()



display_scale = (600,600)
display = pygame.display.set_mode(display_scale)
pygame.display.set_caption("Game")

usr_width = 60
usr_height = 60
usr_x,usr_y = display_scale[0]/3,display_scale[1]-100
clock = pygame.time.Clock()
makeJump = False
jumpCounter = 10

def jump():
    global makeJump,jumpCounter,usr_y
    if jumpCounter >=-10:
        if jumpCounter <0:
            usr_y += (jumpCounter**2) /2
        else:
            usr_y -= (jumpCounter**2) /2
        jumpCounter -=1
    else:
        jumpCounter = 10
        makeJump = False
def run_game():
    global makeJump
    game = True


    while game:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        keys = pygame.key.get_pressed()
        if keys[pygame.K_SPACE]:
            makeJump=True

        if makeJump:
            jump()



        display.fill((255,255,255))
        pygame.draw.rect(display,(255,0,0),(usr_x,usr_y,usr_width,usr_height))
        pygame.display.update()
        clock.tick(60)


run_game()

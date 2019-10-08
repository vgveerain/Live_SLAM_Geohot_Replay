import os
import cv2
# import pygame
os.environ["PYSDL2_DLL_PATH"] = "C:\\Users\\vgvee\\AppData\\Local\\Programs\\Python\\Python37-32\\Lib\\site-packages"
import sdl2.ext

W,H = 1920//2,1080//2
# pygame.init()
# screen = pygame.display.set_mode((W,H))

# surface = pygame.Surface((W,H)).convert()

cv2.namedWindow('image', cv2.WINDOW_NORMAL)

sdl2.ext.init()
# window = sdl2.ext.Window("SDL2 Window", size=(W,H))
# window.show()

def process_frame(img):
    # cv2.waitKey()
    img = cv2.resize(img, (W,H))

    events = sdl2.ext.get_events()

    cv2.imshow('image',img)

    # surf = pygame.surfarray.make_surface(img.swapaxes(0,1)).convert()
    # print(img)
    # screen.blit(surf,(0,0))
    # pygame.display.flip()
    # print(img.shape)

if __name__ == "__main__":
    cap = cv2.VideoCapture("test_countryroad.mp4")
    while cap.isOpened():
        ret, frame = cap.read()
        if ret==True:
            process_frame(frame)
        else:
            break
import os
os.environ["PYSDL2_DLL_PATH"] = "C:\\Users\\vgvee\\AppData\\Local\\Programs\\Python\\Python37-32\\Lib\\site-packages"
import sdl2.ext

class Display(object):
    def __init__(self, W, H):
        sdl2.ext.init()
        self.W = W
        self.H = H
        self.window = sdl2.ext.Window("SDL2 Window", size=(W,H))
        self.window.show()
    
    def paint(self, img):
        events = sdl2.ext.get_events()
        for event in events:
            if event.type == sdl2.SDL_QUIT:
                exit(0)
        surf = sdl2.ext.pixels2d(self.window.get_surface())
        surf[:] = img.swapaxes(0,1)[:, :, 0]
        self.window.refresh()
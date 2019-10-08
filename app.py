import cv2
from display import Display
import numpy as np

W,H = 1920//2,1080//2
disp = Display(W,H)

class FeatureExtractor(object):
    GX=16//2
    GY=12//2
    
    def __init__(self):
        self.orb = cv2.ORB_create(1000)

    def extract(self, img):
        # sx = W//self.GX
        # sy = H//self.GY
        # akp = []
        # for ry in range(0, H, sy):
        #     for rx in range(0, W, sx):
        #         img_chunk = img[ry:ry+sy, rx:rx+sx]
        #         # print(img_chunk.shape)
        #         # kp, des = self.orb.detectAndCompute(img_chunk, None)
        #         kp = self.orb.detect(img_chunk, None)
        #         for p in kp:
        #             p.pt = (p.pt[0]+rx, p.pt[1]+ry)
        #             # print(p)
        #             akp.append(p)
        # return akp

        feats = cv2.goodFeaturesToTrack(np.mean(img, axis=2).astype(np.uint8), 3000, qualityLevel=0.01, minDistance=3)
        # print(feats)
        return feats

fe = FeatureExtractor()

def process_frame(img):
    img = cv2.resize(img, (W,H))
    kp1 = fe.extract(img)
    # cv2.drawKeypoints(img,kp1, img,color=(0,255,0), flags=0)
    for f in kp1:
        # print(f)
        u,v = map(lambda x : int(round(x)), f[0])
        cv2.circle(img, (u,v), 3, color=(0,255,0))
    disp.paint(img)

if __name__ == "__main__":
    cap = cv2.VideoCapture("test_countryroad.mp4")
    while cap.isOpened():
        ret, frame = cap.read()
        if ret==True:
            process_frame(frame)
        else:
            break
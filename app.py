import cv2
from display import Display
from extractor import Extractor

W,H = 1920//2,1080//2
disp = Display(W,H)

extractor = Extractor()

def process_frame(img):
    img = cv2.resize(img, (W,H))
    # kps, des, matches = extractor.extract(img)
    matches = extractor.extract(img)

    print("%d matches" % (len(matches)))

    if matches is None:
        return
    # for p in kps:
    #     u,v = map(lambda x : int(round(x)), p.pt)
    #     cv2.circle(img, (u,v), 3, color=(0,255,0))

    for pt1, pt2 in matches:
        u1,v1 = map(lambda x : int(round(x)), pt1.pt)
        u2,v2 = map(lambda x : int(round(x)), pt2.pt)
        cv2.circle(img, (u1,v1), 3, color=(0,255,0))
        cv2.line(img, (u1, v1), (u2, v2), color=(255,0,0))

    disp.paint(img)

if __name__ == "__main__":
    cap = cv2.VideoCapture("test_countryroad.mp4")
    while cap.isOpened():
        ret, frame = cap.read()
        if ret==True:
            process_frame(frame)
        else:
            break
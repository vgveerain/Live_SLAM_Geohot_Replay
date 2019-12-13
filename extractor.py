import cv2
import numpy as np
from skimage.measure import ransac
from skimage.transform import FundamentalMatrixTransform
from skimage.transform import EssentialMatrixTransform

# [x,y] => [x, y, 1]
def add_ones(x):
    return np.concatenate([x, np.ones((x.shape[0], 1))], axis=1)

# f_est_avg = []

def extractRt(E):
    W=np.mat([[0,-1,0],[1,0,0],[0,0,1]], dtype=float)
    U, d, VT = np.linalg.svd(E)
    assert np.linalg.det(U) > 0
    if np.linalg.det(VT) < 0:
        VT *= - 1.0
    R = np.dot(np.dot(U, W), VT)
    if np.sum(R.diagonal()) < 0:
        R = np.dot(np.dot(U, W.T), VT)
    t = U[:, 2]
    Rt = np.concatenate([R, t.reshape(3,1)], axis=1)
    return Rt

class Extractor(object):    
    def __init__(self, K):
        self.orb = cv2.ORB_create()
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.last = None
        self.K = K
        self.Kinv = np.linalg.inv(self.K)
    
    def normalize(self, pts):
        return np.dot(self.Kinv, add_ones(pts).T).T[:, 0:2]

    def denormalize(self, pt):
        # return int(round(pt[0]+self.w//2)), int(round(pt[1]+self.h//2))
        # return np.dot(self.Kinv, add_ones(pts).T).T[:, 0:2]
        ret = np.dot(self.K, np.array([pt[0], pt[1], 1.0]))
        # print(ret)
        #ret /= ret[2]
        return int(round(ret[0])), int(round(ret[1]))

    def extract(self, img):
        # detection
        feats = cv2.goodFeaturesToTrack(np.mean(img, axis=2).astype(np.uint8), 3000, qualityLevel=0.01, minDistance=3)

        # extraction
        kps = [cv2.KeyPoint(x=f[0][0], y=f[0][1], _size=20) for f in feats]
        kps, des = self.orb.compute(img, kps)

        # matching
        # matches = None
        # if self.last is not None:
        #     matches = self.bf.match(des, self.last["des"])
        #     matches = zip([kps[m.queryIdx] for m in matches], [self.last["kps"][m.trainIdx] for m in matches])

        ret = []
        if self.last is not None:
            matches = self.bf.knnMatch(des, self.last['des'], k=2)
            for m,n in matches:
                if m.distance < 0.75*n.distance:
                    kp1 = kps[m.queryIdx].pt
                    kp2 = self.last['kps'][m.trainIdx].pt
                    ret.append((kp1, kp2))

        # filter
        Rt = None
        if len(ret) > 0:
            ret = np.array(ret)

            # normalize co-ordinates
            # ret[:, :, 0] -= img.shape[0]//2
            # ret[:, :, 1] -= img.shape[1]//2
            ret[:, 0, :] = self.normalize(ret[:, 0, :])
            # print(ret[:, 0, :])
            ret[:, 1, :] = self.normalize(ret[:, 1, :])

            model, inliers = ransac((ret[:, 0], ret[:, 1]),
                                    EssentialMatrixTransform,
                                    # FundamentalMatrixTransform,
                                    min_samples=8,
                                    # residual_threshold=1,
                                    residual_threshold=0.005,
                                    max_trials=100)
            # print(sum(inliers), len(inliers))
            ret = ret[inliers]
            Rt = extractRt(model.params)
            print(Rt)

        #return    
        self.last = {"kps":kps, "des":des}
        return ret, Rt
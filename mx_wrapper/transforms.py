import cv2
import mxnet as mx
import numpy as np


class GaussianBlur(mx.gluon.block.Block):
    def __init__(self, min_ksize, max_ksize, min_sigma, max_sigma):
        super().__init__()
        self.min_ksize = min_ksize
        self.max_ksize = max_ksize
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma

    def forward(self, x):
        ksize = np.random.randint(self.min_ksize, self.max_ksize + 1)
        if ksize % 2 == 0:
            ksize += 1
        sigma = np.random.rand() * (self.max_sigma - self.min_sigma) + self.min_sigma
        x = x.asnumpy()
        x = cv2.GaussianBlur(x, (ksize, ksize), sigmaX=sigma, sigmaY=sigma)
        return mx.nd.array(x, dtype=np.uint8)


class SimpleMotionBlur(mx.gluon.block.Block):
    def __init__(self, min_ksize, max_ksize):
        super().__init__()
        self.min_ksize = min_ksize
        self.max_ksize = max_ksize

    def forward(self, x):
        ksize = np.random.randint(self.min_ksize, self.max_ksize + 1)
        kernel = np.zeros((ksize, ksize))

        direction = np.random.randint(0, 2)
        if direction == 0:
            kernel[:, (ksize - 1) // 2] = np.ones(ksize)
        else:
            kernel[(ksize - 1) // 2, :] = np.ones(ksize)
        kernel /= ksize
        x = x.asnumpy()
        x = cv2.filter2D(x, -1, kernel)
        return mx.nd.array(x, dtype=np.uint8)


class RandomBlur(mx.gluon.block.Block):
    def __init__(self):
        super().__init__()
        self.gaussian_blur = GaussianBlur(min_ksize=3, max_ksize=3, min_sigma=1.5, max_sigma=1.5)
        self.motion_blur = SimpleMotionBlur(min_ksize=5, max_ksize=10)

    def forward(self, x):
        p = np.random.rand()
        if p >= 0.5:
            return self.gaussian_blur(x)
        else:
            return self.motion_blur(x)


class Rotate(mx.gluon.block.Block):
    def __init__(self, max_angle=20, max_scale=1.2):
        super().__init__()
        self.max_angle = max_angle
        self.max_scale = max_scale

    def forward(self, x):
        angle = np.random.randint(-self.max_angle, self.max_angle + 1)
        scale = np.random.rand() * (self.max_scale - 1) + 1

        if angle == 0 and scale == 1:
            return x

        rows, cols = x.shape[0], x.shape[1]
        x = x.asnumpy()
        M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, scale)
        dst = cv2.warpAffine(x, M, (cols, rows))
        return mx.nd.array(dst, dtype=np.uint8)

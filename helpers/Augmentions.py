import cv2
import numpy as np
import random
from PIL import Image, ImageEnhance, ImageFilter


class FaceAugmentor:
    #Low light
    def low_light(self, face_bgr):
        face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(face_rgb)

        img = ImageEnhance.Brightness(img).enhance(0.45)
        img = ImageEnhance.Contrast(img).enhance(0.7)
        img = img.filter(ImageFilter.GaussianBlur(radius=1))

        return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    #Bright light
    def bright(self, face):
        return cv2.convertScaleAbs(face, alpha=1.3, beta=10)

    # Contrast 3shan lw camera mokhtlfa
    def contrast(self, face):
        return cv2.convertScaleAbs(face, alpha=1.2, beta=0)

    #Color jitter
    def color_jitter(self, face):
        hsv = cv2.cvtColor(face, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[..., 1] *= random.uniform(0.8, 1.2)
        hsv[..., 2] *= random.uniform(0.8, 1.2)
        hsv = np.clip(hsv, 0, 255).astype(np.uint8)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # ---- Noise (low quality webcam) ----
    def add_noise(self, face):
        noise = np.random.normal(0,0.9, face.shape).astype(np.uint8)
        return cv2.add(face, noise)


    #Rotation
    def rotate(self, face, angle=5):
        h, w = face.shape[:2]
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        return cv2.warpAffine(face, M, (w, h))

    #Partial crop
    def partial_crop(self, face):
        h, w = face.shape[:2]
        return face[int(0.15*h):int(0.9*h), int(0.1*w):int(0.9*w)]

    def generate(self, face):
        variants = [
            self.low_light(face),
            self.bright(face),
            self.contrast(face),
            self.color_jitter(face),
            self.add_noise(face),
            self.rotate(face, 25),
            self.rotate(face, -25),
            self.partial_crop(face),
        ]

        #random.shuffle(variants)
        return variants[:7]

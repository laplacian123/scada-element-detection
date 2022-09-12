import cv2
import random
import os
import numpy as np
import sys


def read_and_resize(src_img_path, size = None):
    img = cv2.imread(src_img_path)
    img = cv2.resize(img, size, cv2.INTER_CUBIC) if size else img
    return img


class ImgAugmenter:
    def __init__(self, img_class, src_img_path = None, img_size = None):
        self.img_class = img_class
        self.gen_dir = os.path.join("./gen_img", self.img_class)
        self.src_img_path = src_img_path or f"./src_img/{self.img_class}.jpg"
        self.img = read_and_resize(self.src_img_path)
        self.img_size = img_size or (self.img.shape[1], self.img.shape[0])
        self.counter = 0

        if not os.path.exists(self.gen_dir):
            os.makedirs(self.gen_dir)
        
    def rotate(self, img, max_angle = 180):
        """
        generate randomly rotated image with max rotate angle of max_angle in degrees
        """

        angle = int(random.uniform(-max_angle, max_angle))
        M = cv2.getRotationMatrix2D((int(self.img_size[0] / 2), int(self.img_size[1] / 2)), angle, 1)
        gen_img = cv2.warpAffine(img, M, (self.img_size[0], self.img_size[1]))
        return gen_img

    def zoom(self, img, h_frac = 0.5, v_frac = 0.5):
        """
        generate randomly zoomed image
        """
        h_value = random.uniform(h_frac, 1)
        v_value = random.uniform(v_frac, 1)
        h_taken = int(h_value * self.img_size[0])
        w_taken = int(v_value * self.img_size[1])
        h_start = random.randint(0, self.img_size[0] - h_taken)
        w_start = random.randint(0, self.img_size[1] - w_taken)
        gen_img = cv2.resize(img[h_start: h_start + h_taken, w_start: w_start + w_taken, :], self.img_size, cv2.INTER_CUBIC)
        return gen_img

    def gauss_noise(self, img, mean = 0, std = 0.3):
        gen_img = img.copy()
        gauss = np.random.normal(mean, std, (self.img_size[1], self.img_size[0], 3)).astype("uint8")
        gen_img = cv2.add(gen_img, gauss)
        return gen_img

    def sp_noise(self, img, prob = 0.05):
        gen_img = img.copy()
        if len(gen_img.shape) == 2:
            black = 0
            white = 255            
        else:
            colorspace = gen_img.shape[2]
            if colorspace == 3:  # RGB
                black = np.array([0, 0, 0], dtype='uint8')
                white = np.array([255, 255, 255], dtype='uint8')
            else:  # RGBA
                black = np.array([0, 0, 0, 255], dtype='uint8')
                white = np.array([255, 255, 255, 255], dtype='uint8')
        probs = np.random.random(gen_img.shape[:2])
        gen_img[probs < (prob / 2)] = black
        gen_img[probs > 1 - (prob / 2)] = white
        return gen_img

    def blur(self, img, ftype = "gauss", fsize = 9):
        """
        ftype: {"gaussian", "blur", "median"}
        """
        gen_img = img.copy()
        if ftype == "gaussian":
            return cv2.GaussianBlur(gen_img, (fsize, fsize), 0)
        if ftype == "blur":
            return cv2.blur(gen_img, (fsize, fsize))
        if ftype == "median":
            return cv2.medianBlur(gen_img, fsize)
        else:
            return gen_img

    def cutout(self, img, amount = 0.5):
        gen_img = img.copy()
        mask_width = random.randint(0, int(amount * self.img_size[0]))
        mask_height = random.randint(0, int(amount * self.img_size[1]))
        mask_x1 = random.randint(0, self.img_size[0] - mask_width)
        mask_y1 = random.randint(0, self.img_size[1] - mask_height)
        mask_x2 = mask_x1 + mask_width
        mask_y2 = mask_y1 + mask_height
        cv2.rectangle(gen_img, (mask_x1, mask_y1), (mask_x2, mask_y2), (0, 0, 0), thickness = -1)
        return gen_img

    def bright_jitter(self, img, max_value = 50):
        gen_img = img.copy()
        value = random.randint(-max_value, max_value)
        h, s, v = cv2.split(cv2.cvtColor(gen_img, cv2.COLOR_BGR2HSV))
        if value >= 0:
            lim = 255 - value
            v[v > lim] = 255
            v[v <= lim] += value
        else:
            lim = np.absolute(value)
            v[v < lim] = 0
            v[v >= lim] -= lim
        gen_img = cv2.cvtColor(cv2.merge((h, s, v)), cv2.COLOR_HSV2BGR)
        return gen_img

    def saturation_jitter(self, img, max_value = 50):
        gen_img = img.copy()
        value = random.randint(-max_value, max_value)
        h, s, v = cv2.split(cv2.cvtColor(gen_img, cv2.COLOR_BGR2HSV))
        if value >= 0:
            lim = 255 - value
            s[s > lim] = 255
            s[s <= lim] += value
        else:
            lim = np.absolute(value)
            s[s < lim] = 0
            s[s >= lim] -= lim
        gen_img = cv2.cvtColor(cv2.merge((h, s, v)), cv2.COLOR_HSV2BGR)
        return gen_img

    def contrast_jitter(self, img, brightness = 10, min_value = 40):
        contrast = random.randint(min_value, 100)
        gen_img = np.int16(img) 
        gen_img = np.uint8(np.clip(gen_img * int(contrast / 127 + 1) - contrast + brightness, 0, 255))
        return gen_img

    def augment_all(self, num_per_step, pipeline, pipeline_kwargs = {}):
        while pipeline:
            func_name = pipeline.pop(0)
            func_kwargs = pipeline_kwargs[func_name] if func_name in pipeline_kwargs else None
            if os.listdir(self.gen_dir):
                src_img_list = [os.path.join(self.gen_dir, fname) for fname in os.listdir(self.gen_dir)]
            else:
                src_img_list = [self.src_img_path]
            for img_path in src_img_list:
                img = read_and_resize(img_path, self.img_size)
                for _ in range(num_per_step):
                    func = getattr(self, func_name)
                    if func_kwargs:
                        gen_img = func(img, **func_kwargs)
                    else:
                        gen_img = func(img)

                    cv2.imwrite(os.path.join(self.gen_dir, f"{self.img_class}_{self.counter}.jpg"), gen_img)
                    self.counter += 1

class BgAugmenter(ImgAugmenter):
    def __init__(self, src_img_path = None, img_size = None):
        self.img_class = "background"
        self.gen_dir = os.path.join("./gen_img", self.img_class)
        self.img_size = img_size
        self.counter = 0

        if not os.path.exists(self.gen_dir):
            os.makedirs(self.gen_dir)      

    def augment_all(self, num_per_step, pipeline, pipeline_kwargs = {}):
        bg_list = []
        for a in os.listdir("./src_img"):
            if a[:2] == "bg":
                bg_list.append(a)
        for a in bg_list:
            src_img_path = os.path.join("./src_img", a)
            p = pipeline.copy()
            while p:
                func_name = p.pop(0)
                func_kwargs = pipeline_kwargs[func_name] if func_name in pipeline_kwargs else None
                img = read_and_resize(src_img_path, self.img_size)
                self.img_size = (img.shape[1], img.shape[0])
                for _ in range(num_per_step):
                    func = getattr(self, func_name)
                    if func_kwargs:
                        gen_img = func(img, **func_kwargs)
                    else:
                        gen_img = func(img)

                    cv2.imwrite(os.path.join(self.gen_dir, f"{self.img_class}_{self.counter}.jpg"), gen_img)
                    self.counter += 1





if __name__ == "__main__":
    # usage: python augment.py "tr" "cb" "mccb"
    cls_list = sys.argv[1:]
    for cls in cls_list:
        img = ImgAugmenter(cls)
        img.augment_all(3, ["blur", "bright_jitter", "saturation_jitter", "contrast_jitter", "gauss_noise", "cutout"])

    bg = BgAugmenter()
    bg.augment_all(20, ["blur", "zoom", "sp_noise", "bright_jitter", "saturation_jitter", "contrast_jitter", "gauss_noise", "cutout"])
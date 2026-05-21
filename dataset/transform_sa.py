#!/usr/bin/python
# -*- encoding: utf-8 -*-


from PIL import Image
import PIL.ImageEnhance as ImageEnhance
import random
random.seed(233)
import numpy as np
np.random.seed(233)

class pairColorJitter(object):
    def __init__(self, brightness=None, contrast=None, saturation=None, *args, **kwargs):
        if not brightness is None and brightness>0:
            self.brightness = [max(1-brightness, 0), 1+brightness]
        if not contrast is None and contrast>0:
            self.contrast = [max(1-contrast, 0), 1+contrast]
        if not saturation is None and saturation>0:
            self.saturation = [max(1-saturation, 0), 1+saturation]

    def __call__(self, im, ref_im):

        r_brightness = random.uniform(self.brightness[0], self.brightness[1])
        r_contrast = random.uniform(self.contrast[0], self.contrast[1])
        r_saturation = random.uniform(self.saturation[0], self.saturation[1])
        im = ImageEnhance.Brightness(im).enhance(r_brightness)
        im = ImageEnhance.Contrast(im).enhance(r_contrast)
        im = ImageEnhance.Color(im).enhance(r_saturation)

        ref_im = ImageEnhance.Brightness(ref_im).enhance(r_brightness)
        ref_im = ImageEnhance.Contrast(ref_im).enhance(r_contrast)
        ref_im = ImageEnhance.Color(ref_im).enhance(r_saturation)

        return im, ref_im

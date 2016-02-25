# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 16:14:23 2015

@author: Vahndi
"""

import cv2

from OpenCvImage import OpenCvImage

img = cv2.imread('./test files/P1060072.jpg')
Img = OpenCvImage(img)
Img2 = Img.replaceHueRange((0, 50), (50, 100))
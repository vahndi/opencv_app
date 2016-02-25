# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 01:26:06 2015

@author: vahndi
"""

from OpenCvImage import OpenCvImage
from OpenCvWindow import OpenCvWindow
import cv2


imgOriginal = OpenCvImage.fromFile('./P1060072.jpg')
if imgOriginal is not None:

    imgGrayscale = imgOriginal.toGrayScale()
    imgEdges = imgOriginal.toGaussianBlurred().getCannyEdges()
    
    winOriginal = OpenCvWindow('Original')
    winEdges = OpenCvWindow('Edges')
    
    winOriginal.showImage(imgOriginal)
    winEdges.showImage(imgEdges)
    
    cv2.waitKey()                               # hold windows open until user presses a key
    cv2.destroyAllWindows()                     # remove windows from memory

else:
    
    print 'image load failed'
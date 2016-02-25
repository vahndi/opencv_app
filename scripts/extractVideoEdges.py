# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 01:26:06 2015

@author: vahndi
"""

from OpenCvVideoCapture import OpenCvVideoFileCapture
from OpenCvVideoWriter import OpenCvVideoWriter



origFn = 'Arsenal 4-0 Aston Villa - 2015 FA Cup Final _ Goals & Highlights.mp4'
newFn = 'Arsenal Villa Edges3.mp4'

vidOrig = OpenCvVideoFileCapture(origFn)
vidNew = OpenCvVideoWriter(newFn, vidOrig.getSize(), vidOrig.getFramesPerSecond())

try:
    for f in range(1000):
#    for f in range(vidOrig.getFrameCount()):
        print '\rConverting frame %i...' %f, 
#        img = vidOrig.readImage().toGrayScale().toGaussianBlurred().getCannyEdges().toColour()
#        img = vidOrig.readImage().toGrayScale().getCannyEdges().toColour()
        img = vidOrig.readImage().toColoursCMV()
        vidNew.writeImage(img)

except:
    pass


vidOrig.release()
vidNew.release()

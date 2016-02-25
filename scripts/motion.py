# -*- coding: utf-8 -*-
"""
Created on Sat Aug  8 23:38:13 2015

@author: vahndi
"""

from OpenCvVideoCapture import OpenCvVideoCapture, OpenCvVideoCameraCapture
from OpenCvVideoWriter import OpenCvVideoWriter
from OpenCvImage import OpenCvImage
from OpenCvWindow import OpenCvWindow
import numpy as np
import cv2


def previewVideoCamera(cameraIndex = 0):
    '''
    Shows a preview on screen of the video recorded by the specified camera
    '''
    capture = OpenCvVideoCameraCapture(cameraIndex)
    win = OpenCvWindow('Preview for camera %i' % cameraIndex)
    while True:
        try:
            image = capture.readImage()
#            image.drawRectangle((10, 10, 50, 50), (0, 120, 255))
#            OpenCvImage.copyRectangle(image, image, (10, 10, 50, 50), (100, 100, 50, 50))
#            OpenCvImage.swapRectangles(image, image, [(10, 10, 50, 50), (100, 100, 50, 50), (320, 240, 50, 50)])
            win.showImage(image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        except KeyboardInterrupt:
            break
    
    capture.release()
    win.destroyWindow()


def plotMotionHistogram(numberOfFrames, cameraIndex = 0):
    '''
    Plots a histogram of motion over the specified number of frames
    '''
    capture = OpenCvVideoCapture(cameraIndex)
    capture.plotSumsAbsDiff(numberOfFrames)
    capture.release()


def recordMotionAbsoluteThreshold(fileName, cameraIndex = 0,
                          maxCaptureFrames = 10000, threshold = 1e6, framesPerSecond = 15, FourCC = 'mp4v'):
    '''
    Records motion exceeding a threshold frame to frame difference value
    Finish recording by pressing Ctrl + C
    '''
    capture = OpenCvVideoCapture(cameraIndex) # Can't use VideoCameraCapture for this yet as it returns 0,0 for the size
    writer = OpenCvVideoWriter(fileName, capture.getSize(), framesPerSecond = 15, FourCC = FourCC)
    
    numRecordedFrames = 0
    
    while numRecordedFrames != maxCaptureFrames:
        
        try:
            
            frame = capture.readFrame()
            frameAbsDiff = np.sum(capture.absDiff())
            if frameAbsDiff is not None:
                if frameAbsDiff > threshold:
                    writer.writeFrame(frame)
                    numRecordedFrames += 1
             
        except KeyboardInterrupt:
            
            break
            
    capture.release()
    writer.release()

def recordMotionMedianThreshold(fileName, cameraIndex = 0,
                             maxCaptureFrames = 10000, numHistoryFrames = 170, medianMultiplier = 1.05,
                             framesPerSecond = 15, FourCC = 'mp4v'):
    '''
    Records motion exceeding a threshold frame to frame difference value
    Finish recording by pressing Ctrl + C
    '''
    capture = OpenCvVideoCapture(cameraIndex) # Can't use VideoCameraCapture for this yet as it returns 0,0 for the size
    writer = OpenCvVideoWriter(fileName, capture.getSize(), framesPerSecond = framesPerSecond, FourCC = FourCC)
    
    historyDiffs = np.zeros(numHistoryFrames)
    iHistory = 0
    numRecordedFrames = 0
    numAnalysedFrames = 0
    
    while numRecordedFrames != maxCaptureFrames:
        
        try:
            
            frame = capture.readFrame()
            frameAbsDiff = np.sum(capture.absDiff())
            if frameAbsDiff is not None:
                threshold = np.median(historyDiffs) * medianMultiplier
                historyDiffs[iHistory] = frameAbsDiff
                iHistory += 1
                iHistory = iHistory % numHistoryFrames
                if frameAbsDiff > threshold and numAnalysedFrames >= numHistoryFrames:
                    writer.writeFrame(frame)
                    numRecordedFrames += 1
            numAnalysedFrames += 1
             
        except KeyboardInterrupt:
            
            break
            
    capture.release()
    writer.release()



previewVideoCamera(1)
#recordMotionMedianThreshold()
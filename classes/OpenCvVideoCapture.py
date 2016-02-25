import cv2
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt

from OpenCvImage import OpenCvImage



class OpenCvVideoCapture(object):
    
    def __init__(self, cameraIndex = 0, fileName = None, previewWindow = None):
        
        if fileName is not None:
            self._fileName = fileName
            self._videoCapture = cv2.VideoCapture(fileName)
            self._captureType = 'file'
        else:
            self._videoCapture = cv2.VideoCapture(cameraIndex)
            self._captureType = 'camera'
        
        self._previousFrame = None
        self._currentFrame = None
        self._previewWindow = previewWindow
    
    def reloadVideo(self):
        
        self._videoCapture = cv2.VideoCapture(self._fileName)
        
        
        
    # Properties        
        
    def getWidth(self):
            return int(self._videoCapture.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))
    def getHeight(self):
            return int(self._videoCapture.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))
    def getSize(self):
        return (self.getWidth(), self.getHeight())

    def getFourCC(self):
        return int(self._videoCapture.get(cv2.cv.CV_CAP_PROP_FOURCC))

    def getPosFrames(self):
        return int(self._videoCapture.get(cv2.cv.CV_CAP_PROP_POS_FRAMES))    
    def getPosMilliseconds(self):
        return int(self._videoCapture.get(cv2.cv.CV_CAP_PROP_POS_MSEC))
    
    def getFrameCount(self):
        return int(self._videoCapture.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))

    def getFrameIndex(self):
        return int(self._videoCapture.get(cv2.cv.CV_CAP_PROP_POS_FRAMES))
        
    def setFrameIndex(self, index):
        self._videoCapture.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, int(index))
    

    # Methods

    def readFrame(self, index = None):
        '''
        returns None if the capture is not successful
        '''
        if index is not None:
            self._videoCapture.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, index)
        success, frame = self._videoCapture.read()
        if success:
            self._previousFrame = self._currentFrame
            self._currentFrame = frame
        
            if self._previewWindow is not None:
                self._previewWindow.showImage(OpenCvImage(frame))
        
        return frame
        
        
    def readImage(self):
        '''
        Returns an OpenCvImage of the next frame in the file
        '''
        frame = self.readFrame()
        if frame is None:
            return None
        else:
            return OpenCvImage(frame)


    def release(self):
        '''
        Ends the capturing of video and closes any preview window
        '''
        self._videoCapture.release()
        if self._previewWindow is not None:
            self._previewWindow.destroyWindow()
        
    
    def isOpened(self):
        '''
        Returns the isOpened property of the capture
        '''
        return self._videoCapture.isOpened()
        
        
    # Functions
    def absDiff(self):
        '''
        returns the absdiff between the current and previous frame
        '''
        if self._previousFrame is not None and self._currentFrame is not None:
            return cv2.absdiff(self._currentFrame, self._previousFrame)
        else:
            return None
            
    def plotSumsAbsDiff(self, numberOfFrames = 150):
        
        sumsAbsDiff = np.zeros(numberOfFrames)
        self.readFrame()
        for f in range(numberOfFrames):
            self.readFrame()
            sumsAbsDiff[f] = np.sum(self.absDiff())
        fig = plt.figure(figsize = (16, 9))
        ax = fig.add_subplot(121)
        ax.ticklabel_format(style = 'sci', scilimits = (0, 0))
        ax.hist(sumsAbsDiff, bins = 10)
        ax2 = fig.add_subplot(122)
        ax2.plot(np.arange(numberOfFrames), sumsAbsDiff)
        ax2.ticklabel_format(style = 'sci', scilimits = (0, 0))


class OpenCvVideoFileCapture(OpenCvVideoCapture):
    
    
    def __init__(self, fileName):
        
        OpenCvVideoCapture.__init__(self, fileName = fileName)
        
    # Properties
    def getFramesPerSecond(self):
        
        return int(self._videoCapture.get(cv2.cv.CV_CAP_PROP_FPS))
    

class OpenCvVideoCameraCapture(OpenCvVideoCapture):
    
    def __init__(self, cameraIndex = 0):
        
        OpenCvVideoCapture.__init__(self, cameraIndex = cameraIndex)
    
    # Properties
    
    def getFramesPerSecond(self, numCalibrationFrames = 100):
        
        t1 = datetime.now()
        for x in range(numCalibrationFrames):
            self.readFrame()
        t2 = datetime.now()
        tDiff = t2 - t1
        timeTotal = tDiff.seconds + tDiff.microseconds / 1e6
        return int(round(numCalibrationFrames / timeTotal))
        
        
    
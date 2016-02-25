import cv2
import numpy as np


class OpenCvWindow(object):
    
    
    def __init__(self, windowName, windowType = 'fixed', flipLR = True, keypressCallback = None):
        '''
        Wrapper class for an OpenCV Named Window
        '''
        assert windowType in ('fixed', 'resizable') # TODO: Add OpenGL
        
        self._windowName = windowName
        
        if windowType == 'fixed':
            cv2.namedWindow(windowName, cv2.WINDOW_AUTOSIZE)
        elif windowType == 'resizable':
            cv2.namedWindow(windowName, cv2.WINDOW_NORMAL)
        
        self._flipLR = flipLR
        self.keypressCallback = keypressCallback
        
        
    def showImage(self, openCvImage):
        
        if self._flipLR:
            cv2.imshow(self._windowName, np.fliplr(openCvImage.image()))
        else:
            cv2.imshow(self._windowName, openCvImage.image())
        

    def destroyWindow(self):
        
        cv2.destroyWindow(self._windowName)
        self._isWindowCreated = False


    def processEvents(self):
        
        keycode = cv2.waitKey(1)
        if self.keypressCallback is not None and keycode != -1:
            # Discard any non-ASCII info encoded by GTK.
            keycode &= 0xFF
            self.keypressCallback(keycode)
            

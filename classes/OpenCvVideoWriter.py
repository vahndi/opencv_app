import cv2



class OpenCvVideoWriter(object):
    
    
    def __init__(self, fileName, size, framesPerSecond = 30, FourCC = 'mp4v'):
        
        self._videoWriter = cv2.VideoWriter()
        success = self._videoWriter.open(fileName, 
                                         cv2.cv.CV_FOURCC(FourCC[0], FourCC[1], FourCC[2], FourCC[3]),
                                         framesPerSecond, size, True) 

        assert success, 'Failed to open file: %s' % fileName


    def writeFrame(self, frame):
        
        self._videoWriter.write(frame)
        
        
    def writeImage(self, openCvImage):
        
        self.writeFrame(openCvImage.image())
        
    
    def release(self):
        
        self._videoWriter.release()

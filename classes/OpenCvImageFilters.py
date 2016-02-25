
class OpenCvEdgeFilter(object):
    
    
    def __init__(self, blurKernelSize = 5, 
                 cannyEdgesThreshold1 = 100, cannyEdgesThreshold2 = 200):
        
        self._initBlurKernelSize = blurKernelSize
        self._initCannyEdgesThreshold1 = cannyEdgesThreshold1
        self._initCannyEdgesThreshold2 = cannyEdgesThreshold2
        
        self._blurKernelSize = blurKernelSize
        self._cannyEdgesThreshold1 = cannyEdgesThreshold1
        self._cannyEdgesThreshold2 = cannyEdgesThreshold2
        
    
    def applyToImage(self, openCvImage):
        
        return openCvImage.toGrayScale(
               ).toGaussianBlurred(
                   kernelSize = (self._blurKernelSize, 
                                 self._blurKernelSize)).getCannyEdges(
                                     threshold1 = self._cannyEdgesThreshold1,
                                     threshold2 = self._cannyEdgesThreshold2)

                         
    def setBlurKernelSize(self, blurKernelSize):
        self._blurKernelSize = blurKernelSize
    def resetBlurKernelSize(self):
        self._blurKernelSize = self._initBlurKernelSize

        
    def setCannyEdgesThreshold1(self, cannyEdgesThreshold1):
        self._cannyEdgesThreshold1 = cannyEdgesThreshold1
    def resetCannyEdgesThreshold1(self):
        self._cannyEdgesThreshold1 = self._initCannyEdgesThreshold1
    
    def setCannyEdgesThreshold2(self, cannyEdgesThreshold2):
        self._cannyEdgesThreshold2 = cannyEdgesThreshold2
    def resetCannyEdgesThreshold2(self):
        self._cannyEdgesThreshold2 = self._initCannyEdgesThreshold2
    
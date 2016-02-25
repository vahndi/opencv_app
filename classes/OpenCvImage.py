import cv2
import cv2.cv as cv
import numpy as np
from OpenCvColours import *
from copy import deepcopy



hsv_map = np.zeros((180, 256, 3), np.uint8)
h, s = np.indices(hsv_map.shape[:2])
hsv_map[:, :, 0] = h
hsv_map[:, :, 1] = s
hsv_map[:, :, 2] = 255
hsv_map = cv2.cvtColor(hsv_map, cv2.COLOR_HSV2BGR)


class OpenCvImage(object):

    
    def __init__(self, image, fileName = None):
        
        self._image = image
        self._fileName = fileName
        self._shape = image.shape[1::-1]
        
        self._isColour = False
        if image.ndim == 3:
            self._isColour = True
        
    
    @classmethod
    def fromFile(cls, fileName):
        
        image = cv2.imread(fileName)
        if image is None:
            return None
        else:
            return cls(image)
            
    def saveAs(self, filename):
        
        cv2.imwrite(filename, self._image)


    def image(self):
        
        return self._image
    
    
    def shape(self):
        
        return self._shape
        
        
    def toFile(self, asFileName = None):
        
        if self._fileName is None or asFileName is None:
            return False
        else:
            if asFileName is not None:
                self._fileName = asFileName
            cv2.imwrite(self._fileName, self._image)


    def isColour(self):
        
        return self._isColour


    def toGrayScale(self):
        '''
        Returns a grayscale OpenCvImage version of the OpenCvImage if it is colour
        otherwise returns the OpenCvImage itself
        '''
        if self._isColour:
            try:
                return OpenCvImage(cv2.cvtColor(self._image, cv2.COLOR_BGR2GRAY))
            except:
                return None
        else:
            return self
           
           
    def toColour(self):
        
        if self._isColour:
            return self
        else:
            return OpenCvImage(cv2.cvtColor(self._image, cv2.COLOR_GRAY2BGR))


    def toColoursRC(self):
        
        if not self._isColour:
            return None
        else:
            b, g, r = cv2.split(self._image)
            cv2.addWeighted(b, 0.5, g, 0.5, 0, b)
            return OpenCvImage(cv2.merge((b, b, r)))

    
    def toColoursRGV(self):
        
        if not self._isColour:
            return None
        else:
            b, g, r = cv2.split(self._image)
            cv2.min(b, g, b)
            cv2.min(b, r, b)
            return OpenCvImage(cv2.merge((b, g, r)))


    def toColoursCMV(self):
        
        if not self._isColour:
            return None
        else:
            b, g, r = cv2.split(self._image)
            b = cv2.max(b, g)
            b = cv2.max(b, r)
            return OpenCvImage(cv2.merge((b, g, r)))


    def toGaussianBlurred(self, kernelSize = (5, 5), sigmaX = 0, sigmaY = None):
        '''
        Blurs an image using a Gaussian filter.
        Inputs:
            :kernelSize: Gaussian kernel size. ksize.width and ksize.height can differ but they both must be positive and odd. Or, they can be zeros and then they are computed from sigma.
            :sigmaX: Gaussian kernel standard deviation in X direction.
            :sigmaY: Gaussian kernel standard deviation in Y direction; if sigmaY is zero, it is set to be equal to sigmaX, if both sigmas are zeros, they are computed from ksize.width and ksize.height , respectively (see getGaussianKernel() for details); to fully control the result regardless of possible future modifications of all this semantics, it is recommended to specify all of ksize, sigmaX, and sigmaY.
            :borderType (not implemented): pixel extrapolation method (see borderInterpolate() for details).
        Outputs:
            OpenCvImage of the same size and type as the original
        '''
        try:
            return OpenCvImage(cv2.GaussianBlur(self._image, kernelSize, sigmaX, sigmaY))
        except:
            return None
            
    
    def toEqualised(self):
        '''
        Equalises the histogram of the image
        '''
        return OpenCvImage(cv2.equalizeHist(self._image))
    
    
    def getSize(self, divisor = 1):
        '''
        Returns the dimensions in pixels of the image optionally divided by a factor
        '''
        h, w = self._image.shape[:2]
        return (int(w / divisor), int(h / divisor))

    def getArea(self):
        '''
        Returns the area of the image in square pixels
        '''
        h, w = self._image.shape[:2]
        return h * w
        

    def getCannyEdges(self, threshold1 = 100, threshold2 = 200):
        '''
        The function finds edges in the input image image and marks them in the output map edges using the Canny algorithm. 
        The smallest value between threshold1 and threshold2 is used for edge linking. 
        The largest value is used to find initial segments of strong edges. 
        See http://en.wikipedia.org/wiki/Canny_edge_detector
        
        Inputs:
            :threshold1: first threshold for the hysteresis procedure.
            :threshold2: second threshold for the hysteresis procedure.
            :apertureSize (not implemented): aperture size for the Sobel() operator.
            :L2gradient (not implemented): a flag, indicating whether a more accurate L_2 norm should be used to calculate the image gradient magnitude ( L2gradient=true ), or whether the default  L_1 norm  is enough ( L2gradient=false ).
        
        Outputs:
            :edges: output edge map; it has the same size and type as the image
            
        Documentation:
            http://docs.opencv.org/modules/imgproc/doc/feature_detection.html#canny
        '''
        
        return OpenCvImage(cv2.Canny(self._image, threshold1, threshold2))
    
    
    def getHoughCircles(self):
        
        circles = cv2.HoughCircles(self.toGrayScale()._image, cv.CV_HOUGH_GRADIENT, 1, 20,
                                   param1 = 50, param2 = 30, minRadius = 0, maxRadius = 50)
                                   
        if circles is not None:
            return circles[0, :]
            
        return None


    def getHSVmapImage(self, scale = 10, vMin = 0, subRect = None):
        
        if self.isColour():
            
            try:
                # crop image
                image = self._image
                if subRect is not None:
                    image = image[subRect[1]: subRect[1] + subRect[3],
                                  subRect[0]: subRect[0] + subRect[2], :]
                hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                # set values below the vMin threshold to zero
                dark = hsv[..., 2] < vMin
                hsv[dark] = 0
                h = cv2.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
                h = np.clip(h * 0.005 * scale, 0, 1)
                vis = hsv_map * h[:, :, np.newaxis]
                vis2 = np.vstack((vis, vis))
                return OpenCvImage(np.array(vis2, dtype = np.uint8))
            except:
                return None
            
        return None
        
    
    def getMaskedHSVImage(self, hRange = (0, 180), sRange = (0, 255), vRange = (0, 255), maskIsInclusive = True):
        '''
        Although the OpenCV hue range is from 0 to 180, a range from 0 to 360 can be supplied
        to allow for ranges which cross the range e.g. from 150 to 30
        If maskIsInclusive, then the HSV range is included in the returned image 
        otherwise it is excluded and the rest of the image is returned
        '''
        image = self._image
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # Define unwanted ranges
        if hRange[0] % 180 < hRange[1] % 180:
            if maskIsInclusive:
                outH = (hsv[:, :, 0] < hRange[0] % 180) | (hsv[:, :, 0] > hRange[1] % 180)
            else:
                outH = (hsv[:, :, 0] >= hRange[0] % 180) & (hsv[:, :, 0] <= hRange[1] % 180)
        else:
            if maskIsInclusive:
                outH = (hsv[:, :, 0] < hRange[0] % 180) & (hsv[:, :, 0] > hRange[1] % 180)  
            else:
                outH = (hsv[:, :, 0] >= hRange[0] % 180) & (hsv[:, :, 0] <= hRange[1] % 180)  
        
        if maskIsInclusive:
            outS = (hsv[:, :, 1] < sRange[0]) | (hsv[:, :, 1] > sRange[1])
        else:
            outS = (hsv[:, :, 1] >= sRange[0]) & (hsv[:, :, 1] <= sRange[1])
        
        outV = (hsv[:, :, 2] < vRange[0]) | (hsv[:, :, 2] > vRange[1])
            
        # Zero out unwanted ranges
        hsv[outH] = 0
        hsv[outS] = 0
        hsv[outV] = 0

        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        return OpenCvImage(bgr)
    
    
    def replaceHueRange(self, originalRange, newRange):
        
        # Create a masked image with only the hues in the range
        maskedImage = self.getMaskedHSVImage(hRange = originalRange)
        
        # Get HSV versions of the original and masked images
        hsvImage = cv2.cvtColor(self.image(), cv2.COLOR_BGR2HSV)
        hsvMaskedImage = cv2.cvtColor(maskedImage.image(), cv2.COLOR_BGR2HSV)
        
        # Rescale the hues in the masked image
        hsvMaskedImage[:, :, 0] -= originalRange[0]
        hsvMaskedImage[:, :, 0] *= (newRange[1] - newRange[0]) / (originalRange[1] - originalRange[0])
        hsvMaskedImage[:, :, 0] += newRange[0]
        hsvMaskedImage[:, :, 0] = hsvMaskedImage[:, :, 0] % 180
        
        # Replace the matching hues in the original image with the replaced ones from the masked image
        matchArray = (hsvImage[:, :, 0] >= originalRange[0]) & (hsvImage[:, :, 0] <= originalRange[1])
        hsvImage[matchArray] = hsvMaskedImage[matchArray]
        bgrImage = cv2.cvtColor(hsvImage, cv2.COLOR_HSV2BGR)
        return OpenCvImage(bgrImage)


    def replaceHueSaturationRange(self, originalHueRange, originalSaturationRange,
                                        newHueRange, newSaturationRange):
        
        # Create a masked image with only the hues in the range
        maskedImage = self.getMaskedHSVImage(hRange = originalHueRange, 
                                             sRange = originalSaturationRange)
        
        # Get HSV versions of the original and masked images
        hsvImage = cv2.cvtColor(self.image(), cv2.COLOR_BGR2HSV)
        hsvMaskedImage = cv2.cvtColor(maskedImage.image(), cv2.COLOR_BGR2HSV)
        
        # Rescale the hues in the masked image
        hsvMaskedImage[:, :, 0] -= originalHueRange[0]
        hsvMaskedImage[:, :, 0] *= (newHueRange[1] - newHueRange[0]) / (originalHueRange[1] - originalHueRange[0])
        hsvMaskedImage[:, :, 0] += newHueRange[0]
        hsvMaskedImage[:, :, 0] = hsvMaskedImage[:, :, 0] % 180

        # Rescale the saturations in the masked image
        hsvMaskedImage[:, :, 1] -= originalSaturationRange[0]
        hsvMaskedImage[:, :, 1] *= (newSaturationRange[1] - newSaturationRange[0]) / (originalSaturationRange[1] - originalSaturationRange[0])
        hsvMaskedImage[:, :, 1] += newSaturationRange[0]
        hsvMaskedImage[:, :, 1] = hsvMaskedImage[:, :, 1] % 255
        
        # Replace the matching hues in the original image with the replaced ones from the masked image
        matchArray = (hsvImage[:, :, 0] >= originalHueRange[0]) & \
                     (hsvImage[:, :, 0] <= originalHueRange[1])   & \
                     (hsvImage[:, :, 1] >= originalSaturationRange[0]) & \
                     (hsvImage[:, :, 1] <= originalSaturationRange[1])
                     
        hsvImage[matchArray] = hsvMaskedImage[matchArray]
        
        bgrImage = cv2.cvtColor(hsvImage, cv2.COLOR_HSV2BGR)
        return OpenCvImage(bgrImage) 


    def getHSVrange(self, subRect = None):
        
        image = self._image
        if subRect is not None:
            image = image[subRect[1]: subRect[1] + subRect[3],
                          subRect[0]: subRect[0] + subRect[2], :]
        hsvImage = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hueRange = (hsvImage[:, :, 0].min(), hsvImage[:, :, 0].max())
        if hueRange[1] - hueRange[0] > 90:
            hsvImage[hsvImage[:, :, 0] < 45] += 180
            hueRange = (hsvImage[:, :, 0].min(), hsvImage[:, :, 0].max())
        saturationRange = (hsvImage[:, :, 1].min(), hsvImage[:, :, 1].max())
        valueRange = (hsvImage[:, :, 2].min(), hsvImage[:, :, 2].max())
        return (hueRange, saturationRange, valueRange)
            
            
    def replaceHSVrange(self, originalHueRange, originalSaturationRange, originalValueRange,
                              newHueRange, newSaturationRange, newValueRange):
        
        # Create a masked image with only the hues in the range
        maskedImage = self.getMaskedHSVImage(hRange = originalHueRange, 
                                             sRange = originalSaturationRange,
                                             vRange = originalValueRange)
        
        # Get HSV versions of the original and masked images
        hsvImage = cv2.cvtColor(self.image(), cv2.COLOR_BGR2HSV)
        hsvMaskedImage = cv2.cvtColor(maskedImage.image(), cv2.COLOR_BGR2HSV)
        
        # Rescale the hues in the masked image
        hsvMaskedImage[:, :, 0] -= originalHueRange[0]
        hsvMaskedImage[:, :, 0] *= (newHueRange[1] - newHueRange[0]) / (originalHueRange[1] - originalHueRange[0])
        hsvMaskedImage[:, :, 0] += newHueRange[0]
        hsvMaskedImage[:, :, 0] = hsvMaskedImage[:, :, 0] % 180

        # Rescale the saturations in the masked image
        hsvMaskedImage[:, :, 1] -= originalSaturationRange[0]
        hsvMaskedImage[:, :, 1] *= (newSaturationRange[1] - newSaturationRange[0]) / (originalSaturationRange[1] - originalSaturationRange[0])
        hsvMaskedImage[:, :, 1] += newSaturationRange[0]
        hsvMaskedImage[:, :, 1] = hsvMaskedImage[:, :, 1] % 255
        
        # Rescale the values in the masked image
        hsvMaskedImage[:, :, 2] -= originalValueRange[0]
        hsvMaskedImage[:, :, 2] *= (newValueRange[1] - newValueRange[0]) / (originalValueRange[1] - originalValueRange[0])
        hsvMaskedImage[:, :, 2] += newValueRange[0]
        hsvMaskedImage[:, :, 2] = hsvMaskedImage[:, :, 2] % 255
        
        # Replace the matching hues in the original image with the replaced ones from the masked image
        matchArray = (hsvImage[:, :, 0] >= originalHueRange[0]) & \
                     (hsvImage[:, :, 0] <= originalHueRange[1]) & \
                     (hsvImage[:, :, 1] >= originalSaturationRange[0]) & \
                     (hsvImage[:, :, 1] <= originalSaturationRange[1]) & \
                     (hsvImage[:, :, 2] >= originalValueRange[0]) & \
                     (hsvImage[:, :, 2] <= originalValueRange[1])

        hsvImage[matchArray] = hsvMaskedImage[matchArray]
        
        bgrImage = cv2.cvtColor(hsvImage, cv2.COLOR_HSV2BGR)
        return OpenCvImage(bgrImage) 
        

    def flipHorizontal(self):
        '''
        Flips the internal image object horizontally
        '''
        self._image = cv2.flip(self._image, 1)


    def flippedHorizontal(self):
        '''
        Returns a horizontally flipped version of the OpenCvImage
        '''
        return OpenCvImage(cv2.flip(self._image, 1))

        
    def flipVertical(self):
        '''
        Flips the internal image object vertically
        '''
        self._image = cv2.flip(self._image, 0)

    
    def flippedVertical(self):
        '''
        Returns a vertically flipped version of the OpenCvImage
        '''
        return OpenCvImage(cv2.flip(self._image, 0))
        

    def drawRectangle(self, rect, colour = None):
        '''
        Draws a rectangle on top of the image in the specified colour
        Inputs:
            :rect: a tuple of the top-left corner and dimensions of the rectangle (x, y, w, h)
        '''
        if rect is None:
            return
        if colour is None:
            if self._isColour:
                colour = (255, 0, 0)
            else:
                colour = 0

        x, y, w, h = rect
        cv2.rectangle(self._image, (x, y), (x + w, y + h), colour)
    
    
    def drawCircle(self, centre, radius, colour = white, thickness = 1):
        '''
        Draws a circle on top of the image in the specified colour
        '''
        if colour is None:
            if self._isColour:
                colour = (255, 0, 0)
            else:
                colour = 0
        cv2.circle(self._image, centre, radius, colour, thickness)

    
    def drawCircles(self, circles, colour = white, thickness = 1):
        
        if circles is None:
            return
            
        for circle in circles:
            self.drawCircle((int(circle[0]), int(circle[1])), int(circle[2]), colour, thickness)
    
    
    def getZoomed(self, zoomLevel, interpolation = cv2.INTER_LINEAR):
        '''
        Returns an OpenCvImage resized to the specified zoom level
        '''
        currentSize = self.getSize()
        newWidth = int(currentSize[0] * zoomLevel)
        newHeight = int(currentSize[1] * zoomLevel)
        return OpenCvImage(cv2.resize(self._image, (newWidth, newHeight), interpolation = interpolation))


    @classmethod
    def copyRectangle(cls, srcOpenCvImage, dstOpenCvImage, srcRectangle, dstRectangle,
                      interpolation = cv2.INTER_LINEAR):
        '''
        Copy a rectangular region from the source to the destination.
        Inputs:
            :srcOpenCvImage: the source OpenCvImage object (or an image array)
            :dstOpenCvImage: the destination OpenCvImage object (or an image array)
            :srcRect: a tuple of the top-left corner and dimensions of the source rectangle (x, y, w, h)
            :dstRect: a tuple of the top-left corner and dimensions of the destination rectangle (x, y, w, h)
            :interpolation: One of [cv2.INTER_NEAREST (nearest neighbour - cheap but produces blocky results),
                                    cv2.INTER_LINEAR (default - bilinear - good compromise between quality and cost in real-time), 
                                    cv2.INTER_AREA (pixel area relation - may offer a better compromise between cost and quality when downsampling but produces blocky results when upscaling), 
                                    cv2.INTER_CUBIC (bicubic - over a 4x4 pixel neighbourhood, a high-cost and high-quality approach), 
                                    cv2.INTER_LANCZOS4 (lanczos interpolation - over an 8x8 pixel neighbourhood, the highest-cost, highest quality approach)]
        '''
        x0, y0, w0, h0 = srcRectangle
        x1, y1, w1, h1 = dstRectangle
        
        if type(srcOpenCvImage) is OpenCvImage:
            src = srcOpenCvImage._image
        else:
            src = srcOpenCvImage
        if type(dstOpenCvImage) is OpenCvImage:
            dst = dstOpenCvImage._image
        else:
            dst = dstOpenCvImage
            
        # Resize the contents of the source sub-rectangle and put the result in the destination sub-rectangle.
        dst[y1: y1 + h1, x1: x1 + w1] = cv2.resize(src[y0: y0 + h0, x0: x0 + w0], 
                                                   (w1, h1), interpolation = interpolation)

    
    @classmethod                               
    def swapRectangles(cls, srcOpenCvImage, dstOpenCvImage, rectangles, interpolation = cv2.INTER_LINEAR):
        '''
        Perform a circular swap of the rectangles into the destination image
                Copy a rectangular region from the source to the destination.
            Inputs:
                :srcOpenCvImage: the source OpenCvImage object
                :dstOpenCvImage: the destination OpenCvImage object
                :rectangles: a list of tuples of the top-left corner and dimensions of the rectangles to swap (x, y, w, h)
                :interpolation: One of [cv2.INTER_NEAREST (nearest neighbour - cheap but produces blocky results),
                                        cv2.INTER_LINEAR (default - bilinear - good compromise between quality and cost in real-time), 
                                        cv2.INTER_AREA (pixel area relation - may offer a better compromise between cost and quality when downsampling but produces blocky results when upscaling), 
                                        cv2.INTER_CUBIC (bicubic - over a 4x4 pixel neighbourhood, a high-cost and high-quality approach), 
                                        cv2.INTER_LANCZOS4 (lanczos interpolation - over an 8x8 pixel neighbourhood, the highest-cost, highest quality approach)]
        '''
        src = srcOpenCvImage._image
        dst = dstOpenCvImage._image
        
        if dst is not src:
            dst[:] = src
        
        numRects = len(rectangles)
        if numRects < 2:
            return
        
        # Copy the contents of the last rectangle into temporary storage.
        x, y, w, h = rectangles[numRects - 1]
        temp = src[y:y+h, x:x+w].copy()
        
        # Copy the contents of each rectangle into the next.
        i = numRects - 2
        while i >= 0:
            cls.copyRectangle(src, dst, rectangles[i], rectangles[i + 1], interpolation)
            i -= 1
        
        # Copy the temporarily stored content into the first rectangle.
        cls.copyRectangle(temp, dst, (0, 0, w, h), rectangles[0], interpolation)
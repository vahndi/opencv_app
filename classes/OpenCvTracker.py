from __future__ import division
from OpenCvColours import *
import cv2



class OpenCvTrackedFeature(object):


    def __init__(self, featureName, cascadePath, bounds = None, debugRectangleColour = (255, 255, 255)):
        '''
        Inputs:
            :featureName: a string to identify the tracked feature / subfeature e.g. 'Face' / 'Left Eye'
            :cascadePath: the relative path to the object's cascade .xml file
            :bounds: for child features, the relative ((x1, y1), (x2, y2)) coordinates 
                     of the bounding box of the parent, where 0 <= x1, y1, x2, y2 <= 1
        '''
        self.featureName = featureName
        self.classifier = cv2.CascadeClassifier(cascadePath)
        self._children = []
        self.bounds = bounds
        self.debugRectangleColour = debugRectangleColour
        
    
    def addChild(self, childFeature):
        '''
        Inputs:
            :childFeature: another OpenCvTrackedFeature to add as a child
            :childBounds: the relative coordinates within the parent object's 
                             rectangle where the child feature should be located 
                             ((x1,y1),(x2,y2)) where 0 <= x1, y1, x2, y2 <= 1
        '''
        assert childFeature.bounds is not None, 'Child feature must have bounds'
        self._children.append(childFeature)


    def getChildren(self):
        
        return self._children



class OpenCvDetectedFeature(object):
    '''
    A container for the name and location of a detected feature
    '''
    def __init__(self, featureName, rectangle, debugRectangleColour = (255, 255, 255)):
        
        
        self.featureName = featureName
        self.rectangle = rectangle
        self.debugRectangleColour = debugRectangleColour
        self._children = []


    def addChild(self, openCvDetectedFeature):
        
        self._children.append(openCvDetectedFeature)
        

    def getChildren(self):
        
        return self._children



class OpenCvFeatureTracker(object):
    
    
    def __init__(self, scaleFactor = 1.2, minNeighbors = 2, flags = cv2.cv.CV_HAAR_SCALE_IMAGE):
        '''
        When inheriting this class, set each trackedFeature's rectangle to None
        '''
        self.scaleFactor = scaleFactor
        self.minNeighbors = minNeighbors
        self.flags = flags
        self._detectedFeatures = []
        # When inheriting this class, override this statement and assign an
        # OpenCvTrackedFeature with optional child tracked features
        self.mainFeature = None 
        
    
    @property
    def detectedFeatures(self):
        '''
        The tracked objects.
        '''
        return self._detectedFeatures


    def detectFeatures(self, openCvImage):
        '''
        Update the tracked features.
        '''
        self._detectedFeatures = []
        image = openCvImage.toGrayScale().toEqualised()
        minSize = image.getSize(8)
                
        if self.mainFeature is not None:
            
            mainClassifier = self.mainFeature.classifier
            featureRectangles = mainClassifier.detectMultiScale(image._image, self.scaleFactor, 
                                                                self.minNeighbors, self.flags, minSize)
            
            if featureRectangles is not None:
                for featureRectangle in featureRectangles:
                    detectedFeature = OpenCvDetectedFeature(self.mainFeature.featureName, featureRectangle, self.mainFeature.debugRectangleColour)
                    
                    x, y, w, h = featureRectangle
                    
                    for childFeature in self.mainFeature.getChildren():
                        
                        (x1, y1), (x2, y2) = childFeature.bounds
                        searchRect = (x + x1 * w, y + y1 * h, x + x2 * w, y + y2 * h)
                        childRect = self._detectOneObject(childFeature.classifier, openCvImage, searchRect, 64)
                        if childRect is not None:
                            detectedFeature.addChild(OpenCvDetectedFeature(childFeature.featureName, childRect, childFeature.debugRectangleColour))
                    
                    self._detectedFeatures.append(detectedFeature)


    def _detectOneObject(self, classifier, image, rect,
                          imageSizeToMinSizeRatio):
        
        x, y, w, h = rect
        
        minSize = image.getSize(imageSizeToMinSizeRatio)
        subImage = image._image[y: y + h, x: x + w]
        subRects = classifier.detectMultiScale(subImage, self.scaleFactor, self.minNeighbors,
                                               self.flags, minSize)
        
        if len(subRects) == 0:
            return None
        
        subX, subY, subW, subH = subRects[0]
        return (int(x + subX), int(y + subY), int(subW), int(subH))


    def drawDebugRectangles(self, openCvImage):
        '''
        Draw rectangles around the detected features.
        '''
        for detectedFeature in self._detectedFeatures:
            
            if not openCvImage.isColour():
                debugRectangleColour = (255)
            else:
                debugRectangleColour = detectedFeature.debugRectangleColour
        
            openCvImage.drawRectangle(detectedFeature.rectangle, debugRectangleColour)  
            
            for childFeature in detectedFeature.getChildren():
                
                if not openCvImage.isColour():
                    debugRectangleColour = (255)
                else:
                    debugRectangleColour = childFeature.debugRectangleColour
        
                openCvImage.drawRectangle(childFeature.rectangle, debugRectangleColour) 



class OpenCvFaceTracker(OpenCvFeatureTracker):


    def __init__(self, scaleFactor = 1.2, minNeighbors = 2, flags = cv2.cv.CV_HAAR_SCALE_IMAGE):
        
        super(OpenCvFaceTracker, self).__init__(scaleFactor, minNeighbors, flags = flags)
        
        face = OpenCvTrackedFeature('Face', '../cascades/haarcascade_frontalface_alt.xml', debugRectangleColour=red)
        face.addChild(OpenCvTrackedFeature('Left Eye', '../cascades/haarcascade_eye.xml', bounds = ((1 / 7, 0), (3 / 7, 0.5)), debugRectangleColour=chartreuse))
        face.addChild(OpenCvTrackedFeature('Right Eye', '../cascades/haarcascade_eye.xml', bounds = ((4 / 7, 0), (6 / 7, 0.5)), debugRectangleColour=green))
        face.addChild(OpenCvTrackedFeature('Nose', '../cascades/haarcascade_mcs_nose.xml', bounds = ((0.25, 0.25), (0.75, 0.75)), debugRectangleColour=orange))
        face.addChild(OpenCvTrackedFeature('Mouth', '../cascades/haarcascade_mcs_mouth.xml', bounds = ((1 / 6, 2/3), (5 / 6, 1)), debugRectangleColour=yellow))
        
        self.mainFeature = face


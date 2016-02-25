# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 21:02:05 2015

@author: Vahndi
"""

import pygame as pg
import cv2
import os
import pickle

from OpenCvVideoCapture import OpenCvVideoCapture
from OpenCvVideoWriter import OpenCvVideoWriter
from OpenCvTracker import OpenCvFaceTracker
from OpenCvImage import OpenCvImage
from OpenCvColours import *

from PygameFileBrowser import PygameFileBrowser
from PygameButtonSets import PygameOptionButtonSet, PygameScrollButtonSet, PygamePushButtonSet
from PygameKeypressHandler import PygameKeypressHandler, PygameKeypressHandlerSet
from PygameApplication import PygameApplication
from OpenCvImageFilters import OpenCvEdgeFilter
from PygameScreenArea import PygameScreenArea
from PygameTrackbar import PygameTrackbar



class OpenCvApplication(PygameApplication):
    
    
    def __init__(self):
        '''
        Runs an application to edit and manipulate captures
        '''
        
        super(OpenCvApplication, self).__init__('OpenCV App', (1000, 800))
        
        # Captures
        self._cameraCapture = None
        self._videoFileCapture = None
        self._photo = None
        self._videoFileBrowser = PygameFileBrowser(os.path.realpath(__file__).rstrip('OpenCvApplication.py'), self._appDisplay)
        self._photoFileBrowser = PygameFileBrowser(os.path.realpath(__file__).rstrip('OpenCvApplication.py'), self._appDisplay)
        
        # OpenCV add-ins
        self._faceTracker = OpenCvFaceTracker()
        self._edgeFilter = OpenCvEdgeFilter()
        
        # State variables
        self._edgeFilterOn = False
        self._trackFaces = False
        self._swapFaces = False
        self._drawDebugRectangles = False
        self._drawHoughCircles = False
        self._browsingForVideo = False
        self._browsingForPhoto = False
        self._videoFilePlaying  = False
        self._HSVmap = False
        self._applyHSVmap = False
        self._zoomLevel = 1
        self._findAndReplaceHSV = False
        self._findHSVrange = None 
        self._replaceHSVrange = None

        
        # Screen Areas
        self._captureScreenArea = PygameScreenArea(self._appDisplay, [10, 10, 0, 0])
        self._HSVmapScreenArea = PygameScreenArea(self._appDisplay, [60, 10, 0, 0])
        self._screenAreas = [self._captureScreenArea, self._HSVmapScreenArea]
        
        # Settings
        self._maxInitPreviewHeight = 500
        self._image = None
        
        # Button sets
        captureButtonArgs = (('camera', 'Camera Capture', False, self.toggleCameraCapture, None, None),
                             ('file', 'Video File Capture', False, self.toggleVideoFileCapture, None, self.stopBrowsingForVideoFile),
                             ('photo', 'Photo', False, self.togglePhotoCapture, None, None))
        processingButtonArgs = (('edges', 'Edge Filter', False, self.toggleEdges, None, None),
                                ('faces', 'Track Faces', False, self.toggleTrackFaces, None, None),
                                ('faceSwap', 'Swap Faces', False, self.toggleSwapFaces, None, None),
                                ('circles', 'Hough Circles', False, self.toggleHoughCircles, None, None),
                                ('debugRects', 'Debug Rectangles', False, self.toggleDebugRectangles, None, None),
                                ('HSVmap', 'HSV map', False, self.toggleHSVmap, None, None),
                                ('findReplaceHSVs', 'Replace HSV', False, self.toggleFindAndReplaceHSV, None, None))        
        edgesButtonArgs = (('kernel size', 'Kernel Size', range(1, 50, 2), 5, self._edgeFilter.resetBlurKernelSize, 
                            self._edgeFilter.setBlurKernelSize, self._edgeFilter.setBlurKernelSize),
                           ('threshold1', 'Threshold 1', range(0, 256), 100, self._edgeFilter.resetCannyEdgesThreshold1,
                            self._edgeFilter.setCannyEdgesThreshold1, self._edgeFilter.setCannyEdgesThreshold1),
                           ('threshold2', 'Threshold 2', range(0, 256), 200, self._edgeFilter.resetCannyEdgesThreshold2,
                            self._edgeFilter.setCannyEdgesThreshold2, self._edgeFilter.setCannyEdgesThreshold2))                             
        videoCaptureButtonArgs = (('play', 'Play / Pause', False, self.toggleVideoFilePlaying, None, None),)        
        captureScrollButtonArgs = (('size', 'Size', [0.125, 0.25, 0.5, 0.66, 0.75, 1, 1.5, 2, 4], 1, self.setZoomLevel, self.setZoomLevel, self.setZoomLevel),)
        hsvMapButtonArgs = (('apply hsv map', 'Apply to Image', False, self.toggleApplyHSVmap, None, None),)
        captureButtons = PygameOptionButtonSet(self._appDisplay, [10, 50], captureButtonArgs,
                                               isVisibleFunction = self.isNotBrowsingForVideo,
                                               exclusiveSelect = True, buttonWidth =  150)
        processingButtons = PygameOptionButtonSet(self._appDisplay, [10, 100], processingButtonArgs,
                                                  isVisibleFunction = self.hasActiveCapture,
                                                  exclusiveSelect = False, buttonWidth =  150)
        videoCaptureButtons = PygameOptionButtonSet(self._appDisplay, [10, 150], videoCaptureButtonArgs,
                                                    isVisibleFunction = self.hasVideoFileCapture,
                                                    exclusiveSelect = False, buttonWidth =  150)                                                    
        self._capturePreviewButtons = PygameScrollButtonSet(self._appDisplay, [10, 200], captureScrollButtonArgs,
                                                      isVisibleFunction = self.hasActiveCapture, 
                                                      buttonWidth =  150)                                                      
        edgeFilterButtons = PygameScrollButtonSet(self._appDisplay, [10, 250], edgesButtonArgs,
                                                  isVisibleFunction = self.edgeFilterOn, 
                                                  buttonWidth = 200, fontSize = 9)
        hsvMapButtons = PygameOptionButtonSet(self._appDisplay, [10, 300], hsvMapButtonArgs,
                                              isVisibleFunction = self.displayingHSVmap, exclusiveSelect = False, buttonWidth = 150)
        hsvMapSaveButtons = PygamePushButtonSet(self._appDisplay, [10,350],
                                                (('save additive', 'Save Include', self._saveIncludeHSVregion),
                                                 ('save subractive', 'Save Exclude', self._saveExcludeHSVregion),
                                                 ('write video', 'Write Video', self._saveVideo),
                                                 ('set find range', 'Set as Find', self._setHSVfindRange),
                                                 ('set replace range', 'Set as Replace', self._setHSVreplaceRange)),
                                                 isVisibleFunction = self.displayingHSVmap)
        self._videoCaptureTrackBar = None

        self._buttonSets = (captureButtons, 
                            processingButtons, 
                            videoCaptureButtons, 
                            self._capturePreviewButtons,
                            edgeFilterButtons,
                            hsvMapButtons,
                            hsvMapSaveButtons)

        # Keypress Handlers        
        videoBrowsingKeypressHandlerSet = PygameKeypressHandlerSet([PygameKeypressHandler(pg.K_ESCAPE, self.stopBrowsingForVideoFile)],
                                                                   conditionFunction = self.isBrowsingForVideo,
                                                                   notHandledFunction = self.handleVideoBrowsingKeypress,
                                                                   sendEvent = True)
        photoBrowsingKeypressHandlerSet = PygameKeypressHandlerSet([PygameKeypressHandler(pg.K_ESCAPE, self.stopBrowsingForPhotoFile)],
                                                                   conditionFunction = self.isBrowsingForPhoto,
                                                                   notHandledFunction = self.handlePhotoBrowsingKeypress,
                                                                   sendEvent = True)
        self._keypressHandlerSets = (videoBrowsingKeypressHandlerSet,
                                     photoBrowsingKeypressHandlerSet)


    def showImage(self, openCvImage, location = (0, 0)):
        '''
        Shows an image at the specified location tuple
        '''
        if openCvImage.isColour():
            captureConversionType = cv2.COLOR_BGR2RGB
        else:
            captureConversionType = cv2.COLOR_GRAY2RGB
        rgbFrame = cv2.cvtColor(openCvImage.image(), captureConversionType)
        pygameFrame = pg.image.frombuffer(rgbFrame.tostring(), openCvImage.shape(), 'RGB')
        self._appDisplay.blit(pygameFrame, location)
        
        
    # Option Toggles
    # --------------
    def toggleEdges(self):
        self._edgeFilterOn = not self._edgeFilterOn
        self.clearScreen()
    
    def toggleTrackFaces(self):
        self._trackFaces = not self._trackFaces
    
    def toggleSwapFaces(self):
        self._swapFaces = not self._swapFaces
        
    def toggleDebugRectangles(self):
        self._drawDebugRectangles = not self._drawDebugRectangles
    
    def toggleHoughCircles(self):
        self._drawHoughCircles = not self._drawHoughCircles
        
    def toggleFindAndReplaceHSV(self):
        self._findAndReplaceHSV = not self._findAndReplaceHSV

    def toggleCameraCapture(self):
        if self._cameraCapture is not None:
            self.killCameraCapture()
            self.clearScreen()
        else:
            self.initCameraCapture()

    def toggleVideoFileCapture(self):
        if self._videoFileCapture is not None:
            # Stop the video file capture
            self.killVideoFileCapture()
            self.clearScreen()
            self._videoFilePlaying = False
            self._browsingForVideo = False
        else:
            if self._cameraCapture is not None:
                self.killCameraCapture()
            # Browse for a video file                    
            self._browsingForVideo = True     
            
    def togglePhotoCapture(self):
        if self._photo is not None:
            self._photo = None
            self.clearScreen()
            self._browsingForPhoto = False
        else:
            if self._cameraCapture is not None:
                self.killCameraCapture()
            if self._videoFileCapture is not None:
                self.killVideoFileCapture()
            self.clearScreen()
            self._browsingForPhoto = True  

    def toggleVideoFilePlaying(self):
        self._videoFilePlaying = not self._videoFilePlaying
        if self._videoFilePlaying:
            self.clearScreen()

    def toggleHSVmap(self):
        self._HSVmap = not self._HSVmap
        self.clearScreen()
        
    def displayingHSVmap(self):
        return self._HSVmap and self.hasActiveCapture()
        
    def toggleApplyHSVmap(self):
        self._applyHSVmap = not self._applyHSVmap

 
    # Captures
    # --------  
    def initCameraCapture(self):
        self.clearScreen()
        self._cameraCapture = OpenCvVideoCapture(0, fileName = None)
        self._cameraShape = self._cameraCapture.getSize()
    
    def initVideoFileCapture(self, videoFilePath):
        self.clearScreen()
        self._videoFileCapture = OpenCvVideoCapture(fileName = videoFilePath)
        
        # Reduce preview height to <= self._maxInitPreviewHeight
        self._zoomLevel = 1
        previewHeight = self._videoFileCapture.getSize()[1]
        while previewHeight > self._maxInitPreviewHeight:
            self._capturePreviewButtons.scrollDownButton('size')
            previewHeight = self._videoFileCapture.getSize()[1] * self._zoomLevel
        self._setVideoFileCaptureIndex(0)
        self._videoCaptureTrackBar = PygameTrackbar(self._appDisplay, (self._captureScreenArea.getLeft(),
                                                                       self._captureScreenArea.getBottom() + 1), 
                                                    self._videoFileCapture.getSize()[0] * self._zoomLevel,
                                                    maxValue = self._videoFileCapture.getFrameCount() - 1,
                                                    numValues = self._videoFileCapture.getFrameCount(),
                                                    valueChangingFunction = self._setVideoFileCaptureIndex,
                                                    valueChangedFunction = self._setVideoFileCaptureIndex)
        self._trackBars = (self._videoCaptureTrackBar,)
    
    def killCameraCapture(self):
        if self._cameraCapture is not None:
            self.clearScreen()
            self._cameraCapture.release()
            self._cameraCapture = None
            self._captureScreenArea.reset()
            self._image = None
    
    def killVideoFileCapture(self):
        if self._videoFileCapture is not None:
            self._videoFileCapture.release()
            self._videoFileCapture = None
            self._captureScreenArea.reset()
            self._videoCaptureTrackBar = None
            self._trackBars = []
            self._image = None
    
    def stopBrowsingForVideoFile(self):
        self.clearScreen()
        self._browsingForVideo = False
        
    def stopBrowsingForPhotoFile(self):
        self.clearScreen()
        self._browsingForPhoto = False
        
    def handleVideoBrowsingKeypress(self, event):
        self._videoFileBrowser.handleEvent(event)
        selectedVideoFile = self._videoFileBrowser.getSelectedFile()
        if selectedVideoFile is not None:
            self.stopBrowsingForVideoFile()
            self._videoFileBrowser.reset()
            self.initVideoFileCapture(selectedVideoFile)
    
    def handlePhotoBrowsingKeypress(self, event):
        self._photoFileBrowser.handleEvent(event)
        selectedPhotoFile = self._photoFileBrowser.getSelectedFile()
        if selectedPhotoFile is not None:
            self._photo = OpenCvImage.fromFile(selectedPhotoFile)
#            self._photo = OpenCvImage(self._photo.image().transpose(1, 0, 2))
            # Reduce preview height to <= self._maxInitPreviewHeight
            self._zoomLevel = 1
            previewHeight = self._photo.getSize()[1]
            while previewHeight > self._maxInitPreviewHeight:
                self._capturePreviewButtons.scrollDownButton('size')
                previewHeight = self._photo.getSize()[1] * self._zoomLevel
    
    def setZoomLevel(self, zoomLevel):
        self.clearScreen()
        self._zoomLevel = zoomLevel
        if self._videoCaptureTrackBar is not None:
            self._videoCaptureTrackBar.setWidth(self._videoFileCapture.getWidth() * self._zoomLevel)

    def clearScreen(self):
        try:
            self._appDisplay.fill(black)
        except:
            pass

    # State properties

    def hasVideoFileCapture(self):
        return self._videoFileCapture is not None
        
    def hasCameraCapture(self):
        return self._cameraCapture is not None
        
    def hasActiveCapture(self):
        return self.hasVideoFileCapture() or self.hasCameraCapture() or self._photo is not None
        
    def hasStreamingCapture(self):
        return (self.hasVideoFileCapture() and self._videoFilePlaying) or self.hasCameraCapture()

    def getActiveCapture(self):
        if self.hasVideoFileCapture():
            return self._videoFileCapture
        elif self.hasCameraCapture():
            return self._cameraCapture
        return None

    def isBrowsingForVideo(self):
        return self._browsingForVideo
        
    def isBrowsingForPhoto(self):
        return self._browsingForPhoto

    def isNotBrowsingForVideo(self):
        return not self._browsingForVideo    
    
    def isNotBrowsingForPhoto(self):
        return not self._browsingForPhoto   
    
    def edgeFilterOn(self):
        return self._edgeFilterOn
    
    # Actions
    
    def _loadSettings(self):
        settingsFn = '/Users/Vahndi/Desktop/settings.pkl'
        if os.path.exists(settingsFn):
            settings = pickle.load(open(settingsFn, 'rb'))
        else:
            settings = {}
        return settings
        
    def _saveSettings(self, settings):
        pickle.dump(settings, open('/Users/Vahndi/Desktop/settings.pkl', 'wb'))
        
    
    def _saveIncludeHSVregion(self):
        
        settings = self._loadSettings()
        if 'Included HSV Regions' in settings.keys():
            includedHSVregions = settings['Included HSV Regions']
        else:
            includedHSVregions = []
        includedHSVregions.append(self._HSVmapScreenArea.getRectangleTuple())
        settings['Included HSV Regions'] = includedHSVregions
        self._saveSettings(settings)
    
    
    def _saveExcludeHSVregion(self):
        
        settings = self._loadSettings()
        if 'Excluded HSV Regions' in settings.keys():
            excludedHSVregions = settings['Excluded HSV Regions']
        else:
            excludedHSVregions = []
        excludedHSVregions.append(self._HSVmapScreenArea.getRectangleTuple())
        settings['Excluded HSV Regions'] = excludedHSVregions
        self._saveSettings(settings)


    def _setHSVfindRange(self):
        
        if self._captureScreenArea.hasRectangle():
            rect = self._captureScreenArea.getRectangleTuple()
            image = self._captureScreenArea.getOpenCvImage()
            self._findHSVrange = image.getHSVrange(rect)


    def _setHSVreplaceRange(self):
        
        if self._captureScreenArea.hasRectangle():
            rect = self._captureScreenArea.getRectangleTuple()
            image = self._captureScreenArea.getOpenCvImage()
            self._replaceHSVrange = image.getHSVrange(rect)


    def _saveVideo(self):
        
        settings = self._loadSettings()
        
        vidcap = self._videoFileCapture
        vidcap.reloadVideo()
        writer = OpenCvVideoWriter('/Users/Vahndi/Desktop/temp_exc.avi', vidcap.getSize())
        
        for i in range(vidcap.getFrameCount()):
            print '\rProcessing frame %i' %i
            originalImage = vidcap.readImage()
            image = self.processImage(originalImage.getZoomed(0.5))
#            # Calculate image from included HSV regions
#            frames = []            
#            for inc in settings['Included HSV Regions']:
#                frames.append(originalImage.getMaskedHSVImage((inc[1], inc[1] + inc[3]), (inc[0], inc[0] + inc[2]), maskIsInclusive = True).image())
#            frame = np.zeros((originalImage.getSize()[1], originalImage.getSize()[0], 3), dtype = np.uint8)
#            for f in frames:
#                frame = frame | f
#            # Calculate image from excluded HSV regions
#            frames = []            
#            for exc in settings['Excluded HSV Regions']:
#                frames.append(image.getMaskedHSVImage((exc[1], exc[1] + exc[3]), 
#                                                              (exc[0], exc[0] + exc[2]), 
#                                                              maskIsInclusive = False).image())
#            frame = originalImage.image()
#            for f in frames:
#                frame = frame & f
            writer.writeFrame(image.image())
        writer.release()


    def _setVideoFileCaptureIndex(self, index):
        
        self._videoFileCapture.setFrameIndex(index)
        self._image = self._videoFileCapture.readImage()

    def processImage(self, image):
        '''
        Applies the current settings to the image
        '''
        # Hue replacement
        if self._findAndReplaceHSV:
            if None not in [self._findHSVrange, self._replaceHSVrange]:
                image = image.replaceHSVrange(self._findHSVrange[0], self._findHSVrange[1], self._findHSVrange[2],
                                              self._replaceHSVrange[0], self._replaceHSVrange[1], self._replaceHSVrange[2])
        # Face-tracking                
        if self._trackFaces:
            self._faceTracker.detectFeatures(image)
            if self._swapFaces:
                OpenCvImage.swapRectangles(image, image, [df.rectangle for df in self._faceTracker.detectedFeatures])
            if self._drawDebugRectangles:
                self._faceTracker.drawDebugRectangles(image)
        # Edges
        if self._edgeFilterOn:
            image = self._edgeFilter.applyToImage(image)
        # Circles
        if self._drawHoughCircles:
            image.drawCircles(simage.getHoughCircles())

        return image
        

    def drawScreen(self):
        
        image = None
        capImage = None
                
        # Get an image from the active capture if there is one
        if self._cameraCapture is not None:
            capImage = self._cameraCapture.readImage().flippedHorizontal()  
            self._image = capImage
        elif self._videoFileCapture is not None:
            if self._videoFilePlaying:
                capImage = self._videoFileCapture.readImage()     
                if capImage is None:
                    self._videoFileCapture.reloadVideo()
                    capImage = self._videoFileCapture.readImage()  
                self._image = capImage
                self._videoCaptureTrackBar.setCurrentValue(self._videoFileCapture.getFrameIndex())
            else:
                capImage = self._image
        elif self._photo is not None:
            capImage = self._photo
        
        # Process capture image if there is an active capture
        if capImage:
            image = capImage
            # Hue replacement
            if self._findAndReplaceHSV:
                if None not in [self._findHSVrange, self._replaceHSVrange]:
                    image = image.replaceHSVrange(self._findHSVrange[0], self._findHSVrange[1], self._findHSVrange[2],
                                                  self._replaceHSVrange[0], self._replaceHSVrange[1], self._replaceHSVrange[2])
            # Face-tracking                
            if self._trackFaces:
                self._faceTracker.detectFeatures(image)
                if self._swapFaces:
                    OpenCvImage.swapRectangles(image, image, [df.rectangle for df in self._faceTracker.detectedFeatures])
                if self._drawDebugRectangles:
                    self._faceTracker.drawDebugRectangles(image)
            # Edges
            if self._edgeFilterOn:
                image = self._edgeFilter.applyToImage(image)
            # Circles
            if self._drawHoughCircles:
                image.drawCircles(simage.getHoughCircles())
            if self._zoomLevel != 1:
                image = image.getZoomed(self._zoomLevel)
        
        # Show images
        if image:
            self._captureScreenArea.setOpenCvImage(image)
            if self._HSVmap:
                subRect = self._captureScreenArea.getRectangle()
                hsvImage = None                
                if subRect:
                    subRectArea = max(subRect.getArea(), 1)
                    subRect = subRect.getTuple()
                    hsvImage = image.getHSVmapImage(subRect = subRect, 
                                                    scale = 10 * image.getArea() / subRectArea)
                else:
                    hsvImage = image.getHSVmapImage(scale = 10)
                    
                if hsvImage:
                    self._HSVmapScreenArea.setOpenCvImage(hsvImage)
                    self._HSVmapScreenArea.setLeft(self._captureScreenArea.getRight() + 50)
                    self._HSVmapScreenArea.draw()
                    hsvRect = self._HSVmapScreenArea.getRectangle()
                    if hsvRect and self._applyHSVmap:
                        try:
                            maskedImage = capImage.getMaskedHSVImage(hRange = (hsvRect.getTop(), hsvRect.getBottom()), 
                                                                     sRange = (hsvRect.getLeft(), hsvRect.getRight()))
                            if self._zoomLevel != 1:
                                maskedImage = maskedImage.getZoomed(self._zoomLevel)
                            self._captureScreenArea.setOpenCvImage(maskedImage)
                        except:
                            pass
            
            self._captureScreenArea.draw() 
        else:
            self.clearScreen()

        # Modes
        if self._browsingForVideo:
            self.clearScreen()
            self._videoFileBrowser.printPath()
        elif self._browsingForPhoto:
            self._photoFileBrowser.printPath()
        
        # Draw buttons
        buttonSetY = 50
        if self.hasActiveCapture():
            if self._videoFileCapture is not None:
                self._videoCaptureTrackBar.setOrigin((self._captureScreenArea.getLeft(), 
                                                      self._captureScreenArea.getBottom() + 5))
                self._videoCaptureTrackBar.draw()
                buttonSetY += self._videoCaptureTrackBar.getHeight()
            buttonSetY += self._captureScreenArea.getHeight() + 15
        for buttonSet in self._buttonSets:
            if buttonSet.isVisible():
                buttonSet.setOriginY(buttonSetY)
                buttonSet.draw()
                buttonSetY += buttonSet.getHeight() + 15


    def cleanUp(self):
        
        self.killCameraCapture()
        self.killVideoFileCapture()



app = OpenCvApplication()
app.run()




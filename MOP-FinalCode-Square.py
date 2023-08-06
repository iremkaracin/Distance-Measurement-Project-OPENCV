from cmath import nan
from cv2 import threshold
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import cv2 as cv
from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import imutils
from os import listdir
from os.path import isfile, join
import gc



class FieldCalculater:
    def __init__(self):
        self.image_pixel_map = {}
        pass

    def readImage(self, path):
        self.image_name = path
        self.image = cv.imread(path)
        

    def cropImage(self, bottom, top, left, right):
        self.cropped_image = self.image[bottom:top, left:right]
        cv.imwrite("cropped.png", self.cropped_image)

    def getImageSize(self):
        return self.image.shape

    # Otsu's thresholding
    def otsuThreshold(self, image):
        self.gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        self.blur = cv.GaussianBlur(self.gray,(21,21),0)
        ret2, self.threshold_image = cv.threshold(self.blur, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
        cv.imwrite("threshold.png", self.threshold_image)

    def opening(self, image):
        k = np.ones([15,15])
        self.opening_img = cv.morphologyEx(image,cv.MORPH_OPEN,k)
        cv.imwrite("opening.png", self.opening_img)

    def edgeDetect(self, image):
        self.edged = cv.Canny(image, 100, 200)
        self.edged = cv.dilate(self.edged, None, iterations=1)
        self.edged = cv.erode(self.edged, None, iterations=1)

        cnts = cv.findContours(self.edged.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)

        # sort the contours from left-to-right and initialize the
        # 'pixels per metric' calibration variable
        (cnts, _) = contours.sort_contours(cnts)
        
        # loop over the contours individually
        for c in cnts:
        # if the contour is not sufficiently large, ignore it
            if cv.contourArea(c) < 500:
                continue  
            # compute the rotated bounding box of the contour
            
            box = cv.minAreaRect(c)
            box = cv.cv.BoxPoints(box) if imutils.is_cv2() else cv.boxPoints(box)
            box = np.array(box, dtype="int")
            
            # order the points in the contour such that they appear
            # in top-left, top-right, bottom-right, and bottom-left
            # order, then draw the outline of the rotated bounding
            # box
            box = perspective.order_points(box)
            (tl, tr, br, bl) = box
            
            # top left ile top right noktalarinin mesafesi * top left ile bottom left arasindaki mesafe
            cornerPointsDistanceTop = dist.euclidean(tl ,tr)
            cornerPointsDistanceLeft = dist.euclidean(tl, bl)
            self.object_height_px = cornerPointsDistanceLeft

            # Resmin ustune cizilen noktalardan ilgisiz ALAN:
            #print(cornerPointsDistanceTop,cornerPointsDistanceLeft)
        
            cv.imwrite("edged.png", self.edged)
    
    def addImagePixelMap(self):
        self.image_pixel_map[self.image_name] = self.getObjectHeightPx()     
    
    def getObjectHeightPx(self):
        return self.object_height_px
                                     
    def getThresholdImage(self):
        return self.threshold_image

    def getCropImage(self):
        return self.cropped_image

    def getOpeningImage(self):
        return self.opening_img

    def calculateDistance (self):          
        focal_length_mm = 34
        object_height_mm = 19
        image_height_px = 4000
        sensor_height = 14.9
        object_height_px = float(self.object_height_px)
        
        #distance equation
        
        distance_mm = float((focal_length_mm*object_height_mm*image_height_px)/(object_height_px*sensor_height))
        print(distance_mm)
        return distance_mm
        
        
    def calculate(self, path):
        self.readImage(path)
        dimensions = self.getImageSize()
        #print(dimensions)
        self.cropImage(int(dimensions[0]/4.5), int(dimensions[0]/1.5), int(dimensions[1]/3), int(dimensions[1]/1.5))
        croppedImage = self.getCropImage()
        self.otsuThreshold(croppedImage)
        thresholdImage = self.getThresholdImage()
        self.opening(self.threshold_image)
        opening = self.getOpeningImage()
        self.edgeDetect(opening)
        self.addImagePixelMap()
        return self.calculateDistance()
       
       
    def ctsCalculate(self, path):
        self.readImage(path)
        self.otsuThreshold(self.image)
        self.edgeDetect(self.getThresholdImage())
        
        
         
            
        
    
def isimage(path):
    if path[-3:].lower() == 'jpg':
        return True
    
"""
if __name__ == "__main__":
    folder_path = 'C:\\Users\\iremk\\OneDrive\\Desktop\\Field'
    onlyfiles = [f for f in listdir(folder_path) if isimage(f)]
    
    distance_list = []
    fieldCalculater = FieldCalculater()
    for i in onlyfiles:
        print(i)
        result_dis = fieldCalculater.calculate(folder_path + '\\' + i)
        distance_list.append(result_dis)
    #print(distance_list)
    #print(fieldCalculater.image_pixel_map)
    
        
    pixel_list = list(fieldCalculater.image_pixel_map.values())
    #print(pixel_list)
    
    y = distance_list
    x = pixel_list
    plt.title( "Object Height (px) vs The Object Position (mm)")
    plt.xlabel("Object Height in pixels")
    plt.ylabel("Object Distance in mm")
    plt.grid()
    plt.plot(x,y)
    plt.show()
"""

if __name__ == "__main__":
    fieldCalculater = FieldCalculater()
    fieldCalculater.ctsCalculate("C:\\Users\\iremk\\OneDrive\\Desktop\\laser2.png")
    print("bitti")
    

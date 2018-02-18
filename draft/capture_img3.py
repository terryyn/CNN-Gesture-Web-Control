import cv2
import numpy as np
import os

#save = False
#numberPerGesture = 

#def saveImage(img):
#    save=False
x1=20
y1=50
h1=156
w1=180


showVideo = True
methodMask = True
selectRec = True
betterQuality = False
waitTime = 500
count = 0
saveimg = False


def Mask(img,num):
    
    if selectRec:
        cv2.rectangle(img,(x1,y1),(x1+w1,y1+h1),(255,0,0),1)
        img = img[y1:y1+h1, x1:x1+w1]

    lowerBound = np.array([0, 0, 60])
    upperBound = np.array([30, 200, 255])
    
    #kernelOpen=np.ones((5,5))
    kernelClose= np.ones((20,20))

    #convert BGR to HSV
    imgHSV= cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

    #create mask
    mask=cv2.inRange(imgHSV,lowerBound,upperBound)

    #morphology
    #mask=cv2.morphologyEx(mask, cv2.MORPH_OPEN,kernelOpen)
    mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,kernelClose)

    mask = cv2.GaussianBlur(mask, (15,15), 1)

    if betterQuality:
        kernelBettter=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
        mask = cv2.erode(mask, kernelBettter, iterations = 1)
        mask = cv2.dilate(mask, kernelBettter, iterations = 1)
    
    #bitwise and mask original frame
    imgBit = cv2.bitwise_and(img, img, mask = mask)

    # color to grayscale
    imgGray = cv2.cvtColor(imgBit, cv2.COLOR_BGR2GRAY)
    #print(imgGray) 
    
    imgGray = cv2.resize(imgGray,(52,60),interpolation = cv2.INTER_AREA)
    if saveimg:
        print(cv2.imwrite('./1/g'+str(num)+'.ppm', imgGray))
    #drawing rectangle over objects

#    if save == True:
#        saveImage(imgBit)

    return imgGray
    
def alternative(img):
    if selectRec:
        cv2.rectangle(img,(x1,y1),(x1+w1,y1+h1),(255,0,0),1)
        img = img[y1:y1+h1, x1:x1+w1]
    maskGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    maskBlur = cv2.GaussianBlur(maskGray,(5,5),1)
    maskThs = cv2.adaptiveThreshold(maskBlur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,2)
    ret,maskFinal = cv2.threshold(maskThs, 70, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    return maskFinal

def main():
    count = 0
    cam= cv2.VideoCapture(0)
    ret = cam.set(3,104)
    ret = cam.set(4,120)

    while True: #(count < 10):
        ret, img=cam.read()
        img = cv2.flip(img,3)

        if ret :
            if methodMask:
                count=count+1
                mask=Mask(img,count)
            else:
                mask=alternative(img)
        if showVideo:
            cv2.imshow("mask",mask)
            cv2.imshow("Original",img)
        cv2.waitKey(waitTime)


if __name__ == "__main__":
    main()

        
    

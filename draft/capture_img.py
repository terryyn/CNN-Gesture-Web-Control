import cv2
import numpy as np


#save = False
#numberPerGesture = 

#def saveImage(img):
#    save=False
x1=150
y1=50
h1=150
w1=150


showVideo = True
methodMask = True
selectRec = True

def Mask(img):
    if selectRec:
        cv2.rectangle(img,(x1,y1),(x1+w1,y1+h1),(255,0,0),1)
        img = img[y1:y1+h1, x1:x1+w1]

    lowerBound = np.array([0, 20, 60])
    upperBound = np.array([20, 180, 255])
    
    #kernelOpen=np.ones((5,5))
    kernelClose= np.ones((5,5))

    #convert BGR to HSV
    imgHSV= cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

    #create mask
    mask=cv2.inRange(imgHSV,lowerBound,upperBound)

    #morphology
    #mask=cv2.morphologyEx(mask, cv2.MORPH_OPEN,kernelOpen)
    mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,kernelClose)

    mask = cv2.GaussianBlur(mask, (15,15), 1)
    
    #bitwise and mask original frame
    imgBit = cv2.bitwise_and(img, img, mask = mask)

    # color to grayscale
    imgBit = cv2.cvtColor(imgBit, cv2.COLOR_BGR2GRAY)
    #drawing rectangle over objects

#    if save == True:
#        saveImage(imgBit)

    return imgBit
    
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
    cam= cv2.VideoCapture(0)
    ret = cam.set(3,72)
    ret = cam.set(4,64)

    while True:
        ret, img=cam.read()
        img = cv2.flip(img,3)

        if ret :
            if methodMask:
                mask=Mask(img)
            else:
                mask=alternative(img)
        if showVideo:
            cv2.imshow("mask",mask)
            cv2.imshow("Original",img)
        
        cv2.waitKey(10)


if __name__ == "__main__":
    main()

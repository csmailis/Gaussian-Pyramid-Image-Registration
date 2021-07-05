##################### AUTHOR: CHRISTOS SMAILIS #####################
import cv2
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.cm as cm
import math
import os
import time
#Function for computing Normalized Cross Correlation between 2 images
def NormalizedCrossCorrelation(T,I):
    IW, IH = I.shape[::-1]
    TW, TH = T.shape[::-1]
    I=np.float64(I)
    T=np.float64(T)
    Nominator = np.sum( np.multiply(T-T.mean(),I-I.mean()) )
    Denominator = np.sqrt(np.multiply( np.sum(np.square(T-T.mean())), np.sum(np.square(I-I.mean())) ))
    R=Nominator/Denominator
    return R

#Function that performs image rotation
def rotateImage(image, angle):
    (h, w) = image.shape[:2]
    center = (w / 2, h / 2)
    # rotate the image by 'angle' degrees
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    result = cv2.warpAffine(image, M, (w, h))
    return result

#Angle Estimation using Gaussian pyramids
def findBestAngleGaussPyr(original,rotated,level,initialAngle):
    maxNcc=-1
    maxAngle=-180
    resultingImage=[]
    print
    #if Level = 4 use angle range = [-70,70]
    if initialAngle == 70:
        print "Level: "+str(level)
        print "Image Size: " + str(rotated.shape)
        print "Angle Range: ("+str(-initialAngle)+", "+str(-initialAngle)+")"
        for angle in xrange(-initialAngle,initialAngle,1):
            #rotating image for angles within angle range
            GaRotated=rotateImage(rotated, angle)
            #computing NormalizedCrossCorrelation
            ncc = NormalizedCrossCorrelation(GaRotated,original)
            # finding angle with the higher  NormalizedCrossCorrelation
            if (maxNcc<ncc):
                maxNcc=ncc
                maxAngle=angle
                resultingImage=GaRotated
    else:
        #else compute a di9fferent angle range per level
        #(angle range is reduced per level)
        angleVariance=180*(level+1)*2/100
        print "Level: "+str(level)
        print "Image Size: " + str(rotated.shape)
        angleRange = range(initialAngle-int(angleVariance),initialAngle+int(angleVariance),1)
        print "Angle Range: "+str(angleRange[0])+","+str(angleRange[-1])
            
        for angle in angleRange:
            #rotating image for angles within angle range
            GaRotated = rotateImage(rotated, angle)
            #computing NormalizedCrossCorrelation
            ncc = NormalizedCrossCorrelation(GaRotated,original)
            # finding angle with the higher  NormalizedCrossCorrelation
            if (maxNcc<ncc):
                maxNcc=ncc
                maxAngle=angle
                resultingImage=GaRotated        
                
            
    plt.figure(figsize=(20, 6))
    plt.subplot(131),plt.imshow(rotated,cmap = 'gray')
    plt.title('Source'), plt.xticks([]), plt.yticks([])
    plt.subplot(132),plt.imshow(original,cmap = 'gray')
    plt.title('Target'), plt.xticks([]), plt.yticks([])
    plt.subplot(133),plt.imshow(resultingImage,cmap = 'gray')
    plt.title('Result'), plt.xticks([]), plt.yticks([])
    plt.suptitle(" Gaussian Pyramid Level="+str(level)+", Rotation Angle="+str(maxAngle)+", NCC="+str(maxNcc))
    
    plt.savefig("plots/plot"+" Gaussian Pyramid Level="+str(level)+", Rotation Angle="+str(maxAngle)+", NCC="+str(maxNcc)+'.png', format='png')
    plt.close()
    print "NCC: "+str(maxNcc)
    print "Estmated Angle: "+str(maxAngle)
    return maxAngle


def findBestAngleSimple(original,rotated,initialAngle):
    maxNcc=-1
    maxAngle=-180
    resultingImage=[]
    print
    print "Angle Range: ("+str(-initialAngle)+", "+str(initialAngle)+")"
    print "Image Size: " + str(rotated.shape)
    for angle in xrange(-initialAngle,initialAngle,1):
        #rotating image for angles within angle range
        GaRotated=rotateImage(rotated, angle)
        #computing NormalizedCrossCorrelation
        ncc= NormalizedCrossCorrelation(GaRotated,original)
        # finding angle with the higher  NormalizedCrossCorrelation
        if (maxNcc<ncc):
            maxNcc=ncc
            maxAngle=angle
            resultingImage=GaRotated

            
    plt.figure(figsize=(20, 6))
    plt.subplot(131),plt.imshow(rotated,cmap = 'gray')
    plt.title('Source'), plt.xticks([]), plt.yticks([])
    plt.subplot(132),plt.imshow(original,cmap = 'gray')
    plt.title('Target'), plt.xticks([]), plt.yticks([])
    plt.subplot(133),plt.imshow(resultingImage,cmap = 'gray')
    plt.title('Result'), plt.xticks([]), plt.yticks([])
    plt.suptitle(" Simple Registration, Rotation Angle="+str(maxAngle)+", NCC="+str(maxNcc))
    

    plt.savefig("plots/plot"+" Simple NCC, Rotation Angle="+str(maxAngle)+", NCC="+str(maxNcc)+'.png', format='png')
    plt.close()
    print "NCC: "+str(maxNcc)
    print "Estmated Angle: "+str(maxAngle)
    return maxAngle

####################################################### Main Program ###########################################
if __name__ == '__main__':

#####Gaussian Pyramid Registration
    #All plots for this assignment are stored within a plots directory
    # that is created automatically after code execution
    if not os.path.exists("plots/"):
        os.makedirs("plots/")

    A = cv2.imread('Original.png')
    A = cv2.cvtColor(A, cv2.COLOR_BGR2GRAY)
    B = cv2.imread('Rotated.png')
    B = cv2.cvtColor(B, cv2.COLOR_BGR2GRAY)
     
    # generate Gaussian pyramids for source and target
    Ga = A.copy() # target image
    Gb = B.copy() # source image
    initialAngle=70
    GaList=[Ga]
    GbList=[Gb]
    print "####### Gaussian Pyramid NCC Registration #######"
    start = time.time()

    for level in xrange(4):
        Ga = cv2.pyrDown(Ga)
        GaList.append(Ga)
        Gb = cv2.pyrDown(Gb)
        GbList.append(Gb)

    for level in xrange(5):
        initialAngle=findBestAngleGaussPyr(GaList[4-level],GbList[4-level],4-level,initialAngle)
    end = time.time()
    print
    print "Time Elapsed:"
    print(end - start)
#####Simple NCC Registration
    print
    print "####### Simple NCC Registration #######"
    start = time.time()
    findBestAngleSimple(A,B,70)
    end = time.time()
    print
    print "Time Elapsed:"
    print(end - start)



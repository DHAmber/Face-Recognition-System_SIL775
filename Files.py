import glob as g
import cv2 as cv
import numpy as np
import Utils as U


data_face = 'PositiveImage/*.jpg'
data_no_face = 'NegativeImageg/*.jpg'

def PopulateImages(IsPositive):
    if IsPositive:
        img_path = g.glob(data_face)
    else:
        img_path=g.glob(data_no_face)
    Images = []
    for i in range(len(img_path)):
        # for i in range(25):
        img = cv.imread(img_path[i], 0)
        img_arr = np.array(img, dtype=np.float32)
        Images.append(img_arr)
    return Images

def ResizeImage(Images):
    for i in range(len(Images)):
        Images[i].resize(U.SIZE,refcheck=False)
    return Images

def NormalizedImage(Images):
    NormalizedImg=[]
    for i in range(len(Images)):
        NormalizedImg.append(cv.normalize(Images[i], None, alpha=0, beta=1, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F))
    return NormalizedImg

def Partition(images):
    totalImages = len(images)
    eightyPer=round(totalImages*80/100)
    tenPer=round(totalImages*10/100)
    train_imgs = images[:eightyPer]
    dev_imgs = images[eightyPer:eightyPer + tenPer]
    test_imgs = images[eightyPer + tenPer:]
    return train_imgs, dev_imgs, test_imgs




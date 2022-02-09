from functools import partial
from multiprocessing import Pool, cpu_count

from progressbar import progressbar

import Utils
import numpy as np
import HaarFeature as haar

def CreateFeatureTemplate(ImgWidth,ImgHeight):
    haarFeatures=[]
    for feature in Utils.FeatureType:
        feat_start_width = max(Utils.min_feature_width, feature.value[0])
        for feature_w in range(Utils.min_feature_width,feat_start_width,feature.value[0]):
            feat_start_height = max(Utils.min_feature_height, feature.value[1])
            for feature_h in range(Utils.min_feature_height,feat_start_height,feature.value[1]):
                for i in range(ImgWidth-feat_start_width):
                    for j in range(ImgHeight-feat_start_height):
                        haarFeatures.append(haar(feature, (i, j), feature_w, feature_h, 0,1))
                        haarFeatures.append(haar(feature, (i, j), feature_w, feature_h, 0,-1))
    return haarFeatures

def GetVotesForHaarFeature(Pos_IntegralImage,Neg_IntegralImage):
    bar = progressbar.ProgressBar()
    NUM_PROCESS = cpu_count() * 3
    pool = Pool(processes=NUM_PROCESS)

    Num_Pos = len(Pos_IntegralImage)
    Num_Neg = len(Neg_IntegralImage)
    Num_Imags = Num_Pos + Num_Neg

    X, Y = Pos_IntegralImage[0].shape

    Mx_Feature_W=Utils.max_feature_width
    Mn_Feature_W=Utils.min_feature_width

    Mx_Feature_H=Utils.max_feature_height
    Mn_Feature_H=Utils.min_feature_height

    OneArrayPostive=np.ones(Num_Pos)
    OneArrayNegative = np.ones(Num_Neg)

    ImgWeightPos=OneArrayPostive * 1/(2*Num_Pos)
    ImgWeightNeg = OneArrayNegative * 1/(2*Num_Neg)

    total_Weight=np.hstack((ImgWeightPos,ImgWeightNeg))
    labels=np.hstack((OneArrayPostive,OneArrayNegative))

    Images=Pos_IntegralImage+Neg_IntegralImage

    features = CreateFeatureTemplate(X, Y)

    num_features = len(features)
    feature_index = list(range(num_features))

    votes = np.zeros((Num_Imags, num_features))

    for i in bar(range(Num_Imags)):
        votes[i, :] = np.array(list(pool.map(partial(Utils.get_feature_vote, image=Images[i]), features)))
    return votes


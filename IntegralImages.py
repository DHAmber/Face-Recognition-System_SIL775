import Utils
import numpy as np

def IntegralImage(img):
    #img = [[1, 12, 45, 10], [6, 5, 11, 4], [3, 7, 10, 8], [5, 9, 4, 7]]
    image = np.array(img)
    x, y = np.shape(image)
    result = np.zeros_like(image)

    for i in range(x):
        for j in range(y):
            s = 0
            for m in range(i + 1):
                for n in range(j + 1):
                    s = s + image[m][n]
            result[i][j] = s
    #print(result)
    return result


def GetIntegralImages(Images):
    result=[]
    for i in range(len(Images)):
        result.append(IntegralImage(Images[i]))
    return result
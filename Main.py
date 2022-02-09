import Files
import Utils
import IntegralImages
import AdaBoost

if __name__ == '__main__':
    print('Images are loading...')
    P=Files.PopulateImages(True)
    N=Files.PopulateImages(False)

    print('Images are resizing...')
    Pos = Files.ResizeImage(P)
    Neg = Files.ResizeImage(N)

    print('Images are normalizing...')
    PosImages = Files.NormalizedImage(Pos)
    NegImages = Files.NormalizedImage(Neg)

    Pos_Train,Pos_dev,Pos_test=Files.Partition(PosImages)
    Neg_Train, Neg_dev, Neg_test = Files.Partition(NegImages)

    print('Image loaded and partioned for training and testing')

    Pos_IntegralImage_Train=IntegralImages.GetIntegralImages(Pos_Train)
    Pos_IntegralImage_dev=IntegralImages.GetIntegralImages(Pos_dev)
    Pos_IntegralImage_test=IntegralImages.GetIntegralImages(Pos_test)

    Neg_IntegralImage_Train = IntegralImages.GetIntegralImages(Neg_Train)
    Neg_IntegralImage_dev = IntegralImages.GetIntegralImages(Neg_dev)
    Neg_IntegralImage_test = IntegralImages.GetIntegralImages(Neg_test)

    print('Positive and Negative Integral Iamges are ready')

    Votes= AdaBoost.GetVotesForHaarFeature(Pos_IntegralImage_Train, Neg_IntegralImage_Train)
    print(Votes)




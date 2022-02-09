from enum import Enum

SIZE = (24, 24)

num_classifier = 10
min_feature_height = 4
max_feature_height = 10
min_feature_width = 4
max_feature_width = 10

class FeatureType(Enum):
    HaarFeature_TWO_VERTICAL = (1, 2)
    HaarFeature_TWO_HORIZONTAL = (2, 1)
    HaarFeature_THREE_VERTICAL = (1, 3)
    HaarFeature_THREE_HORIZONTAL = (3, 1)
    HaarFeature_FOUR = (2, 2)

def GetSum(self, int_img):
        score, white, black = 0, 0, 0

        if self.type == FeatureType.TWO_VERTICAL:
            white =white+ get_WindowSum(int_img, self.top_left,
                             (int(self.top_left[0] + self.width), int(self.top_left[1] + self.height / 2)))
            black =black+ get_WindowSum(int_img, (self.top_left[0],
                                      int(self.top_left[1] + self.height / 2)), self.bottom_right)

        elif self.type == FeatureType.TWO_HORIZONTAL:
            white =white+ get_WindowSum(int_img, self.top_left,
                             (int(self.top_left[0] + self.width / 2), self.top_left[1] + self.height))
            black =black+ get_WindowSum(int_img,
                            (int(self.top_left[0] + self.width / 2), self.top_left[1]), self.bottom_right)

        elif self.type == FeatureType.THREE_VERTICAL:
            white =white+ get_WindowSum(int_img, self.top_left,
                             (self.bottom_right[0], int(self.top_left[1] + self.height / 3)))
            black =black+ get_WindowSum(int_img, (self.top_left[0], int(self.top_left[1] + self.height / 3)),
                            (self.bottom_right[0], int(self.top_left[1] + 2 * self.height / 3)))
            white =white+ get_WindowSum(int_img, (self.top_left[0],
                                       int(self.top_left[1] + 2 * self.height / 3)), self.bottom_right)

        elif self.type == FeatureType.THREE_HORIZONTAL:
            white =white+ get_WindowSum(int_img, self.top_left,
                             (self.bottom_right[0], int(self.top_left[1] + self.height / 3)))
            black =black+ get_WindowSum(int_img, (self.top_left[0], int(self.top_left[1] + self.height / 3)),
                            (self.bottom_right[0], int(self.top_left[1] + 2 * self.height / 3)))
            white =white+ get_WindowSum(int_img, (self.top_left[0], int(
                self.top_left[1] + 2 * self.height / 3)), self.bottom_right)

        elif self.type == FeatureType.FOUR:
            white =white+ get_WindowSum(int_img, self.top_left,
                             (int(self.top_left[0] + self.width / 2), int(self.top_left[1] + self.height / 2)))
            black =black+ get_WindowSum(int_img, (int(self.top_left[0] + self.width / 2), self.top_left[1]),
                            (self.bottom_right[0], int(self.top_left[1] + self.height / 2)))
            black =black+ get_WindowSum(int_img, (self.top_left[0], int(self.top_left[1] + self.height / 2)),
                            (int(self.top_left[0] + self.width / 2), self.bottom_right[1]))
            white =white+ get_WindowSum(int_img, (int(self.top_left[0] + self.width / 2),
                                       int(self.top_left[1] + self.height / 2)), self.bottom_right)

        score = white - black
        return score


def get_WindowSum(int_img, top_left, bottom_right):
    # get summed value over a rectangle (attention to the tuple)

    top_left = (top_left[1], top_left[0])
    bottom_right = (bottom_right[1], bottom_right[0])
    # must swap the tuples since the orientation of the coordinate system
    if top_left == bottom_right:
        return int_img[top_left]
    top_right = (bottom_right[0], top_left[1])
    bottom_left = (top_left[0], bottom_right[1])

    return int_img[bottom_right] + int_img[top_left] - int_img[bottom_left] - int_img[top_right]

def get_feature_vote(feature, image):
    # para feature: HaarLikeFeature object
    # para image: integral image
    return feature.get_vote(image)
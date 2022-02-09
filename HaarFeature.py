class HaarFeature(object):
    def __init__(self, feature_type, position, width, height, threshold, parity, weight=1):
        self.type = feature_type
        self.top_left = position
        self.bottom_right = (position[0] + width, position[1] + height)
        self.width = width
        self.height = height
        self.threshold = threshold
        self.parity = parity    #For Direction
        self.weight = weight
from Data_Sturcture import FeatureMap, Kernel


class PoollingLayer:
    def __init__(self, feature: FeatureMap, kernel: Kernel):
        self.feature = feature
        self.kernel = kernel

import numpy as np
from Data_Sturcture import FeatureMap

class ReLU:
    def __init__(self, feature_map: FeatureMap):
        self.input = feature_map
        self.output = None

    def Calculate(self) -> FeatureMap:
        relu_result = np.maximum(0, self.input.feature)
        self.output = FeatureMap(
            relu_result,
            self.input.label)
        return self.output

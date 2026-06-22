import numpy as np
from Data_Sturcture import FeatureMap, Vector


def Relu(input_data):
    """
        FeatureMap -> FeatureMap
        Vector -> Vector
    """

    if isinstance(input_data, FeatureMap):
        relu_result = np.maximum(0, input_data.feature)
        output = FeatureMap(relu_result, input_data.label)
        return output

    elif isinstance(input_data, Vector):
        relu_result = np.maximum(0, input_data.vector)
        output = Vector(relu_result, input_data.label)
        return output

    else:
        raise TypeError("Relu 的输入必须是 FeatureMap 或 Vector")

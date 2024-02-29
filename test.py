from modules.cnns.resnet import ResNetLayer
from tools.analyze import *

if __name__ == "__main__":
    input = torch.rand(1, 3, 256, 256)

    res_layer = ResNetLayer(3, 6)

    print(count_parameters(res_layer))
    print(profile_model(res_layer, input))
import torch.nn as nn
import torchvision.models as models


def build_model(model_index=0):
    match model_index:
        case 0: model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        case 1: model = models.efficientnet_b1(weights=models.EfficientNet_B1_Weights.DEFAULT)
        case 2: model = models.efficientnet_b2(weights=models.EfficientNet_B2_Weights.DEFAULT)
        case 3: model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.DEFAULT)
        case 4: model = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.DEFAULT)
        case 5: model = models.efficientnet_b5(weights=models.EfficientNet_B5_Weights.DEFAULT)
        case 6: model = models.efficientnet_b6(weights=models.EfficientNet_B6_Weights.DEFAULT)
        case 7: model = models.efficientnet_b7(weights=models.EfficientNet_B7_Weights.DEFAULT)
        case _: return None

    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, 1),
        nn.Sigmoid()  # normalized age in [0, 1]
    )
    return model
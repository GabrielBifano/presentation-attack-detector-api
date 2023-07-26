from torch import nn
from torch import load
import torchvision.models as models

class SpoofIdentifier(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            models.resnet34(weights=models.ResNet34_Weights.DEFAULT),
            nn.Flatten(),
            nn.Linear(1000, 2),
        )

    def forward(self, x):
        return self.model(x)


def load_model():
    weights_path = './src/model_resnet34_6epochs.pth'
    weights = load(weights_path, map_location="cpu")
    model = SpoofIdentifier()
    model.load_state_dict(weights)
    return model 
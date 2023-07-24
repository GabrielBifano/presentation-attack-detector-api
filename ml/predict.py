from ml.model import load_model
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

# To be applied
transformer = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224),antialias=True),
    transforms.Normalize([0.5, 0.5, 0.5],[0.5, 0.5, 0.5],)
])

def predict_image():

    imgset = ImageFolder('./data/', transform=transformer,)
    imgloader = DataLoader(
        imgset,
        64, # batch size
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    model = load_model()
    model.train = False

    for data, _ in imgloader:
        output = model(data)
        _, predicted = output.max(1)
        return predicted[0]
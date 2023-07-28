from torchvision.io import read_image
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms

train_path = '**the path to the dataset**/train'
test_path = '**the path to the dataset**/test'


train_transformer = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomVerticalFlip(0.5),
    transforms.Resize((224, 224), antialias=True),
    transforms.Normalize([0.5, 0.5, 0.5],
                         [0.5, 0.5, 0.5],),
])

trainset = ImageFolder(train_path, transform=train_transformer,)
TRAIN_Loader = DataLoader(
    trainset,
    200, # batch size
    shuffle=True,
    num_workers=0,
    pin_memory=True,
)


test_transformer = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224), antialias=True),
    transforms.Normalize([0.5, 0.5, 0.5],
                         [0.5, 0.5, 0.5],)
])

testset = ImageFolder(test_path, transform=test_transformer,)
TEST_Loader = DataLoader(
    testset,
    200, # batch size
    shuffle=True,
    num_workers=0,
    pin_memory=True,
)
from imp import load_module
from torchvision import transforms


def dream_predict(img):

    model = load_module()
    model.train = False

    t_img = transforms.ToTensor()(img)
    t_img = transforms.Resize((224, 224), antialias=True)(t_img)
    t_img = transforms.Normalize([0.5, 0.5, 0.5],[0.5, 0.5, 0.5],)(t_img)
    t_img = t_img.unsqueeze(0).to('cuda')

    out = model(t_img)
    _, pred = out.max(1)
    return pred
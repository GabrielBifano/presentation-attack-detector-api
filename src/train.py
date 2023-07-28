from torch.optim import Adam
from src.model import SpoofIdentifier
from trainer import Learner

model = SpoofIdentifier()
opt = Adam(model.parameters(), ls=1e-4)

if __name__ == '__main__':
    trainer = Learner(model, opt)
    trainer.train()
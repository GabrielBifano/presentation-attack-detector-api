from torch import save
from tester import Tester
from learner import Learner
from torch.optim import Adam
from src.model import SpoofIdentifier

model = SpoofIdentifier()
opt = Adam(model.parameters(), ls=1e-4)

test_model_as_well = True
save_weights_as_well = True

if __name__ == '__main__':
    trainer = Learner(model, opt)
    trainer.train() # might pass num of epochs
    
    if save_weights_as_well:
        # you might change the file's name
        filename = 'trained_weights'
        with open(f'./weights/{filename}.pth', 'wb') as f:
            save(trainer.model.state_dict(), f)
    
    if test_model_as_well:
        tester = Tester(trainer.model)
        tester.test_model()
        tester.confusion_matrix()
import time
from dataset import TRAIN_Loader
from torch import nn

class Learner:

    def __init__(
        self,
        model,
        optimizer,
        dataloader = TRAIN_Loader,
        loss_func = nn.CrossEntropyLoss,
        ):

        self.model = model
        self.dataloader = dataloader
        self.loss_func = loss_func
        self.optimizer = optimizer


    def train(self, epochs=6):
        for epoch  in range(epochs):
            start_time = time.time()

            for batch in self.dataloader:
                X, y = batch
                X, y = X.to('cuda'), y.to('cuda')
                yhat = self.model(X)
                loss = self.loss_func(yhat, y)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            # printing each epoch loss and time
            end_time = time.time()
            print(f'Epoch {epoch}   has a loss of   {loss:.10f}   and took   {(end_time - start_time):.1f} seconds')
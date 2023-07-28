import matplotlib.pyplot as plt
from dataset import TEST_Loader
import torch

class Tester:
    def __init__(
        self,
        model,
        dataloader=TEST_Loader,
        ):
        self.model = model
        self.dataloader = dataloader 
        self.c_matrix = torch.zeros(2, 2).to('cuda')

    def test_model(self):
        
        correct = 0
        total = 0

        self.c_matrix = torch.zeros(2, 2).to('cuda')
        self.model.train = False

        for data, target in self.dataloader:
            data, target = data.to('cuda'), target.to('cuda')
            output = self.model(data)
            _, predicted = output.max(1)
            for i in range(predicted.shape[0]):
                self.c_matrix[predicted[i], target[i]] += 1
                total += 1
                correct += (predicted[i] == target[i]).sum().item()

        accuracy = correct / total
        print(f'Accuracy = {(accuracy * 100):.4f}')

    def confusion_matrix(self):

        conf = self.c_matrix.cpu()

        _, ax = plt.subplots()
        ax.set_xticks([])
        ax.set_yticks([])
        ax.imshow(conf, cmap="Blues")
        ax.set_title("Confusion Matrix")
        ax.set_ylabel('Actual\n\nspoof                           live', fontsize=12)
        ax.set_xlabel('live                           spoof\n\nPredicted', fontsize=12)

        for i in range(2):
            for j in range(2):
                text = f"{conf[i, j]}"
                ax.text(j, i, text, va="center", ha="center")
        plt.show()
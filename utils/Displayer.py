import torch
import torchmetrics


class displayer(object):
    def __init__(self, name_list):
        self.name_list = name_list
        self.metric_list = [0, 0]

    def cal_accuray(self, pred, label):
        self.metric_list[0] += (pred.argmax(1) == label).type(torch.float).sum().item()

    def record_loss(self, loss):
        self.metric_list[1] += loss.item()

    def get_avg(self, full_count, batch_count):
        self.metric_list[0] = self.metric_list[0] / full_count
        self.metric_list[1] = self.metric_list[1] / batch_count

        return self.metric_list

    def reset(self):
        self.metric_list = [0] * len(self.name_list)

    def display(self):
        for i, value in enumerate(self.metric_list):
            print(f"{self.name_list[i]} : {self.metric_list[i]:.4f}  ", end="")
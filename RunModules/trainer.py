import Options
import os
import re
import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.autograd import grad as torch_grad
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from Model.PyramidViG import Pyramid_ViG
from utils.DataLoader import Loader
from utils.Displayer import displayer

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class train(Options.param):
    def __init__(self):
        super(train, self).__init__()
        os.makedirs(self.OUTPUT_CKP, exist_ok=True)
        os.makedirs(self.OUTPUT_LOSS, exist_ok=True)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.disp_tr = displayer(["train_acc, train_loss"])
        self.disp_te = displayer(["val_acc, val_loss"])

    def init_weight(self, module):
        class_name = module.__class__.__name__

        if class_name.find("Conv2d") != -1:
            nn.init.normal_(module.weight.data, 0.0, 0.02)

        elif class_name.find("BatchNorm2d") != -1:
            nn.init.normal_(module.weight.data, 1.0, 0.02)
            nn.init.constant(module.bias.data, 0.0)

    def run(self):
        print('--------------------------------------------------')
        print(f'[DEVICE] : {self.device}')
        print('--------------------------------------------------')

        PyramidViG = Pyramid_ViG().to(self.device)

        if self.CKP_LOAD:
            ckp = torch.load(f'{self.OUTPUT_CKP}', map_location=self.device)
            PyramidViG.load_state_dict(ckp["PyramidViG_state_dict"])
            epoch = ckp["epoch"] + 1
        else:
            # PyramidViG.apply(self.init_weight)
            epoch = 0

        PyramidViG.train()

        transform = transforms.Compose(
            [
                transforms.Resize((self.SIZE, self.SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ]
        )
        tr_dataset = Loader(self.DATASET_PATH, self.DATA_STYPE[0], self.DATA_CLS, transform)
        te_dataset = Loader(self.DATASET_PATH, self.DATA_STYPE[1], self.DATA_CLS, transform)

        criterion_CEL = nn.CrossEntropyLoss(label_smoothing=0.1)

        summary = SummaryWriter()

        optim_AdamW = optim.AdamW(list(PyramidViG.parameters()), lr=self.LR, weight_decay=0.05)
        scheduler_cosin = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=optim_AdamW, T_0=20, T_mult=1, eta_min=1e-5)

        for ep in range(epoch, self.EPOCH):
            tr_dataloader = DataLoader(dataset=tr_dataset, batch_size=self.BATCHSZ, shuffle=True)
            te_dataloader = DataLoader(dataset=te_dataset, batch_size=self.BATCHSZ, shuffle=True)

            # training Loop
            for idx, (item, label) in enumerate(tqdm.tqdm(tr_dataloader, desc=f'TRAINING EPOCH [{ep} / {self.EPOCH}]')):
                item = item.to(self.device)
                label = label.to(self.device)

                output = PyramidViG(item)
                loss_CEL = criterion_CEL(output, label)

                """
                loss average 값 구하는 display function
                """
                self.disp_tr.cal_accuray(output, label)
                self.disp_tr.record_loss(loss_CEL)

                optim_AdamW.zero_grad()
                loss_CEL.backward()
                optim_AdamW.step()

            scheduler_cosin.step()

            # PyramidViG.eval()
            with torch.no_grad:
                PyramidViG.eval()
                for idx, (item, label) in enumerate(tqdm.tqdm(te_dataloader, desc=f'TESTING EPOCH [{ep} / {self.EPOCH}]')):
                    item = item.to(self.device)
                    label = label.to(self.device)

                    output = PyramidViG(item)
                    loss = criterion_CEL(output, label)

                    self.disp_te.cal_accuray(output, label)
                    self.disp_te.record_loss(loss)


            tr_item_list = self.disp_tr.get_avg(len(tr_dataloader.dataset), len(tr_dataloader))
            te_item_list = self.disp_te.get_avg(len(te_dataloader.dataset), len(te_dataloader))

            print(f"===> EPOCH[{ep}/{self.EPOCH}] || train acc : {tr_item_list[0]}   |   train loss : {tr_item_list[1]}"
                  f"   |   test acc : {te_item_list[0]}   |   test loss : {te_item_list[1]} ||")

            summary.add_scalar("train/acc", tr_item_list[0], ep)
            summary.add_scalar("train/loss", tr_item_list[1], ep)

            summary.add_scalar("test/acc", te_item_list[0], ep)
            summary.add_scalar("test/loss", te_item_list[1], ep)

            self.disp_tr.reset()
            self.disp_te.reset()

            torch.save(
                {
                    "PyramidViG_state_dict": PyramidViG.state_dict(),
                    "optim_AdamW_state_dict" : optim_AdamW.state_dict(),
                    "epoch": ep,
                },
                os.path.join(f"{self.OUTPUT_CKP}/ckp", f"{epoch}.pth"),
            )







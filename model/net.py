import torch
import torch.nn as nn
import torch.nn.functional as F

from model.layers import PEPX, Flatten


class Net(nn.Module):
    def __init__(self, **kwargs):
        """
        Creates the Net model.
        Refer to the paper for more details: https://arxiv.org/pdf/2003.09871.pdf
        Refer Repository: https://github.com/iliasprc/Net/blob/master/model/model.py
        """
        super(Net, self).__init__()
        filters = {
            "pepx1_1": [56, 56],
            "pepx1_2": [56, 56],
            "pepx1_3": [56, 56],
            "pepx2_1": [56, 112],
            "pepx2_2": [112, 112],
            "pepx2_3": [112, 112],
            "pepx2_4": [112, 112],
            "pepx3_1": [112, 216],
            "pepx3_2": [216, 216],
            "pepx3_3": [216, 216],
            "pepx3_4": [216, 216],
            "pepx3_5": [216, 216],
            "pepx3_6": [216, 216],
            "pepx4_1": [216, 424],
            "pepx4_2": [424, 424],
            "pepx4_3": [424, 424],
        }

        n_classes = kwargs.get("n_classes", 2)

        self.add_module(
            "conv1",
            nn.Conv2d(
                in_channels=3, out_channels=56, kernel_size=7, stride=2, padding=3
            ),
        )
        for key in filters:
            if "pool" in key:
                self.add_module(key, nn.MaxPool2d(filters[key][0], filters[key][1]))
            else:
                self.add_module(key, PEPX(filters[key][0], filters[key][1]))

        self.__forward__ = self.forward_small_net
        self.add_module("flatten", Flatten())
        self.add_module("fc1", nn.Linear(7 * 7 * 424, 512))

        self.add_module("classifier", nn.Linear(512, n_classes))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, 3, 224, 224]

        Returns:
            logits (torch.Tensor): Logits of shape [batch_size, n_classes]
        """
        return self.__forward__(x)

    def forward_small_net(self, x: torch.Tensor, target=None) -> torch.Tensor:
        """
        Forward pass of the small model.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, 3, 224, 224]
            target (torch.Tensor): Target tensor of shape [batch_size, n_classes]

        Returns:
            logits (torch.Tensor): Logits of shape [batch_size, n_classes]
        """
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)

        pepx11 = self.pepx1_1(x)
        pepx12 = self.pepx1_2(pepx11)
        pepx13 = self.pepx1_3(pepx12 + pepx11)

        pepx21 = self.pepx2_1(
            F.max_pool2d(pepx13, 2) + F.max_pool2d(pepx11, 2) + F.max_pool2d(pepx12, 2)
        )
        pepx22 = self.pepx2_2(pepx21)
        pepx23 = self.pepx2_3(pepx22 + pepx21)
        pepx24 = self.pepx2_4(pepx23 + pepx21 + pepx22)

        pepx31 = self.pepx3_1(
            F.max_pool2d(pepx24, 2)
            + F.max_pool2d(pepx21, 2)
            + F.max_pool2d(pepx22, 2)
            + F.max_pool2d(pepx23, 2)
        )
        pepx32 = self.pepx3_2(pepx31)
        pepx33 = self.pepx3_3(pepx31 + pepx32)
        pepx34 = self.pepx3_4(pepx31 + pepx32 + pepx33)
        pepx35 = self.pepx3_5(pepx31 + pepx32 + pepx33 + pepx34)
        pepx36 = self.pepx3_6(pepx31 + pepx32 + pepx33 + pepx34 + pepx35)

        pepx41 = self.pepx4_1(
            F.max_pool2d(pepx31, 2)
            + F.max_pool2d(pepx32, 2)
            + F.max_pool2d(pepx32, 2)
            + F.max_pool2d(pepx34, 2)
            + F.max_pool2d(pepx35, 2)
            + F.max_pool2d(pepx36, 2)
        )
        pepx42 = self.pepx4_2(pepx41)
        pepx43 = self.pepx4_3(pepx41 + pepx42)
        flattened = self.flatten(pepx41 + pepx42 + pepx43)

        fc1out = F.relu(self.fc1(flattened))
        logits = self.classifier(fc1out)
        return logits

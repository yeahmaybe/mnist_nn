import torch.nn as nn


class MnistNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=96, kernel_size=11, stride=2),  # (b x 96 x 14 x 14)
            nn.ReLU(),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),  # section 3.3
            nn.MaxPool2d(kernel_size=3, stride=2),  # (b x 96 x 7 x 7)
            nn.Conv2d(96, 256, 5, padding=2),  # (b x 96 x 7 x 7)
            nn.ReLU(),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2),  # (b x 96 x 3 x 3)
            nn.Conv2d(256, 384, 3, padding=1),  # (b x 96 x 3 x 3)
            nn.ReLU(),
            nn.Conv2d(384, 384, 3, padding=1),  # (b x 96 x 3 x 3)
            nn.ReLU(),
            nn.Conv2d(384, 256, 3, padding=1),  # (b x 96 x 3 x 3)
            # nn.ReLU(),
            # nn.MaxPool2d(kernel_size=3, stride=2),  # (b x 96 x 1 x 1)
        )


        # classifier is just a name for linear layers
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=256, out_features=4096),
            nn.ReLU(),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(),
            nn.Linear(in_features=4096, out_features=num_classes),
        )
        self.init_bias()  # initialize bias

    def init_bias(self):
        for layer in self.net:
            if isinstance(layer, nn.Conv2d):
                nn.init.normal_(layer.weight, mean=0, std=0.01)
                nn.init.constant_(layer.bias, 0)
        # original paper = 1 for Conv2d layers 2nd, 4th, and 5th conv layers
        # nn.init.constant_(self.net[4].bias, 1)
        # nn.init.constant_(self.net[10].bias, 1)
        # nn.init.constant_(self.net[12].bias, 1)

    def forward(self, x):
        """
        Pass the input through the net.
        Args:
            x (Tensor): input tensor
        Returns:
            output (Tensor): output tensor
        """
        x = self.net(x)
        x = x.view(x.size(0), -1)  # reduce the dimensions for linear layer input
        return self.classifier(x)

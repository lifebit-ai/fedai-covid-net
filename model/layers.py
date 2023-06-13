import torch
import torch.nn as nn


class Flatten(nn.Module):
    """
    Flattens the input. Does not affect the batch size.

    Returns:
        Flattened tensor of shape [batch_size, k]
    """

    def forward(self, input):
        return input.view(input.size(0), -1)


class PEPX(nn.Module):
    def __init__(self, n_input, n_out):
        super(PEPX, self).__init__()

        """
        • First-stage Projection: 1×1 convolutions for projecting input features to a lower dimension,

        • Expansion: 1×1 convolutions for expanding features
            to a higher dimension that is different than that of the
            input features,


        • Depth-wise Representation: efficient 3×3 depthwise convolutions for learning spatial characteristics to
            minimize computational complexity while preserving
            representational capacity,

        • Second-stage Projection: 1×1 convolutions for projecting features back to a lower dimension, and

        • Extension: 1×1 convolutions that finally extend channel dimensionality to a higher dimension to produce
            the final features.
        """

        self.network = nn.Sequential(
            nn.Conv2d(in_channels=n_input, out_channels=n_input // 2, kernel_size=1),
            nn.Conv2d(
                in_channels=n_input // 2,
                out_channels=int(3 * n_input / 4),
                kernel_size=1,
            ),
            nn.Conv2d(
                in_channels=int(3 * n_input / 4),
                out_channels=int(3 * n_input / 4),
                kernel_size=3,
                groups=int(3 * n_input / 4),
                padding=1,
            ),
            nn.Conv2d(
                in_channels=int(3 * n_input / 4),
                out_channels=n_input // 2,
                kernel_size=1,
            ),
            nn.Conv2d(in_channels=n_input // 2, out_channels=n_out, kernel_size=1),
            nn.BatchNorm2d(n_out),
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the PEPX block.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, n_input, H, W]

        Returns:
            torch.Tensor: Output tensor of shape [batch_size, n_out, H, W]
        """
        return self.network(x)

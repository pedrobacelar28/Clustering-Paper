import torch
import torch.nn as nn

class GRN(nn.Module):
    """GRN (Global Response Normalization) layer"""

    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, dim, 1))
        self.beta = nn.Parameter(torch.zeros(1, dim, 1))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=-1, keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate=0.2, original_len=2560, ith_block=1):
        super().__init__()

        # Compute the length of the signal after the downsampling
        reduced_len = original_len // (4 ** ith_block)
        self.red = reduced_len

        # Define the main branch (ConvNeXt V2 inspired)
        self.first_branch = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=10, stride=4, padding=4),
            nn.LayerNorm([out_channels, reduced_len // 4]),  # LayerNorm with dynamic shape
            nn.Conv1d(out_channels, 4*out_channels, kernel_size=1, stride=1),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            GRN(4*out_channels),
            nn.Conv1d(4*out_channels, out_channels, kernel_size=1, stride=1),
        )

        # Define the skip connection (right branch)
        self.skip_connection = nn.Sequential(
            nn.MaxPool1d(kernel_size=4, stride=4, padding=0, dilation=1),
            nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1),
        )

        # Define the second branch (ConvNeXt V2 inspired without the first conv)
        self.second_branch = nn.Sequential(
            nn.LayerNorm([out_channels, reduced_len // 4]),  # LayerNorm with dynamic shape
            nn.Conv1d(out_channels, 4*out_channels, kernel_size=1, stride=1),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            GRN(4*out_channels),
            nn.Conv1d(4*out_channels, out_channels, kernel_size=1, stride=1),
        )

    def forward(self, x):
        out = self.first_branch(x)

        residual = self.skip_connection(x)

        out = out + residual

        residual = out

        out = self.second_branch(out)

        out = out + residual

        return out


class StackedResidual(nn.Module):
    def __init__(self, channels, num_blocks, dropout_rate=0.2, original_len=2560):
        super().__init__()

        # Create a list of residual blocks
        self.blocks = nn.ModuleList()
        for i in range(num_blocks):
            in_channels = channels[i]
            out_channels = in_channels + 64
            self.blocks.append(
                ResidualBlock(in_channels, out_channels, dropout_rate=dropout_rate, original_len=original_len, ith_block=i+1)
            )

    def forward(self, x):
        for i, block in enumerate(self.blocks):
            x = block(x)

        return x


class ECG_ResNeXt(nn.Module):
    def __init__(self, n_classes, num_blocks, channels, dropout_rate=0.2, original_len=2560):
        super().__init__()

        # Modify the input layer to follow the ConvNeXt architecture
        reduced_len = original_len // 4  # Adjusting based on the initial downsampling

        self.input_layer = nn.Sequential(
            nn.Conv1d(12, 64, kernel_size=17, stride=4, padding=7, bias=False),
            nn.LayerNorm([64, reduced_len]),  # LayerNorm with dynamic shape
            nn.Conv1d(64, 4*64, kernel_size=1, stride=1, bias=False),  # 1x1 convolution
            nn.GELU(),
            nn.Dropout(dropout_rate),
            GRN(4*64),
            nn.Conv1d(4*64, 64, kernel_size=1, stride=1, bias=False),  # Another 1x1 convolution
            nn.Dropout(dropout_rate),
        )

        self.residual_blocks = StackedResidual(
            channels, num_blocks, dropout_rate=dropout_rate, original_len=original_len
        )
        self.flatten = nn.Flatten()
        self.linear = nn.Sequential(
            nn.Linear(2560, 256),
            nn.ReLU(),
            nn.Dropout(0.7),
            nn.Linear(256, n_classes),
        )

        # Call the weight initialization method after building the layers
        self._initialize_weights()

    def forward(self, x):
        x = self.input_layer(x)
        x = self.residual_blocks(x)
        x = self.flatten(x)
        logits = self.linear(x)
        return logits

    def _initialize_weights(self):
        """Initializes the weights of the model"""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
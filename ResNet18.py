import torch.nn as nn
from torch.nn.functional import relu, avg_pool2d
from avalanche.models import DynamicModule


# Define entire layers
class MLP(nn.Module):
    def __init__(self, sizes) -> None:
        super(MLP, self).__init__()
        layers = [] # It contains every layer in nn.Linear
        # Iterate for each element in sizes
        for i in range(0, len(sizes)-1):
            # Create Layer_i starting from sizes[i] like input size
            # Close the layer with next input layer's size, size[i+1]
            layers.append(nn.Linear(sizes[i], sizes[i+1]))
            # Adding Relu or others activation function if is hidden layer 
            if i < (len(sizes)-2):
                layers.append(nn.ReLU())
        # Create net with nn.Sequential
        self.net = nn.Sequential(*layers)
    def forward(self):
        return self.net(x)
            
        

# Convolution:
#   Kernel 3x3 (K = 3) 
#   RGB         (C = 3)
#   stride 1x1 (S = 1)
#   padding 1x1 (P = 1)
#   in_planes (W = ?)
# Output size (W^1) =  ((W - K + 2P) / S) + 1
def conv3x3(in_planes, out_planes, stride=1):
    # Input channels  = 3 for RGB
    return nn.Conv2d(
        in_channels=in_planes,
        out_channels=out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False
    )
    
class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1, expansion=1) -> None:
        super(BasicBlock, self).__init__()
        # Conv 3x3
        # Input: in_planes -> output: planes
        self.conv1 = conv3x3(in_planes=in_planes, out_planes=planes, stride=stride)
        # Batch normalize 1
        # Input: planes
        self.bn1 = nn.BatchNorm2d(planes)
        # Conv 3x3
        # Input: planes -> stay planes
        self.conv2 = conv3x3(in_planes=planes, out_planes=planes, stride=stride)
        # Batch normalize 2
        # Input: planes
        self.bn2 = nn.BatchNorm2d(planes)
        # Add shortcut
        # If gradient will explode, jump out of the block
        # x = x + f(x)
        self.shortcut = nn.Sequential()
        # If the block will change output size from input size, we will apply only one conv
        if stride != 1 or in_planes != expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_planes,
                    out_channels=expansion*planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),
                nn.BatchNorm2d(expansion*planes)
            )
    
    def forward(self, x):
        # Input: x
        # Relu applied top of the block
        # Apply batch normalize after convolution
        out = relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        # out = out + f(x)
        # If f(x) != out:
        #   applied shortcut
        out += self.shortcut(x)
        # Apply last relu
        out = relu(out)
        return out
        
    
        
    
class ResNet(nn.Module):
    def __init__(self, nblocks,) -> None:
        super(ResNet, self).__init__()
        pass

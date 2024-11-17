import torch
import torch.nn as nn

params_dict = {
        # Coefficients:   width,depth,res,dropout
        'efficientnet-b0': (1.0, 1.0, 224, 0.2),
        'efficientnet-b1': (1.0, 1.1, 240, 0.2),
        'efficientnet-b2': (1.1, 1.2, 260, 0.3),
        'efficientnet-b3': (1.2, 1.4, 300, 0.3),
        'efficientnet-b4': (1.4, 1.8, 380, 0.4),
        'efficientnet-b5': (1.6, 2.2, 456, 0.4),
        'efficientnet-b6': (1.8, 2.6, 528, 0.5),
        'efficientnet-b7': (2.0, 3.1, 600, 0.5),
        'efficientnet-b8': (2.2, 3.6, 672, 0.5),
        'efficientnet-l2': (4.3, 5.3, 800, 0.5),
    }

class MBConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, expansion_factor=6):
        super(MBConv, self).__init__()
        
        self.stride = stride
        hidden_dim = in_channels * expansion_factor
        self.use_residual = self.stride == 1 and in_channels == out_channels
        
        layers = []
        
        # Expansion phase
        if expansion_factor != 1:
            layers.extend([
                nn.Conv2d(in_channels, hidden_dim, 1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True)
            ])
        
        # Depthwise convolution phase
        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride=stride, padding=kernel_size // 2, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True)
        ])
        
        # Projection phase
        layers.extend([
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        ])
        
        self.conv = nn.Sequential(*layers)
        
    def forward(self, x):
        if self.use_residual:
            return x + self.conv(x)
        else:
            return self.conv(x)

class EfficientNetB0(nn.Module):
    def __init__(self, num_classes=1000):
        super(EfficientNetB0, self).__init__()
        
        self.stage1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True)
        )
        
        self.stage2 = MBConv(32, 16, kernel_size=3, stride=1)
        
        self.stage3 = nn.Sequential(
            MBConv(16, 24, kernel_size=3, stride=2),
            MBConv(24, 24, kernel_size=3, stride=1)
        )
        
        self.stage4 = nn.Sequential(
            MBConv(24, 40, kernel_size=5, stride=2),
            MBConv(40, 40, kernel_size=5, stride=1)
        )
        
        self.stage5 = nn.Sequential(
            MBConv(40, 80, kernel_size=3, stride=2),
            MBConv(80, 80, kernel_size=3, stride=1),
            MBConv(80, 80, kernel_size=3, stride=1)
        )
        
        self.stage6 = nn.Sequential(
            MBConv(80, 112, kernel_size=5, stride=1),
            MBConv(112, 112, kernel_size=5, stride=1),
            MBConv(112, 112, kernel_size=5, stride=1)
        )
        
        self.stage7 = nn.Sequential(
            MBConv(112, 192, kernel_size=5, stride=2),
            MBConv(192, 192, kernel_size=5, stride=1),
            MBConv(192, 192, kernel_size=5, stride=1),
            MBConv(192, 192, kernel_size=5, stride=1)
        )
        
        self.stage8 = MBConv(192, 320, kernel_size=3, stride=1)
        
        self.stage9 = nn.Sequential(
            nn.Conv2d(320, 1280, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(1280),
            nn.ReLU6(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(1280, num_classes)
        )
        
    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        x = self.stage6(x)
        x = self.stage7(x)
        x = self.stage8(x)
        x = self.stage9(x)
        return x


def compound_scaling_search(phi):
    # depth, width ,resolution
    best_alpha, best_beta, best_gamma = 1, 1, 1
    best_diff = float('inf')
    
    # Grid search
    for alpha in np.arange(1.0, 6.0, 0.1):
        for beta in np.arange(1.0, 6.0, 0.1):
            for gamma in np.arange(1.0, 6.0, 0.1):
                # Check the constraint
                constraint = alpha * (beta ** 2) * (gamma ** 2)
                diff = abs(constraint - 2)
                
                if diff < best_diff:
                    best_diff = diff
                    best_alpha, best_beta, best_gamma = alpha, beta, gamma
    
    return best_alpha, (best_beta ** phi), (best_gamma ** phi)

def compound_scaling(phi, alpha, beta, gamma):
    assert abs((alpha * (beta ** 2) * (gamma ** 2)) - 2) < 1, "The value of the equation is too far from 2."
    return alpha, beta ** phi, gamma ** phi
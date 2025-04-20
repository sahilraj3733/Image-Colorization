import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Dataset,DataLoader
import torchvision.transforms as transforms

# --- Title ---
st.title("🎨 Grayscale to Color Image Colorization")



## Down Sampling Block
class down_block(nn.Module):
    def __init__(self, in_channels, out_channels, norm=True, act="leaky_relu", use_dropout=False):
        super(down_block, self).__init__()
        layers = []
        
        # Convolution Layers, output is approx half size due to stride = 2, didnt use MaxPooling here
        layers += [nn.Conv2d(in_channels, out_channels, kernel_size = 4, stride = 2, padding = 1, bias=False)]
        
        # BatchNorm, will be turned off for input down block
        if norm: layers += [nn.BatchNorm2d(out_channels)]
            
        # Two options for ReLU activation, LeakyReLU default for down block
        if act == "leaky_relu":  layers += [nn.LeakyReLU(0.2, inplace=True)]
        elif act == "relu": layers += [nn.ReLU(inplace=True)]
            
        if use_dropout: layers += [nn.Dropout(0.5)]
        
        self.conv = nn.Sequential(*layers)
        
        
    def forward(self, x):
        x = self.conv(x)
        return x
        
## Up sampling Block
        
class up_block(nn.Module):
    def __init__(self, in_channels, out_channels, norm=True, act="relu", use_dropout=False):
        super(up_block, self).__init__()
        layers = []
        
        # Two options for ReLU activation, LeakyReLU default for down block
        if act == "leaky_relu":  layers += [nn.LeakyReLU(0.2, inplace=True)]
        elif act == "relu": layers += [nn.ReLU(inplace=True)]
        
        # ConvTranspose2D 
        layers += [nn.ConvTranspose2d(in_channels, out_channels, kernel_size = 4, stride = 2, padding = 1, bias=False)]
        
        # BatchNorm, will be turned off for output up block
        if norm: layers += [nn.BatchNorm2d(out_channels)]
            
        if use_dropout: layers += [nn.Dropout(0.5)]
            
        self.conv = nn.Sequential(*layers)
        
    def forward(self, x):
        x = self.conv(x)
        return x
        
class Unet(nn.Module):
    def __init__(self, in_channels=1, num_filters=64):
        super().__init__()
        # Down Sampling 
        self.input_down = down_block(in_channels = in_channels, out_channels = num_filters, norm=False) # norm=False for input down block
        
        self.down1 = down_block(in_channels = num_filters, out_channels = num_filters*2)
        self.down2 = down_block(in_channels = num_filters*2, out_channels = num_filters*4)
        self.down3 = down_block(in_channels = num_filters*4, out_channels = num_filters*8)
        
        self.down4 = down_block(in_channels = num_filters*8, out_channels = num_filters*8)
        self.down5 = down_block(in_channels = num_filters*8, out_channels = num_filters*8)
        self.down6 = down_block(in_channels = num_filters*8, out_channels = num_filters*8)
        
        # Bottleneck 
        
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels = num_filters*8, out_channels = num_filters*8, kernel_size = 4, stride = 2, padding = 1, bias=False),
            up_block(in_channels = num_filters*8, out_channels = num_filters*8)
        )
        
        # Up Sampling
        
        self.up1 = up_block(in_channels = num_filters*8*2, out_channels = num_filters*8, use_dropout=True)
        self.up2 = up_block(in_channels = num_filters*8*2, out_channels = num_filters*8, use_dropout=True)
        self.up3 = up_block(in_channels = num_filters*8*2, out_channels = num_filters*8, use_dropout=True)
        
        self.up4 = up_block(in_channels = num_filters*8*2, out_channels = num_filters*4)
        self.up5 = up_block(in_channels = num_filters*4*2, out_channels = num_filters*2)
        self.up6 = up_block(in_channels = num_filters*2*2, out_channels = num_filters)
        
        self.output_up = nn.Sequential(
            up_block(in_channels = num_filters*2, out_channels = in_channels*2, norm=False), 
            nn.Tanh()
        )
        
    def forward(self, x):
        d1 = self.input_down(x)
        d2 = self.down1(d1)
        d3 = self.down2(d2)
        d4 = self.down3(d3)
        d5 = self.down4(d4)
        d6 = self.down5(d5)
        d7 = self.down6(d6)
        bottleneck = self.bottleneck(d7)
        up1 = self.up1(torch.cat([bottleneck, d7], 1))
        up2 = self.up2(torch.cat([up1, d6], 1))
        up3 = self.up3(torch.cat([up2, d5], 1))
        up4 = self.up4(torch.cat([up3, d4], 1))
        up5 = self.up5(torch.cat([up4, d3], 1))
        up6 = self.up6(torch.cat([up5, d2], 1))
        output = self.output_up(torch.cat([up6, d1], 1))
        
        return output

class PatchDiscriminator(nn.Module):
    def __init__(self, in_channels, num_filters = 64, n_middle = 3):
        super().__init__()
        layers = []
        layers.append(nn.Conv2d(in_channels, num_filters,4, 2, 1, bias = True))
        layers.append(nn.LeakyReLU(0.2, True))

        for i in range(n_middle):
            layers.append(nn.Conv2d(num_filters * 2 ** i, num_filters * 2 ** (i+1), 4, stride = 1 if i == (n_middle - 1) else 2, padding = 1, bias = False))
            layers.append(nn.BatchNorm2d(num_filters * 2 ** (i+1)))
            layers.append(nn.LeakyReLU(0.2, True))
                                 
        layers.append(nn.Conv2d(num_filters * 2 ** n_middle, 1, 4, 1 ,1, bias = True))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class GANLoss(nn.Module):
    def __init__(self, gan_mode='vanilla', real_label=1.0, fake_label=0.0):
        super().__init__()
        self.register_buffer('real_label', torch.tensor(real_label))
        self.register_buffer('fake_label', torch.tensor(fake_label))
#         self.real_label = torch.tensor(real_label)
#         self.fake_label = torch.tensor(fake_label)
        if gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
    
    def get_labels(self, preds, target_is_real):
        if target_is_real:
            labels = self.real_label
        else:
            labels = self.fake_label
        return labels.expand_as(preds)
    
    def __call__(self, preds, target_is_real):
        labels = self.get_labels(preds, target_is_real)
        loss = self.loss(preds, labels)
        return loss




def init_weights(net, init='norm', gain=0.02):
    
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and 'Conv' in classname:
            if init == 'norm':
                nn.init.normal_(m.weight.data, mean=0.0, std=gain)
            elif init == 'xavier':
                nn.init.xavier_normal_(m.weight.data, gain=gain)
            elif init == 'kaiming':
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif 'BatchNorm2d' in classname:
            nn.init.normal_(m.weight.data, 1., gain)
            nn.init.constant_(m.bias.data, 0.)
            
    net.apply(init_func)
    print(f"model initialized with {init} initialization")
    return net

def init_model(model, device):
    model = model.to(device)
    model = init_weights(model)
    return model

class MainModel(nn.Module):
    def __init__(self, net_G=None, lr_G=2e-4, lr_D=2e-4, 
                 beta1=0.5, beta2=0.999, lambda_L1=100.):
        super().__init__()
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lambda_L1 = lambda_L1
        
        if net_G is None:
            self.net_G = init_model(Unet(in_channels=1, num_filters=64), self.device)
        else:
            self.net_G = net_G.to(self.device)
        self.net_D = init_model(PatchDiscriminator(in_channels=3, num_filters = 64, n_middle = 3), self.device)
        self.GANcriterion = GANLoss(gan_mode='vanilla').to(self.device)
        self.L1criterion = nn.L1Loss()
        self.opt_G = optim.Adam(self.net_G.parameters(), lr=lr_G, betas=(beta1, beta2))
        self.opt_D = optim.Adam(self.net_D.parameters(), lr=lr_D, betas=(beta1, beta2))
    
    def set_requires_grad(self, model, requires_grad=True):
        for p in model.parameters():
            p.requires_grad = requires_grad
        
    def setup_input(self, data):
        L, ab, rgb_img = data
        self.L = L.to(self.device)
        self.ab = ab.to(self.device)
        
    def forward(self):
        self.fake_color = self.net_G(self.L)
    
    def backward_D(self):
        fake_image = torch.cat([self.L, self.fake_color], dim=1)
        fake_preds = self.net_D(fake_image.detach())
        self.loss_D_fake = self.GANcriterion(fake_preds, False)
        real_image = torch.cat([self.L, self.ab], dim=1)
        real_preds = self.net_D(real_image)
        self.loss_D_real = self.GANcriterion(real_preds, True)
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()
    
    def backward_G(self):
        fake_image = torch.cat([self.L, self.fake_color], dim=1)
        fake_preds = self.net_D(fake_image)
        self.loss_G_GAN = self.GANcriterion(fake_preds, True)
        self.loss_G_L1 = self.L1criterion(self.fake_color, self.ab) * self.lambda_L1
        self.loss_G = self.loss_G_GAN + self.loss_G_L1
        self.loss_G.backward()
    
    def optimize(self):
        self.forward()
        self.net_D.train()
        self.set_requires_grad(self.net_D, True)
        self.opt_D.zero_grad()
        self.backward_D()
        self.opt_D.step()
        
        self.net_G.train()
        self.set_requires_grad(self.net_D, False)
        self.opt_G.zero_grad()
        self.backward_G()
        self.opt_G.step()
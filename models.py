import torch
import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):
    def __init__ (self, noise_size=201, cube_resolution=32):
        super(Generator, self).__init__()
        
        self.noise_size = noise_size
        self.cube_resolution = cube_resolution
        
        self.gen_conv1 = torch.nn.ConvTranspose3d(self.noise_size, 256, kernel_size=[4,4,4], stride=[2,2,2], padding=1)
        self.gen_conv2 = torch.nn.ConvTranspose3d(256, 128, kernel_size=[4,4,4], stride=[2,2,2], padding=1)
        self.gen_conv3 = torch.nn.ConvTranspose3d(128, 64, kernel_size=[4,4,4], stride=[2,2,2], padding=1)
        self.gen_conv4 = torch.nn.ConvTranspose3d(64, 32, kernel_size=[4,4,4], stride=[2,2,2], padding=1)
        self.gen_conv5 = torch.nn.ConvTranspose3d(32, 1, kernel_size=[4,4,4], stride=[2,2,2], padding=1)
        
        self.gen_bn1 = nn.BatchNorm3d(256)
        self.gen_bn2 = nn.BatchNorm3d(128)
        self.gen_bn3 = nn.BatchNorm3d(64)
        self.gen_bn4 = nn.BatchNorm3d(32)
        
    
    def forward(self, x, condition):
        
        condition_tensor = condition * torch.ones([x.shape[0],1], device=x.device)
        x = torch.cat([x, condition_tensor], dim=1)
        x = x.view(x.shape[0],self.noise_size,1,1,1)
        
        x = F.relu(self.gen_bn1(self.gen_conv1(x)))
        x = F.relu(self.gen_bn2(self.gen_conv2(x)))
        x = F.relu(self.gen_bn3(self.gen_conv3(x)))
        x = F.relu(self.gen_bn4(self.gen_conv4(x)))
        x = self.gen_conv5(x)
        x = torch.sigmoid(x)
        
        return x.squeeze()
                      
class Discriminator(nn.Module):
    def __init__ (self, cube_resolution=32):
        super(Discriminator, self).__init__()
        
        self.cube_resolution = cube_resolution
        
        self.disc_conv1 = torch.nn.Conv3d(2, 32, kernel_size=[4,4,4], stride=[2,2,2], padding=1)
        self.disc_conv2 = torch.nn.Conv3d(32, 64, kernel_size=[4,4,4], stride=[2,2,2], padding=1)
        self.disc_conv3 = torch.nn.Conv3d(64, 128, kernel_size=[4,4,4], stride=[2,2,2], padding=1)
        self.disc_conv4 = torch.nn.Conv3d(128, 256, kernel_size=[4,4,4], stride=[2,2,2], padding=1)
        self.disc_conv5 = torch.nn.Conv3d(256, 1, kernel_size=[4,4,4], stride=[2,2,2], padding=1)
        
        self.disc_bn1 = nn.BatchNorm3d(32)
        self.disc_bn2 = nn.BatchNorm3d(64)
        self.disc_bn3 = nn.BatchNorm3d(128)
        self.disc_bn4 = nn.BatchNorm3d(256)
        
        self.LRelu = nn.LeakyReLU(0.2, True)
    
    def forward(self, x, condition):
        
        x = x.unsqueeze(1)
        condition_tensor =  condition * torch.ones_like(x, device=x.device)
        x = torch.cat([x, condition_tensor], dim=1)
        
        x = self.LRelu(self.disc_bn1(self.disc_conv1(x)))
        x = self.LRelu(self.disc_bn2(self.disc_conv2(x)))
        x = self.LRelu(self.disc_bn3(self.disc_conv3(x)))
        x = self.LRelu(self.disc_bn4(self.disc_conv4(x)))
        x = self.disc_conv5(x)
        x = torch.sigmoid(x)
        
        return x.squeeze()
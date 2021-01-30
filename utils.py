import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

from tqdm.notebook import tqdm
from IPython.display import clear_output

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

###########################
#### HELPER FUNCTIONS #####
###########################

def show_images(pred, real, log, num_images=25):
    pred_unf = ((pred+1)/2).detach().cpu()
    real_unf = ((real+1)/2).detach().cpu()
    pred_grid = make_grid(pred_unf[:num_images], nrow=5)
    real_grid = make_grid(real_unf[:num_images], nrow=5)
    fig = plt.figure()
    ax1, ax2 = fig.subplots(1, 2)
    plt.title(log)
    ax1.imshow(pred_grid.permute(1, 2, 0).squeeze())
    ax2.imshow(real_grid.permute(1, 2, 0).squeeze())
    plt.show()
    
def initialize_weights(layer, mean=0.0, std=0.02):
    if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.ConvTranspose2d):
        torch.nn.init.normal_(layer.weight, mean, std)
    if isinstance(layer, nn.BatchNorm2d):
        torch.nn.init.normal_(layer.weight, mean, std)
        torch.nn.init.constant_(layer.bias, 0)
        
##################################
######## ConvBnReLU block ########
##################################

class ConvBnReLU(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size=3, stride=2):
        super().__init__()
        self.block=nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())
        
    def forward(self, x):
        return self.block(x)
    
##################################
##### ConvBnLeakyReLU block ######
##################################

class ConvBnLeakyReLU(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size=4, stride=2, alpha=0.2):
        super().__init__()
        self.block=nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(alpha))
        
    def forward(self, x):
        return self.block(x)
    
##################################
########### GENERATOR ############
##################################

class Generator(nn.Module):
    def __init__(self, in_channels, hid_channels, out_channels,
                 kernel_size=3, stride=2):
        super().__init__()
        self.model = nn.Sequential(
            ConvBnReLU(in_channels, 4*hid_channels, kernel_size, stride),
            ConvBnReLU(4*hid_channels, 2*hid_channels, kernel_size=4, stride=1),
            ConvBnReLU(2*hid_channels, hid_channels, kernel_size, stride),
            nn.ConvTranspose2d(hid_channels, out_channels, kernel_size=4, stride=stride),
            nn.Tanh())
        
    def forward(self, x):
        return self.model(x)
    
##################################
########### CRITIC ###############
##################################

class Critic(nn.Module):
    def __init__(self, in_channels, hid_channels,
                 kernel_size=4, stride=2):
        super().__init__()
        self.model = nn.Sequential(
            ConvBnLeakyReLU(in_channels, hid_channels, kernel_size, stride),
            ConvBnLeakyReLU(hid_channels, 2*hid_channels, kernel_size, stride),
            nn.Conv2d(2*hid_channels, 1, kernel_size, stride))
        
    def forward(self, x):
        out = self.model(x)
        return out.view(out.size(0), -1)
    
##################################
######## GENERATOR`s LOSS ########
##################################

class GeneratorLoss(nn.Module):
    def __init__(self,):
        super().__init__()
        
    def forward(self, cfpred):
        loss = -torch.mean(cfpred)
        return loss
    
##################################
######## CRITIC`s PENALTY ########
##################################

class CriticPenalty(nn.Module):
    def __init(self,):
        super().__init__()
        
    def forward(self, mixed, scores):
        grad=torch.autograd.grad(inputs=mixed, outputs=scores,
                            grad_outputs=torch.ones_like(scores),
                            create_graph=True, retain_graph=True)[0]
        grad_=torch.norm(grad.view(grad.size(0), -1), p=2, dim=1)
        penalty=torch.mean((1. - grad_)**2)
        return penalty
    
##################################
########## CRITIC`s LOSS #########
##################################
    
class CriticLoss(nn.Module):
    def __init__(self, lambda_):
        super().__init__()
        self.lambda_=lambda_
        
    def forward(self, fake_pred, real_pred, penalty):
        loss = torch.mean(fake_pred - real_pred + self.lambda_*penalty)
        return loss
    
##################################
############ TRAINER #############
##################################

class Trainer():
    def __init__(self, Generator, Critic, G_optimizer, C_optimizer, lambda_=10, device='cuda:0'):
        self.G = Generator.to(device)
        self.C = Critic.to(device)
        self.G_optim = G_optimizer
        self.C_optim = C_optimizer
        
        self.G_criterion=GeneratorLoss().to(device)
        self.C_penalty=CriticPenalty().to(device)
        self.C_criterion=CriticLoss(lambda_).to(device)
        self.results={'G_loss':[], 'C_loss':[]}
        
    def fit(self, dataloader, epochs=30, repeat=5, device='cuda:0'):
        for epoch in range(1, epochs+1):
            G_losses=[]
            C_losses=[]

            log = f'::::: Epoch {epoch}/{epochs} :::::'
            for real, _ in tqdm(dataloader):
                real = real.to(device)
                # CRITIC
                C_loss=0.
                for _ in range(repeat):
                    self.C_optim.zero_grad()
                    noise = torch.randn((real.size(0), 64, 1, 1)).to(device)
                    fake = self.G(noise).detach()
                    fake_pred = self.C(fake)
                    real_pred = self.C(real)
                    
                    alpha = torch.rand((real.size(0), 1, 1, 1)).requires_grad_().to(device)
                    mixed = alpha* real + (1-alpha)* fake
                    scores = self.C(mixed)
                    # CRITIC`s PENALTY
                    penalty = self.C_penalty(mixed, scores)
                    # CRITIC`s LOSS
                    loss = self.C_criterion(fake_pred, real_pred, penalty)
                    C_loss += loss.item()/repeat
                    loss.backward(retain_graph=True)
                    self.C_optim.step()
                    
                C_losses.append(C_loss)
                # GENERATOR
                self.G_optim.zero_grad()
                noise = torch.randn((real.size(0), 64, 1, 1)).to(device)
                # GENERATOR`s LOSS
                fake = self.G(noise)
                fake_pred = self.C(fake)
                G_loss = self.G_criterion(fake_pred)

                G_losses.append(G_loss.item())
                G_loss.backward()
                self.G_optim.step()
                template = f'::: Generator Loss: {G_loss.item():.3f} | Critic Loss: {C_loss:.3f} :::'

            self.results['G_loss'].append(np.mean(G_losses))
            self.results['C_loss'].append(np.mean(C_losses))
            clear_output(wait=True)
            show_images(fake, real, log+template)

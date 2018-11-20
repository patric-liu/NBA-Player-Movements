import torch
from torch import nn
import torch.nn.functional as F

class NBA_AE(nn.Module):
    def __init__(self,x_size=23,y_size=20, z_size=10):
        super(NBA_AE, self).__init__()
        self.x_size = x_size
        self.y_size = y_size
        self.z_size = z_size

        self.f_en_1 = nn.Linear(x_size, 64)
        self.f_en_2 = nn.Linear(64, 64)
        self.f_en_z = nn.Linear(64, z_size)

        self.f_de_1 = nn.Linear(z_size, 64)
        self.f_de_2 = nn.Linear(64, 64)
        self.f_de_y = nn.Linear(64, y_size)

        self.reconstruction_loss = nn.MSELoss()

    def forward(self, x):
        x = x.float()
        latent_z = self.f_en_z(self.f_en_2(self.f_en_1(x)))
        y_hat_next = self.f_de_y(self.f_de_2(self.f_de_1(latent_z)))
        return y_hat_next

    def loss_funct(self, y_next_hat, x_next):
        squared_error = self.reconstruction_loss(y_next_hat, x_next.float())
        return squared_error
import torch
from torch import nn
import torch.nn.functional as F

class NBA_AE(nn.Module):
    def __init__(self,x_size=23,y_size=23, z_size=64):
        super(NBA_AE, self).__init__()
        self.x_size = x_size
        self.y_size = y_size
        self.z_size = z_size
        width = 64
        self.f_en_1 = nn.Linear(x_size, width)
        self.f_en_2 = nn.Linear(width, width)
        self.f_en_z = nn.Linear(width, z_size)

        self.f_de_1 = nn.Linear(z_size, width)
        self.f_de_2 = nn.Linear(width, y_size)
        #self.f_de_y = nn.Linear(32, y_size)

        self.reconstruction_loss = nn.MSELoss()

    def forward(self, x):
        x = x.float()
        h1 = F.relu(self.f_en_1(x))
        h2 = F.relu(self.f_en_2(h1))
        # Not really latent at the moment, just a bottle neck
        latent_z = F.relu(self.f_en_z(h2))

        h3 = F.relu(self.f_de_1(latent_z))
        y_hat_next = F.relu(self.f_de_2(h3))
       # y_hat_next = F.relu(self.f_de_y(h4))

        return y_hat_next

    def loss_funct(self, y_next_hat, x_next):
        squared_error = self.reconstruction_loss(y_next_hat, x_next.float())
        return squared_error
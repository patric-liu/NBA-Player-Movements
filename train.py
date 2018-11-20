import torch
import numpy as np
from vae_model import NBA_AE
from game_dataset import GameDataset
from torch import optim

def train(model, n_epochs=100, batch_size=1028, learning_rate=1e-5):


    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    model.train()
    X = GameDataset('./data/2016.NBA.Raw.SportVU.Game.Logs/')
    print('Data loaded')
    dataset_loader = torch.utils.data.DataLoader(dataset=X,
                                                batch_size=batch_size,
                                                shuffle=False)
    print('Data Prepped')
    n_batches = len(X)//batch_size
    for i in range(n_epochs):
        train_loss = 0
        for x, y in dataset_loader:
            y_hat = model(x)
            loss = model.loss_funct(y_hat, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.data[0]
        print("epoch: {}, Average training loss: {}".format(i,train_loss/n_batches))

    return model


model = NBA_AE()
train(model)
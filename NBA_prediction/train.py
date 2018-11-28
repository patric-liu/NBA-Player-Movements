import torch
import numpy as np
from vae_model import NBA_AE
from game_dataset import GameDataset
from torch import optim
import matplotlib.pyplot as plt

def train(model, n_epochs=200, batch_size=1024, learning_rate=1e-3):


    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    model.train()
    X = GameDataset('../data/2016.NBA.Raw.SportVU.Game.Logs/',0,1)
    print('Data loaded')
    dataset_loader = torch.utils.data.DataLoader(dataset=X,
                                                batch_size=batch_size,
                                                shuffle=False)
    print('Data Prepped')
    n_batches = len(X)//batch_size
    loss_n = np.zeros(n_epochs)
    for i in range(n_epochs):
        train_loss = 0
        for x, y in dataset_loader:
            y_hat = model(x)
            loss = model.loss_funct(y_hat, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.data[0]
        loss_n[i] = train_loss/n_batches
        print("epoch: {}, Average training loss: {}".format(i,train_loss/n_batches))

    return model, loss_n

def evaluate(model, T=100):
    model.eval()
    X = GameDataset('../data/2016.NBA.Raw.SportVU.Game.Logs/',0,1)
    x_predict = np.zeros((100,2))
    x_actual = np.zeros((100,2))

    dataset_loader = torch.utils.data.DataLoader(dataset=X,
                                                batch_size=1,
                                                shuffle=False)
    i = 0
    for x, y in dataset_loader:

        if i == 0:
            x_predicted = x
        
        x_predicted = model(x_predicted)

        x_predict[i,:] = x_predicted.detach().numpy()[0,0:2]

        x_actual[i,:] = y.numpy()[0,0:2]
        i += 1
        if i ==T: break

    return x_predict, x_actual




model = NBA_AE()
model.load_state_dict(torch.load("./models/vae_1024_1_game"))
#model, losses = train(model)
predicted, actual = evaluate(model)
print(predicted)
print(actual)
#plt.plot(losses)
#plt.xlabel("MSE loss")
#plt.ylabel("Epoch")
#torch.save(model.state_dict(), "./models/vae_1024_1_game")
#plt.show()
#plt.clf()
plt.plot(predicted[:,0],predicted[:,1], label="predicted player position from starting position")
plt.plot(actual[:,0], actual[:,1], label="actual player position")
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.legend()
plt.show()
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import glob
from itertools import chain

class GameDataset(Dataset):

    def __init__(self, game_dir):
        #x: [ob, action]
        #y: [next_ob]
        self.in_state = []
        self.out_state = []
        files = glob.glob(game_dir+'**/*.json')
        n_games = len(files)
        sum_n = 0
        for f in range(5):
        	data_frame = pd.read_json(files[f])
        	n_events = len(data_frame)
        	for e in range(n_events):
        		moments = data_frame['events'][e]['moments']
        		for m in range(len(moments)-1):
        			players = [l[2:4] for l in moments[m][5][1:]]
        			ball = moments[m][5][0][2:]
        			players.append(ball)
        			x = np.array([item for sublist in players for item in sublist],dtype=np.float64)
        			if len(x) != 23: continue
        			players_next = [l[2:4] for l in moments[m+1][5][1:]]
        			y = np.array([item for sublist in players_next for item in sublist],dtype=np.float64)
        			if len(y) != 20: continue
        			self.in_state.append(x)
        			self.out_state.append(y)

    def __len__(self):
        return len(self.in_state)

    def __getitem__(self, idx):
        return (self.in_state[idx], self.out_state[idx])

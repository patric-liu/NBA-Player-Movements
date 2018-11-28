from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import glob
from itertools import chain

class GameDataset(Dataset):

    def __init__(self, game_dir, start, end):
        self.in_state = [] # all 2D player states + 3D ball state
        self.out_state = [] # all 2D player states. We dont really care where the ball goes?
        # Get all json files in sub directories
        files = glob.glob(game_dir+'**/*.json')
        n_games = len(files)
        sum_n = 0
        for f in range(start,end):
            # iterate n games
            data_frame = pd.read_json(files[f])
            n_events = len(data_frame)
            for e in range(n_events):
                # iterate through each play
                moments = data_frame['events'][e]['moments']
                for m in range(len(moments)-1):
                    # make player states and ball state one big vector
                    players = [l[2:4] for l in moments[m][5][1:]]
                    ball = moments[m][5][0][2:]
                    players.append(ball)
                    quarter = moments[m][0]
                    if quarter > 2: break
                    # if quarter > 2: quarter = 1
                    # else: quarter = 0
                    # players.append([quarter])
                    x = np.array([item for sublist in players for item in sublist],dtype=np.float64)
                    # Throw out sample if there arent 10 players on the court for some reason.
                    if len(x) != 23: continue
                    players_next = [l[2:4] for l in moments[m+1][5][1:]]
                    ball = moments[m+1][5][0][2:]
                    players_next.append(ball)
                    y = np.array([item for sublist in players_next for item in sublist],dtype=np.float64)
                    if len(y) != 23: continue

                    self.in_state.append(x)
                    self.out_state.append(y)

    def __len__(self):
        return len(self.in_state)

    def __getitem__(self, idx):
        return (self.in_state[idx], self.out_state[idx])

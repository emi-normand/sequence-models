import torch
import numpy as np
import random

# State transition table
TRANSITIONS = [
    [('T', 1), ('P', 2)],  # 0=B
    [('X', 3), ('S', 1)],  # 1=BT
    [('V', 4), ('T', 2)],  # 2=BP
    [('X', 2), ('S', 5)],  # 3=BTX
    [('P', 3), ('V', 5)],  # 4=BPV
    [('E', -1)],  # 5=BTXS
]

# Symbol encoding
SYMS = {'T': 0, 'P': 1, 'X': 2, 'S': 3, 'V': 4, 'B': 5, 'E': 6}

class ERG(torch.utils.data.Dataset):
    """ Implements Embedded Reber Grammar Problem"""
    def __init__(self, size):
        self.data = []
        for i in range(size):
            self.data.append(self.generate())
    
    def generate(self):
        """ Generates a single sample """
        idx = 0
        out = 'B'
        while idx != -1:
            ts = TRANSITIONS[idx]
            symbol, idx = random.choice(ts)
            out += symbol
        return out
    
    def str_to_vec(self,s):
        """Convert a Reber string to a sequence of unit vectors."""
        a = torch.zeros((len(s), len(SYMS)))
        for i, c in enumerate(s):
            a[i][SYMS[c]] = 1
        return a

    def str_to_next(self,s):
        """Given a Reber string, return a vectorized sequence of next chars.
        This is the target output of the Neural Net."""
        out = torch.zeros((len(s), len(SYMS)))
        idx = 0
        for i, c in enumerate(s[1:]):
            ts = TRANSITIONS[idx]
            for next_c, _ in ts:
                out[i, SYMS[next_c]] = 1

            next_idx = [j for next_c, j in ts if next_c == c]
            assert len(next_idx) == 1
            idx = next_idx[0]

        return out
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # print(self.str_to_vec(self.data[idx]))
        # print(self.str_to_next(self.data[idx]))
        return self.str_to_vec(self.data[idx]), self.str_to_next(self.data[idx])
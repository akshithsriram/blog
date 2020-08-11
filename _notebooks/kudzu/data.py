import numpy as np

class Data:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.length = len(self)
        # Start an array index for later
        self.starts = np.arange(0, self.length)
        
    def __len__(self):
        return self.y.shape[0]
    
    def __getitem__(self, i):
        return self.x[i], self.y[i]
    
# idea+implementation taken from fast.ai
class Sampler:
    def __init__(self, data, bs, shuffle=False):
        self.n = len(data.y)
        self.idxs = np.arange(0, self.n)
        self.bs = bs
        self.shuffle = shuffle
        
    def __iter__(self):
        if self.shuffle:
            np.random.shuffle(self.idxs)
        for i in range(0, self.n, self.bs): 
            yield self.idxs[i:i+self.bs]

# this dataloader uses the Sampler
class Dataloader():
    def __init__(self, data, sampler): 
        self.data = data
        self.sampler = sampler
        self.bs = self.sampler.bs
        self.current_batch = 0
        
    def __iter__(self):
        for idxsample in self.sampler:
            yield self.data[idxsample]
            self.current_batch += 1

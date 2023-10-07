import torch

class dataset(torch.utils.data.Dataset) :
    def __init__(self, context_length) :
        data = open('Shakespeare.txt', 'r').read()[:10000]
        uni = list(set(data))
        uni.sort()

        ctoi = {char:i for i, char in enumerate(uni)}
        self.itoc = {i:char for i, char in enumerate(uni)}

        xy = []
        x = []
        y = []
        
        for i in data :
            xy.append(ctoi[i])
        
        for i in range(len(xy)-context_length) :
            x.append(xy[i:i+context_length])
            y.append(xy[i+context_length])
            
        self.x, self.y = torch.tensor(x), torch.tensor(y)
        
    def __len__(self) :
        return self.x.shape[0]
    
    def __getitem__(self, index) :
        return (self.x[index], self.y[index])
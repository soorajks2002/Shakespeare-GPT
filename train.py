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
    
class PostionalEncoding () :
    def __init__(self, context_length, embedding_size) :
        div_term = torch.arange(embedding_size)/embedding_size
        div_term = 1/torch.pow(1e4, div_term)
        div_term = div_term.view(1, embedding_size)
        
        position_matrix = torch.arange(1, context_length+1).view(context_length,1)
        
        self.position_embedding = position_matrix*div_term
        
        self.position_embedding[:, 0::2] = torch.sin(self.position_embedding[:, 0::2])
        self.position_embedding[:, 1::2] = torch.cos(self.position_embedding[:, 1::2])
    
    def __call__(self) :            
        return self.position_embedding
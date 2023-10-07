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
    
class AttentionHead(torch.nn.Module) :
    def __init__(self, context_length, embed_dim, num_heads) :
        super(AttentionHead, self).__init__()
        self.attention = torch.nn.MultiheadAttention(embed_dim, num_heads)
        self.mask = torch.tril(torch.ones(context_length, context_length), diagonal=0)
        self.mask = self.mask.masked_fill(self.mask==0, float('-inf'))
        
    def __call__(self, x) :
        x = x.permute(1,0,2)
        out, _ = self.attention(x, x, x, attn_mask=self.mask)

        return out.permute(1,0,2)
    
class DeocderNN(torch.nn.Module) :
    def __init__(self, context_length, embedding_dim, n_tokens, n_attn_heads, n_neurons) :
        super(DeocderNN, self).__init__()
        
        self.input_embeddings = torch.nn.Embedding(n_tokens, embedding_dim)
        self.position_encodings = PostionalEncoding(context_length, embedding_dim)
        
        self.attention = AttentionHead(context_length, embedding_dim, n_attn_heads)
        
        self.flatten = torch.nn.Flatten()
        size_inputs = context_length*embedding_dim
        
        self.linear = torch.nn.Linear(size_inputs, n_neurons)
        self.normalize = torch.nn.LayerNorm(n_neurons)
        self.activation = torch.nn.Tanh()
        
        self.output = torch.nn.Linear(n_neurons, n_tokens)
        
    def forward(self, x) :
        x = self.input_embeddings(x)
        
        out = self.attention(x+self.position_encodings())

        out = out + x  # Residual connection
        
        out = self.flatten(out)
        out = self.linear(out)
        out = self.normalize(out)
        out = self.activation(out)
        
        out = self.output(out)
        
        return out
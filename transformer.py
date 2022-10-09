from torch import nn
import torch
import math

class Transformer(nn.Module):
    def __init__(self,vocabulary_size,token_dimension):
        super(Transformer, self).__init__()
        self.positional_encoding = PositionalEncoding(token_dimension)
        self.encoder = TransformerEncoder(input_dimension=token_dimension)
        self.decoder = TransformerDecoder(input_dimension=token_dimension)
        self.linear = nn.Linear(token_dimension, vocabulary_size)
    
    def forward(self,input_token,output_token):
        encoder_input = self.positional_encoding(input_token.shape[0]) + input_token
        x = self.encoder(encoder_input)
        x = self.decoder(x,output_token)
        x = self.linear(x)
        return torch.softmax(x,1)

class TransformerEncoder(nn.Module):
    def __init__(self,input_dimension=512,n=6):
        super(TransformerEncoder, self).__init__()
        self.multi_head_attention = MultiHeadAttention(key_dimension=int(input_dimension/8),input_dimension=input_dimension,num_heads=8)
        self.feed_forward = nn.Linear(input_dimension, input_dimension)
        self.norm = nn.LayerNorm(input_dimension)
        self.n = n
    
    def forward_sublayers(self,x):
        input = x
        x = self.multi_head_attention(input,input,input)
        add_and_norm = self.norm(input+x) # Skip connection
        x = self.feed_forward(add_and_norm)
        add_and_norm = self.norm(add_and_norm+x) # Skip connection
        return add_and_norm
    
    def forward(self,token):
        x = token
        for i in range(self.n):
            x = self.forward_sublayers(x)
        return x
            


class TransformerDecoder(nn.Module):
    def __init__(self,input_dimension=512,n=6):
        super(TransformerDecoder, self).__init__()
        self.masked_multi_head_attention = MultiHeadAttention(key_dimension=int(input_dimension/8),input_dimension=input_dimension,num_heads=8,masked=True)
        self.multi_head_attention = MultiHeadAttention(key_dimension=int(input_dimension/8),input_dimension=input_dimension,num_heads=8)
        self.feed_forward = nn.Linear(input_dimension, input_dimension)
        self.norm = nn.LayerNorm(input_dimension)
        self.n = n
    
    def forward_sublayers(self,encoder_output,outputs):
        input = outputs
        x = self.masked_multi_head_attention(input,input,input)
        add_and_norm = self.norm(input+x) # Skip connection

        encoder_key = encoder_output
        encoder_query = encoder_output
        x = self.multi_head_attention(encoder_key,encoder_query,add_and_norm)
        add_and_norm = self.norm(add_and_norm+x) # Skip connection

        x = self.feed_forward(add_and_norm)
        add_and_norm = self.norm(add_and_norm+x) # Skip connection
        return add_and_norm
    
    def forward(self,expected_output,encoder_output):
        for i in range(self.n):
            x = self.forward_sublayers(encoder_output,expected_output)
        return x
        

class PositionalEncoding(nn.Module):
    def __init__(self,dimension):
        super(PositionalEncoding, self).__init__()
        self.dimension = dimension
    
    def forward(self, length):
        positional_embeddings = torch.empty(length,self.dimension)

        i = torch.arange(0, self.dimension,dtype=torch.float32)
        odds = i % 2 == 1
        even = i % 2 == 0
        for position in range(length): #TODO: Vectorize
            positional_embeddings[position,even] = torch.sin(position/torch.pow(10000, 2 * i[even] / self.dimension))
            positional_embeddings[position,odds] = torch.cos(position/torch.pow(10000, 2 * i[odds] / self.dimension))
        
        return positional_embeddings

class MultiHeadAttention(nn.Module):
    def __init__(self,key_dimension,input_dimension,num_heads=8,masked=False):
        super(MultiHeadAttention, self).__init__()
        self.key_dimension = key_dimension
        self.num_heads = num_heads
        self.key_linear = [nn.Linear(input_dimension, self.key_dimension) for i in range(self.num_heads)]
        self.query_linear = [nn.Linear(input_dimension, self.key_dimension) for i in range(self.num_heads)]
        self.value_linear = [nn.Linear(input_dimension, self.key_dimension) for i in range(self.num_heads)]
        self.final_linear = nn.Linear(self.key_dimension * self.num_heads, input_dimension)
        self.mask=masked
        
        
    
    def scaled_dot_product_attention(self,query,key,value):
        query_key_attention = torch.matmul(query,torch.transpose(key,0,1))/math.sqrt(self.key_dimension)
        print(query_key_attention.shape)
        if self.mask:
            mask = torch.tril(query_key_attention) ==0
            query_key_attention[mask] = float('-inf')
        softmax_query_key = torch.softmax(query_key_attention,1) # [seq_len,seq_len]
        self_attention = torch.matmul(softmax_query_key,value)
        return self_attention

    def forward(self,query,key,value):
        heads = []
        for i in range(self.num_heads):
            projected_key = self.key_linear[i](key)
            
            projected_query = self.query_linear[i](query)
            projected_value = self.value_linear[i](value)
            head = self.scaled_dot_product_attention(projected_query,projected_key,projected_value)
            heads.append(head)
        concat_heads = torch.cat(heads,1)
        output = self.final_linear(concat_heads)
        return output

transformer = Transformer(1,12)
embeddings = transformer(torch.ones(3,12),torch.ones(3,12)*2)
print(embeddings)
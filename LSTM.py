import torch

# class MemoryBlocks(torch.nn.Module):
#     def __init__(self, output_size,hidden_size,num_cells=1,is_peephole=False):
#         super(MemoryBlocks, self).__init__()
#         # print(f"Initializing memory blocks with {num_blocks} blocks and expecting input of size {output}")
#         self.input_gate = torch.nn.Parameter(torch.randn((output_size+hidden_size,hidden_size)))
#         self.input_bias = torch.nn.Parameter(torch.tensor(output_size,dtype=torch.float32))

#         self.forget_gate = torch.nn.Parameter(torch.randn((output_size+hidden_size,hidden_size)))
#         self.forget_bias = torch.nn.Parameter(torch.tensor(output_size,dtype=torch.float32))

#         self.output_gate = torch.nn.Parameter(torch.randn((output_size+hidden_size,hidden_size)))
#         self.output_bias = torch.nn.Parameter(torch.tensor(output_size,dtype=torch.float32))

#         self.is_peephole = is_peephole
#         if(self.is_peephole):
#             self.peephole_input = torch.nn.Parameter(torch.randn((num_cells,1)))
#             self.peephole_forget = torch.nn.Parameter(torch.randn((num_cells,1)))
#             self.peephole_output = torch.nn.Parameter(torch.randn((num_cells,1)))

#         with torch.no_grad():
#             # Bias weights are initialized with negative values for input and
#             # output gates, positive values for forget gates
#             self.input_bias.data = torch.tensor([-1 for x in range(hidden_size)],dtype=torch.float32)
#             self.forget_bias.data = torch.tensor([1 for x in range(hidden_size)],dtype=torch.float32)
#             self.output_bias.data = torch.tensor([-1 for x in range(hidden_size)],dtype=torch.float32)

#         # Memory cells
#         self.cells_state = torch.zeros(hidden_size,num_cells)
#         self.cells_weights = torch.nn.Parameter(torch.randn((hidden_size,num_cells),dtype=torch.float32))
#         self.activation = torch.nn.Sigmoid()
#         self.squash_g = lambda x : (4 / (1+torch.exp(-x))) - 2 # squashes with range [-2,2]
#         self.squash_h = lambda x : (2 / (1 + torch.exp(-x))) - 1 # squashes with range [-1,1]

#     def forward(self,x):
#         print(f"INPUT TO MEMORY BLOCKS: {x.shape}")
#         if self.is_peephole: 
#             gated_input = self.activation(torch.matmul(x,self.input_gate) + self.input_bias + torch.matmul(self.peephole_input,self.memory_cells[0].state)) 
#             forget_input = self.activation(torch.matmul(x,self.forget_gate) + self.forget_bias + torch.matmul(self.peephole_forget,self.memory_cells[0].state))
#             gated_output = self.activation(torch.matmul(x,self.output_gate) + self.output_bias + torch.matmul(self.peephole_output,self.memory_cells[0].state))
#         else:
#             gated_input = torch.matmul(x,self.input_gate)
#             gated_input = self.activation(gated_input + self.input_bias)
#             gated_output = self.activation(torch.matmul(x,self.output_gate) + self.output_bias)
#             forget_input = self.activation(torch.matmul(x,self.forget_gate) + self.forget_bias)
#         print(f"Gated input is {gated_input.shape}, gated output is {gated_output.shape}, forget input is {forget_input.shape}")

#         # Memory cells
#         print(f"Multiplying {forget_input.shape} by {self.cells_state.shape}")
#         forget_input_state = torch.mul(forget_input,self.cells_state)
#         print(f"Obtained {forget_input_state.shape}")
#         print(f"Multiplying {self.cells_weights.shape} by {x.shape}")
#         input = torch.matmul(self.cells_weights,x)
#         print(f"Multiplying {input.shape} by {gated_input.shape}")
#         gate_input = torch.matmul(input,gated_input)
#         self.cells_state = forget_input_state + self.squash_g(gate_input)
#         cell_output = self.squash_h(self.cells_state) * gated_output
#         return cell_output

class LSTM(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_cells=1,device=torch.device('mps')):
        super(LSTM, self).__init__()
        self.gate_forget_weigths = torch.nn.Parameter(torch.randn(input_size+hidden_size,hidden_size))
        self.forget_bias = torch.nn.Parameter(torch.randn(hidden_size))
        self.gate_input_weights = torch.nn.Parameter(torch.randn(input_size+hidden_size,hidden_size))
        self.input_bias = torch.nn.Parameter(torch.randn(hidden_size))
        self.gate_output_weights = torch.nn.Parameter(torch.randn(input_size+hidden_size,hidden_size))
        self.output_bias = torch.nn.Parameter(torch.randn(hidden_size))
        self.output_weights = torch.nn.Parameter(torch.randn(hidden_size,output_size))
        with torch.no_grad():
            # Bias weights are initialized with negative values for input and
            # output gates, positive values for forget gates
            self.input_bias.data = torch.tensor([-1 for x in range(hidden_size)],dtype=torch.float32)
            self.forget_bias.data = torch.tensor([1 for x in range(hidden_size)],dtype=torch.float32)
            self.output_bias.data = torch.tensor([-1 for x in range(hidden_size)],dtype=torch.float32)

        self.cells_weights = torch.nn.Parameter(torch.randn(input_size+hidden_size,hidden_size))

        self.activation = torch.nn.Sigmoid()
        self.squash_g = lambda x : (4 / (1+torch.exp(-x))) - 2 # squashes with range [-2,2]
        self.squash_h = lambda x : (2 / (1 + torch.exp(-x))) - 1 # squashes with range [-1,1]

        self.hidden_state = torch.zeros((1,hidden_size))
        self.cells_state = torch.zeros((1,hidden_size))
        # self.memory_blocks = MemoryBlocks(output_size=output_size,hidden_size=hidden_size,num_cells=num_cells)

    def forward(self, x):
        input = torch.cat((x,self.hidden_state),dim=1)
        forget_gate = self.activation(torch.matmul(input,self.gate_forget_weigths) + self.forget_bias)
        input_gate = self.activation(torch.matmul(input,self.gate_input_weights) + self.input_bias)
        output_gate = self.activation(torch.matmul(input,self.gate_output_weights) + self.output_bias)
        cell_input = torch.matmul(input,self.cells_weights)
        self.cells_state = forget_gate* self.cells_state + input_gate* self.squash_g(cell_input)
        cell_output = output_gate*self.squash_h(self.cells_state)
        self.hidden_state = self.activation(cell_output)
        output = torch.matmul(self.hidden_state,self.output_weights)
        return output

class RNN(torch.nn.Module):
    def __init__(self,input_size,hidden_size,output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.hidden_state = torch.zeros((1,hidden_size))
        self.activation = torch.nn.Sigmoid()

        self.input_weights = torch.nn.Parameter(torch.randn(input_size+hidden_size,hidden_size))
        self.output_weights = torch.nn.Parameter(torch.randn(hidden_size,output_size))
    
    def forward(self,x):
        input = torch.cat((x,self.hidden_state),dim=1)
        self.hidden_state = self.activation(torch.matmul(input,self.input_weights))
        output = self.activation(torch.matmul(self.hidden_state,self.output_weights))
        return output

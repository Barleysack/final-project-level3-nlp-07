import torch
import torch.nn as nn

class LSTMPcell(nn.Module):
  def __init__(self, input_size, hidden_size,projection_size):
    super(LSTMPcell, self).__init__()
    self.input_size = input_size
    self.hidden_size = hidden_size
    self.linear = nn.Linear(input_size + hidden_size, 4*hidden_size)
    self.layernorm = nn.LayerNorm(hidden_size)
    self.projection= nn.Linear(projection_size, hidden_size,bias=False)

  def forward(self, input, state):
    # input: (batch_size, input_size)
    # state : Tuple of h_prev, c_prev
    # h_prev: (batch_size, hidden_size)
    # c_prev: (batch_size, hidden_size)
    # output: (batch_size, projection_size)
    h_prev, c_prev = state
    combined = torch.cat((input, h_prev), 1)
    gates = self.linear(combined) # (batch_size, 4*hidden_size)
    ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)
    ingate = torch.sigmoid(ingate)
    forgetgate = torch.sigmoid(forgetgate)
    cellgate = torch.tanh(cellgate)
    outgate = torch.sigmoid(outgate)
    cell = forgetgate * c_prev + ingate * cellgate
    cell = self.layernorm(cell)
    output = outgate * torch.tanh(cell)
    output = self.projection(output)
    return output, (output,cell)

class LSTMPLayer(nn.Module):
  def __init__(self, input_size, hidden_size, projection_size):
    super(LSTMPLayer, self).__init__()
    self.input_size = input_size
    self.hidden_size = hidden_size
    self.projection_size = projection_size
    self.lstmp = LSTMPcell(input_size, hidden_size,projection_size)

  def forward(self, input, hidden):
    # input: (batch_size, seq_len, input_size)
    # state : Tuple of h_prev, c_prev
    # output: (batch_size, seq_len, projection_size)
    batch_size = input.size(0)
    output = []
    for i in range(input.size(1)):
      hidden, state = self.lstmp(input[:, i, :], state)
      output.append(hidden)
    output = torch.stack(output, dim=1)
    return output, hidden

class DLSTMP(nn.Module):
    def __init__(
      self,
      input_size,
      hidden_size,
      output_size,
      num_layers,
      batch_first,
      dropout,
      ):
        super(DLSTMP, self).__init__()
        self.input_size=input_size
        self.hidden_size=hidden_size
        self.projection_size=output_size
        self.num_layers=num_layers
        self.batch_first=batch_first
        self.dropout=dropout
        self.layers=nn.ModuleList()
        for i in range(self.num_layers):
            self.layers.append(LSTMPLayer(input_size=input_size,hidden_size=self.hidden_size,projection_size=self.projection_size))
            input_size=self.projection_size
            
    def forward(self, input, state):
        # input: (batch, seq_len, input_size)
        # state: Tuple of h_0, c_0
        # output: (batch, seq_len, projection_size)
        if not self.batch_first:
          input=input.permute(1,0,2)
        for i in range(self.num_layers):
            input,state = self.layers[i](input, state)
            if i==self.num_layers-1:
              break
            state=(torch.zeros(input.size(0),self.hidden_size),torch.zeros(input.size(0),self.hidden_size))
        output=input
        return output, state


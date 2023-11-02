import torch.nn as nn
import torch as th
from model.nn.eprop_gate_l0rd import EpropGateL0rd
from model.nn.eprop_transformer_utils import AlphaAttention, InputEmbeding, OutputEmbeding

class EpropGateL0rdTransformer(nn.Module):
    def __init__(
        self, 
        channels,
        multiplier,
        num_objects,
        batch_size,
        heads, 
        depth,
        reg_lambda,
        dropout=0.0
    ):
        super(EpropGateL0rdTransformer, self).__init__()

        num_inputs  = channels
        num_outputs = channels
        num_hidden  = channels
        num_hidden  = channels * multiplier

        print(f"Predictor channels: {num_hidden}@({num_hidden // heads}x{heads})")

        
        self.depth = depth
        _layers = []
        _layers.append(InputEmbeding(num_inputs, num_hidden))

        for i in range(depth):
            _layers.append(AlphaAttention(num_hidden, num_objects, heads, dropout))
            _layers.append(EpropAlphaGateL0rd(num_hidden, batch_size * num_objects, reg_lambda))

        _layers.append(OutputEmbeding(num_hidden, num_outputs))
        self.layers = nn.Sequential(*_layers)

    def get_openings(self):
        openings = 0
        for i in range(self.depth):
            openings += self.layers[2 * (i + 1)].l0rd.openings.item()

        return openings / self.depth

    def get_hidden(self):
        states = []
        for i in range(self.depth):
            states.append(self.layers[2 * (i + 1)].l0rd.get_hidden())

        return th.cat(states, dim=1)

    def set_hidden(self, hidden):
        states = th.chunk(hidden, self.depth, dim=1)
        for i in range(self.depth):
            self.layers[2 * (i + 1)].l0rd.set_hidden(states[i])

    def forward(self, input: th.Tensor) -> th.Tensor:
        return self.layers(input)
    

class EpropAlphaGateL0rd(nn.Module):
    def __init__(self, num_hidden, batch_size, reg_lambda):
        super(EpropAlphaGateL0rd, self).__init__()
        
        self.alpha = nn.Parameter(th.zeros(1)+1e-12)
        self.l0rd  = EpropGateL0rd(
            num_inputs  = num_hidden, 
            num_hidden  = num_hidden, 
            num_outputs = num_hidden, 
            reg_lambda  = reg_lambda,
            batch_size = batch_size
        )

    def forward(self, input):
        return input + self.alpha * self.l0rd(input)
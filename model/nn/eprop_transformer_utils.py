import torch.nn as nn
import torch as th
from model.utils.nn_utils import LambdaModule
from einops import rearrange, repeat, reduce

class AlphaAttention(nn.Module):
    def __init__(
        self,
        num_hidden,
        num_objects,
        heads,
        dropout = 0.0,
        need_weights = False
    ):
        super(AlphaAttention, self).__init__()

        self.to_sequence = LambdaModule(lambda x: rearrange(x, '(b o) c -> b o c', o = num_objects))
        self.to_batch    = LambdaModule(lambda x: rearrange(x, 'b o c -> (b o) c', o = num_objects))

        self.alpha     = nn.Parameter(th.zeros(1)+1e-12)
        self.attention = nn.MultiheadAttention(
            num_hidden, 
            heads, 
            dropout = dropout, 
            batch_first = True
        )
        self.need_weights = need_weights
        self.att_weights  = None

    def forward(self, x: th.Tensor):
        x = self.to_sequence(x)
        att, self.att_weights = self.attention(x, x, x, need_weights=self.need_weights)
        x = x + self.alpha * att
        return self.to_batch(x)

class InputEmbeding(nn.Module):
    def __init__(self, num_inputs, num_hidden):
        super(InputEmbeding, self).__init__()

        self.embeding = nn.Sequential(
            nn.ReLU(),
            nn.Linear(num_inputs, num_hidden),
            nn.ReLU(),
            nn.Linear(num_hidden, num_hidden),
        )
        self.skip = LambdaModule(
            lambda x: repeat(x, 'b c -> b (n c)', n = num_hidden // num_inputs)
        )
        self.alpha = nn.Parameter(th.zeros(1)+1e-12)

    def forward(self, input: th.Tensor):
        return self.skip(input) + self.alpha * self.embeding(input)

class OutputEmbeding(nn.Module):
    def __init__(self, num_hidden, num_outputs):
        super(OutputEmbeding, self).__init__()

        self.embeding = nn.Sequential(
            nn.ReLU(),
            nn.Linear(num_hidden, num_outputs),
            nn.ReLU(),
            nn.Linear(num_outputs, num_outputs),
        )
        self.skip = LambdaModule(
            lambda x: reduce(x, 'b (n c) -> b c', 'mean', n = num_hidden // num_outputs)
        )
        self.alpha = nn.Parameter(th.zeros(1)+1e-12)

    def forward(self, input: th.Tensor):
        return self.skip(input) + self.alpha * self.embeding(input)
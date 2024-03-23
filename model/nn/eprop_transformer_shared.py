import torch.nn as nn
import torch as th
from model.nn.eprop_gate_l0rd import EpropGateL0rdShared
from model.nn.eprop_transformer_utils import AlphaAttention, InputEmbeding, OutputEmbeding

class EpropGateL0rdTransformerShared(nn.Module):
    def __init__(
        self, 
        channels,
        multiplier,
        num_objects,
        batch_size,
        heads, 
        depth,
        reg_lambda,
        dropout=0.0,
        exchange_length = 48,
    ):
        super(EpropGateL0rdTransformerShared, self).__init__()

        num_inputs  = channels
        num_outputs = channels
        num_hidden = channels * multiplier
        num_hidden_gatelord  = num_hidden + exchange_length
        num_hidden_attention = num_hidden + exchange_length + num_hidden_gatelord 

        self.num_hidden = num_hidden
        self.num_hidden_gatelord = num_hidden_gatelord

        #print(f"Predictor channels: {num_hidden}@({num_hidden // heads}x{heads})")

        self.register_buffer('hidden', th.zeros(batch_size * num_objects, num_hidden_gatelord), persistent=False)
        self.register_buffer('exchange_code', th.zeros(batch_size * num_objects, exchange_length), persistent=False)

        self.depth = depth
        self.input_embeding  = InputEmbeding(num_inputs, num_hidden)
        self.attention       = nn.Sequential(*[AlphaAttention(num_hidden_attention, num_objects, heads, dropout) for _ in range(depth)])
        self.l0rds           = nn.Sequential(*[EpropAlphaGateL0rdShared(num_hidden_gatelord, batch_size * num_objects, reg_lambda) for _ in range(depth)])
        self.output_embeding = OutputEmbeding(num_hidden, num_outputs) 

    def get_openings(self):
        openings = []
        for i in range(self.depth):
            openings.append(self.l0rds[i].l0rd.openings_perslot)

        openings = th.mean(th.stack(openings, dim=0), dim=0)
        return openings

    def get_hidden(self):
        return self.hidden

    def set_hidden(self, hidden):
        self.hidden = hidden

    def detach(self):
        self.hidden = self.hidden.detach()

    def reset_state(self):
        self.hidden = th.zeros_like(self.hidden)

    def forward(self, x: th.Tensor) -> th.Tensor:
        x = self.input_embeding(x)
        exchange_code = self.exchange_code.clone() * 0.0
        x_ex = th.concat((x, exchange_code), dim=1)

        for i in range(self.depth):
            # attention layer
            att = self.attention(th.concat((x_ex, self.hidden), dim=1))
            x_ex = att[:, :self.num_hidden_gatelord]

            # gatelord layer
            x_ex, self.hidden = self.l0rds[i](x_ex, self.hidden)

        # only yield x
        x = x_ex[:, :self.num_hidden]
        return self.output_embeding(x)
    
class EpropAlphaGateL0rdShared(nn.Module):
    def __init__(self, num_hidden, batch_size, reg_lambda):
        super(EpropAlphaGateL0rdShared, self).__init__()
        
        self.alpha = nn.Parameter(th.zeros(1)+1e-12)
        self.l0rd  = EpropGateL0rdShared(
            num_inputs  = num_hidden, 
            num_hidden  = num_hidden, 
            num_outputs = num_hidden, 
            reg_lambda  = reg_lambda,
            batch_size = batch_size
        )

    def forward(self, input, hidden):
        output, hidden = self.l0rd(input, hidden)
        return input + self.alpha * output, hidden
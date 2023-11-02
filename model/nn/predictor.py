import torch.nn as nn
import torch as th
from model.nn.eprop_transformer import EpropGateL0rdTransformer
from model.nn.eprop_transformer_shared import EpropGateL0rdTransformerShared
from model.utils.nn_utils import  LambdaModule, Binarize
from model.nn.residual import ResidualBlock
from einops import rearrange, repeat, reduce

__author__ = "Manuel Traub"
    
class LociPredictor(nn.Module): 
    def __init__(
        self, 
        heads: int, 
        layers: int,
        channels_multiplier: int,
        reg_lambda: float,
        num_objects: int, 
        gestalt_size: int, 
        batch_size: int,
        bottleneck: str,
        transformer_type = 'standard',
    ):
        super(LociPredictor, self).__init__()
        self.num_objects = num_objects
        self.std_alpha   = nn.Parameter(th.zeros(1)+1e-16)
        self.bottleneck_type = bottleneck
        self.gestalt_size = gestalt_size

        self.reg_lambda = reg_lambda
        Transformer = EpropGateL0rdTransformerShared if transformer_type == 'shared' else EpropGateL0rdTransformer
        self.predictor  = Transformer(
            channels    = gestalt_size + 3 + 1 + 2, 
            multiplier  = channels_multiplier,
            heads       = heads, 
            depth       = layers,
            num_objects = num_objects,
            reg_lambda  = reg_lambda, 
            batch_size  = batch_size,
        )

        if bottleneck == 'binar':
            print("Binary bottleneck")
            self.bottleneck = nn.Sequential(
                LambdaModule(lambda x: rearrange(x, 'b c -> b c 1 1')),
                ResidualBlock(gestalt_size, gestalt_size, kernel_size=1),
                Binarize(),
                LambdaModule(lambda x: rearrange(x, '(b o) c 1 1 -> b (o c)', o=num_objects))
            )                 
            
        else:
            print("unrestricted bottleneck")
            self.bottleneck = nn.Sequential(
                LambdaModule(lambda x: rearrange(x, 'b c -> b c 1 1')),
                ResidualBlock(gestalt_size, gestalt_size, kernel_size=1),
                nn.Sigmoid(),
                LambdaModule(lambda x: rearrange(x, '(b o) c 1 1 -> b (o c)', o=num_objects))
            )
                
        self.to_batch  = LambdaModule(lambda x: rearrange(x, 'b (o c) -> (b o) c', o=num_objects))
        self.to_shared = LambdaModule(lambda x: rearrange(x, '(b o) c -> b (o c)', o=num_objects))

    def get_openings(self):
        return self.predictor.get_openings()

    def get_hidden(self):
        return self.predictor.get_hidden()

    def set_hidden(self, hidden):
        self.predictor.set_hidden(hidden)

    def forward(
        self, 
        gestalt: th.Tensor, 
        priority: th.Tensor,
        position: th.Tensor,
        slots_closed: th.Tensor,
    ):

        position        = self.to_batch(position)
        gestalt_cur     = self.to_batch(gestalt)
        priority        = self.to_batch(priority)
        slots_closed    = rearrange(slots_closed, 'b o c -> (b o) c').detach()

        input  = th.cat((gestalt_cur, priority, position, slots_closed), dim=1)
        output = self.predictor(input)

        gestalt  = output[:, :self.gestalt_size]
        priority = output[:,self.gestalt_size:(self.gestalt_size+1)]
        xy       = output[:,(self.gestalt_size+1):(self.gestalt_size+3)]
        std      = output[:,(self.gestalt_size+3):(self.gestalt_size+4)]

        position = th.cat((xy, std * self.std_alpha), dim=1)
        
        position = self.to_shared(position)
        gestalt  = self.bottleneck(gestalt)
        priority = self.to_shared(priority)

        return position, gestalt, priority

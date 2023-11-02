import torch.nn as nn
import torch as th
from model.utils.nn_utils import LambdaModule
from einops import rearrange, repeat, reduce
from model.nn.eprop_gate_l0rd import ReTanh

class PerceptGateController(nn.Module):
    def __init__(
        self,
        num_inputs: int,
        num_hidden: list,
        bias: bool,
        num_objects: int,
        gate_noise_level: float = 0.1,
        reg_lambda: float = 0.000005
    ):
        super(PerceptGateController, self).__init__()

        self.to_batch  = LambdaModule(lambda x: rearrange(x, 'b (o c) -> (b o) c', o=num_objects))
        self.to_shared = LambdaModule(lambda x: rearrange(x, '(b o) c -> b o c', o=num_objects))

        self.layers = nn.Sequential(
            nn.Linear(num_inputs, num_hidden[0], bias = bias),
            nn.Tanh(),
            nn.Linear(num_hidden[0], num_hidden[1], bias = bias),
            nn.Tanh(),
            nn.Linear(num_hidden[1], 2, bias = bias)
        )
        self.output_function = ReTanh(reg_lambda)
        self.register_buffer("noise", th.tensor(gate_noise_level), persistent=False)
        self.init_weights()

    def init_weights(self):
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform(layer.weight)
                layer.bias.data.fill_(3.00)

    def forward(self, position_cur, gestalt_cur, priority_cur, slots_occlusionfactor_cur, position_last, gestalt_last, priority_last, slots_occlusionfactor_last, position_last2, evaluate=False):

        position_cur = self.to_batch(position_cur)
        gestalt_cur  = self.to_batch(gestalt_cur)
        priority_cur = self.to_batch(priority_cur)
        position_last = self.to_batch(position_last)
        gestalt_last  = self.to_batch(gestalt_last)
        priority_last = self.to_batch(priority_last)
        slots_occlusionfactor_cur = self.to_batch(slots_occlusionfactor_cur).detach()
        slots_occlusionfactor_last = self.to_batch(slots_occlusionfactor_last).detach()
        position_last2 = self.to_batch(position_last2).detach()

        input  = th.cat((position_cur, gestalt_cur, priority_cur, slots_occlusionfactor_cur, position_last, gestalt_last, priority_last, slots_occlusionfactor_last, position_last2), dim=1)
        output = self.layers(input)
        if evaluate:
            output = self.output_function(output)
        else:
            noise  = th.normal(mean=0, std=self.noise, size=output.shape, device=output.device)    
            output = self.output_function(output + noise)

        return self.to_shared(output)

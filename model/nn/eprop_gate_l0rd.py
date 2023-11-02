import torch.nn as nn
import torch as th
import numpy as np
from torch.autograd import Function
from einops import rearrange, repeat, reduce

__author__ = "Manuel Traub"

class EpropGateL0rdFunction(Function):
    @staticmethod
    def forward(ctx, x, h_last, w_gx, w_gh, b_g, w_rx, w_rh, b_r, args):

        e_w_gx, e_w_gh, e_b_g, e_w_rx, e_w_rh, e_b_r, reg, noise_level = args

        noise = th.normal(mean=0, std=noise_level, size=b_g.shape, device=b_g.device)
        g     = th.relu(th.tanh(x.mm(w_gx.t()) + h_last.mm(w_gh.t()) + b_g + noise))
        r     = th.tanh(x.mm(w_rx.t()) + h_last.mm(w_rh.t()) + b_r)

        h = g * r + (1 - g) * h_last
        
        # Haevisite step function
        H_g = th.ceil(g).clamp(0, 1)

        dg = (1 - g**2) * H_g
        dr = (1 - r**2)

        delta_h = r - h_last

        g_j  = g.unsqueeze(dim=2)
        dg_j = dg.unsqueeze(dim=2)
        dr_j = dr.unsqueeze(dim=2)

        x_i       = x.unsqueeze(dim=1)
        h_last_i  = h_last.unsqueeze(dim=1)
        delta_h_j = delta_h.unsqueeze(dim=2)

        e_w_gh.copy_(e_w_gh * (1 - g_j) + dg_j * h_last_i * delta_h_j)
        e_w_gx.copy_(e_w_gx * (1 - g_j) + dg_j * x_i      * delta_h_j)
        e_b_g.copy_( e_b_g  * (1 - g)   + dg              * delta_h  )

        e_w_rh.copy_(e_w_rh * (1 - g_j) + dr_j * h_last_i * g_j)
        e_w_rx.copy_(e_w_rx * (1 - g_j) + dr_j * x_i      * g_j)
        e_b_r.copy_( e_b_r  * (1 - g)   + dr              * g  )

        ctx.save_for_backward(
            g.clone(), dg.clone(), dg_j.clone(), dr.clone(), x_i.clone(), h_last_i.clone(), 
            reg.clone(), H_g.clone(), delta_h.clone(), w_gx.clone(), w_gh.clone(), w_rx.clone(), w_rh.clone(),
            e_w_gx.clone(), e_w_gh.clone(), e_b_g.clone(), 
            e_w_rx.clone(), e_w_rh.clone(), e_b_r.clone(), 
        )

        return h, th.mean(H_g) 

    @staticmethod
    def backward(ctx, dh, _):

        g, dg, dg_j, dr, x_i, h_last_i, reg, H_g, delta_h, w_gx, w_gh, w_rx, w_rh, \
        e_w_gx, e_w_gh, e_b_g, e_w_rx, e_w_rh, e_b_r = ctx.saved_tensors

        dh_j      = dh.unsqueeze(dim=2)
        H_g_reg   = reg * H_g
        H_g_reg_j = H_g_reg.unsqueeze(dim=2)

        dw_gx = th.sum(dh_j * e_w_gx + H_g_reg_j * dg_j * x_i,      dim=0)
        dw_gh = th.sum(dh_j * e_w_gh + H_g_reg_j * dg_j * h_last_i, dim=0)
        db_g  = th.sum(dh   * e_b_g  + H_g_reg   * dg,              dim=0)

        dw_rx = th.sum(dh_j * e_w_rx, dim=0)
        dw_rh = th.sum(dh_j * e_w_rh, dim=0)
        db_r  = th.sum(dh   * e_b_r , dim=0)

        dh_dg = (dh * delta_h + H_g_reg) * dg
        dh_dr = dh * g * dr

        dx = dh_dg.mm(w_gx) + dh_dr.mm(w_rx) 
        dh = dh * (1 - g) + dh_dg.mm(w_gh) + dh_dr.mm(w_rh)

        return dx, dh, dw_gx, dw_gh, db_g, dw_rx, dw_rh, db_r, None

class ReTanhFunction(Function):
    @staticmethod
    def forward(ctx, x, reg):

        g = th.relu(th.tanh(x))

        # Haevisite step function
        H_g = th.ceil(g).clamp(0, 1)

        dg = (1 - g**2) * H_g

        ctx.save_for_backward(g, dg, H_g, reg)
        return g, th.mean(H_g) 

    @staticmethod
    def backward(ctx, dh, _):

        g, dg, H_g, reg = ctx.saved_tensors

        dx = (dh + reg * H_g) * dg

        return dx, None

class ReTanh(nn.Module):
    def __init__(self, reg_lambda):
        super(ReTanh, self).__init__()
        
        self.re_tanh = ReTanhFunction().apply
        self.register_buffer("reg_lambda", th.tensor(reg_lambda), persistent=False)

    def forward(self, input):
        h, openings = self.re_tanh(input, self.reg_lambda)
        self.openings = openings.item()

        return h


class EpropGateL0rd(nn.Module):
    def __init__(
        self, 
        num_inputs,
        num_hidden,
        num_outputs,
        batch_size,
        reg_lambda = 0,
        gate_noise_level = 0,
    ):
        super(EpropGateL0rd, self).__init__()

        self.register_buffer("reg", th.tensor(reg_lambda).view(1,1), persistent=False)
        self.register_buffer("noise", th.tensor(gate_noise_level), persistent=False)
        self.num_inputs  = num_inputs
        self.num_hidden  = num_hidden
        self.num_outputs = num_outputs

        self.fcn = EpropGateL0rdFunction().apply
        self.retanh = ReTanh(reg_lambda)

        # gate weights and biases
        self.w_gx = nn.Parameter(th.empty(num_hidden, num_inputs))
        self.w_gh = nn.Parameter(th.empty(num_hidden, num_hidden))
        self.b_g  = nn.Parameter(th.zeros(num_hidden))

        # candidate weights and biases
        self.w_rx = nn.Parameter(th.empty(num_hidden, num_inputs))
        self.w_rh = nn.Parameter(th.empty(num_hidden, num_hidden))
        self.b_r  = nn.Parameter(th.zeros(num_hidden)) 

        # output projection weights and bias
        self.w_px = nn.Parameter(th.empty(num_outputs, num_inputs))
        self.w_ph = nn.Parameter(th.empty(num_outputs, num_hidden))
        self.b_p  = nn.Parameter(th.zeros(num_outputs)) 

        # output gate weights and bias
        self.w_ox = nn.Parameter(th.empty(num_outputs, num_inputs))
        self.w_oh = nn.Parameter(th.empty(num_outputs, num_hidden))
        self.b_o  = nn.Parameter(th.zeros(num_outputs)) 

        # input gate eligibilitiy traces
        self.register_buffer("e_w_gx", th.zeros(batch_size, num_hidden, num_inputs), persistent=False)
        self.register_buffer("e_w_gh", th.zeros(batch_size, num_hidden, num_hidden), persistent=False)
        self.register_buffer("e_b_g",  th.zeros(batch_size, num_hidden),             persistent=False)

        # forget gate eligibilitiy traces
        self.register_buffer("e_w_rx", th.zeros(batch_size, num_hidden, num_inputs), persistent=False)
        self.register_buffer("e_w_rh", th.zeros(batch_size, num_hidden, num_hidden), persistent=False)
        self.register_buffer("e_b_r",  th.zeros(batch_size, num_hidden),             persistent=False)

        # hidden state
        self.register_buffer("h_last", th.zeros(batch_size, num_hidden),             persistent=False)

        self.register_buffer("openings", th.zeros(1), persistent=False)

        # initialize weights
        stdv_ih = np.sqrt(6/(self.num_inputs + self.num_hidden))
        stdv_hh = np.sqrt(3/self.num_hidden)
        stdv_io = np.sqrt(6/(self.num_inputs + self.num_outputs))
        stdv_ho = np.sqrt(6/(self.num_hidden + self.num_outputs))

        nn.init.uniform_(self.w_gx, -stdv_ih, stdv_ih)
        nn.init.uniform_(self.w_gh, -stdv_hh, stdv_hh)

        nn.init.uniform_(self.w_rx, -stdv_ih, stdv_ih)
        nn.init.uniform_(self.w_rh, -stdv_hh, stdv_hh)

        nn.init.uniform_(self.w_px, -stdv_io, stdv_io)
        nn.init.uniform_(self.w_ph, -stdv_ho, stdv_ho)

        nn.init.uniform_(self.w_ox, -stdv_io, stdv_io)
        nn.init.uniform_(self.w_oh, -stdv_ho, stdv_ho)

        self.backprop = False

    def reset_state(self):
        self.h_last.zero_() 
        self.e_w_gx.zero_() 
        self.e_w_gh.zero_() 
        self.e_b_g.zero_() 
        self.e_w_rx.zero_() 
        self.e_w_rh.zero_() 
        self.e_b_r.zero_() 
        self.openings.zero_() 

    def backprop_forward(self, x: th.Tensor):

        noise = th.normal(mean=0, std=self.noise, size=self.b_g.shape, device=self.b_g.device)
        g     = self.retanh(x.mm(self.w_gx.t()) + self.h_last.mm(self.w_gh.t()) + self.b_g + noise)
        r     = th.tanh(x.mm(self.w_rx.t()) + self.h_last.mm(self.w_rh.t()) + self.b_r)

        self.h_last = g * r + (1 - g) * self.h_last
        
        # Haevisite step function
        H_g = th.ceil(g).clamp(0, 1)

        self.openings = th.mean(H_g)

        p = th.tanh(x.mm(self.w_px.t()) + self.h_last.mm(self.w_ph.t()) + self.b_p)
        o = th.sigmoid(x.mm(self.w_ox.t()) + self.h_last.mm(self.w_oh.t()) + self.b_o)
        return o * p
    
    def activate_backprop(self):
        self.backprop = True
         
    def deactivate_backprop(self):
        self.backprop = False

    def detach(self):
        self.h_last.detach_()

    def eprop_forward(self, x: th.Tensor):
        h, openings = self.fcn(
            x, self.h_last,
            self.w_gx, self.w_gh, self.b_g, 
            self.w_rx, self.w_rh, self.b_r, 
            (
                self.e_w_gx, self.e_w_gh, self.e_b_g, 
                self.e_w_rx, self.e_w_rh, self.e_b_r,
                self.reg, self.noise
            )
        )

        self.openings = openings
        self.h_last = h

        p = th.tanh(x.mm(self.w_px.t()) + h.mm(self.w_ph.t()) + self.b_p)
        o = th.sigmoid(x.mm(self.w_ox.t()) + h.mm(self.w_oh.t()) + self.b_o)
        return o * p

    def save_hidden(self):
        self.h_last_saved = self.h_last.detach()

    def restore_hidden(self):
        self.h_last = self.h_last_saved

    def get_hidden(self):
        return self.h_last

    def set_hidden(self, h_last):
        self.h_last = h_last

    def forward(self, x: th.Tensor):
        if self.backprop:
            return self.backprop_forward(x)

        return self.eprop_forward(x)


class EpropGateL0rdShared(EpropGateL0rd):
    def __init__(
        self, 
        num_inputs,
        num_hidden,
        num_outputs,
        batch_size,
        reg_lambda = 0,
        gate_noise_level = 0,
    ):
        super().__init__(num_inputs, num_hidden, num_outputs, batch_size, reg_lambda, gate_noise_level)

    def backprop_forward(self, x: th.Tensor, h_last: th.Tensor):

        noise = th.normal(mean=0, std=self.noise, size=self.b_g.shape, device=self.b_g.device)
        g     = self.retanh(x.mm(self.w_gx.t()) + h_last.mm(self.w_gh.t()) + self.b_g + noise)
        r     = th.tanh(x.mm(self.w_rx.t()) + h_last.mm(self.w_rh.t()) + self.b_r)

        h_last = g * r + (1 - g) * h_last
        
        # Haevisite step function
        H_g = th.ceil(g).clamp(0, 1)

        self.openings = th.mean(H_g)

        p = th.tanh(x.mm(self.w_px.t()) + h_last.mm(self.w_ph.t()) + self.b_p)
        o = th.sigmoid(x.mm(self.w_ox.t()) + h_last.mm(self.w_oh.t()) + self.b_o)
        return o * p, h_last
    
    def eprop_forward(self, x: th.Tensor, h_last: th.Tensor):
        h, openings = self.fcn(
            x, h_last,
            self.w_gx, self.w_gh, self.b_g, 
            self.w_rx, self.w_rh, self.b_r, 
            (
                self.e_w_gx, self.e_w_gh, self.e_b_g, 
                self.e_w_rx, self.e_w_rh, self.e_b_r,
                self.reg, self.noise
            )
        )

        self.openings = openings

        p = th.tanh(x.mm(self.w_px.t()) + h.mm(self.w_ph.t()) + self.b_p)
        o = th.sigmoid(x.mm(self.w_ox.t()) + h.mm(self.w_oh.t()) + self.b_o)
        return o * p, h
    
    def forward(self, x: th.Tensor, h_last: th.Tensor = None):

        if h_last is not None:
            if self.backprop:
                return self.backprop_forward(x, h_last)

            return self.eprop_forward(x, h_last)

        # backward compatibility
        if self.backprop:
            x, h = self.backprop_forward(x, self.h_last)

        x, h = self.eprop_forward(x, self.h_last)

        self.h_last = h
        return x
import torch
import torch.nn as nn
import math

torch.pi = torch.acos(torch.zeros(1)).item() * 2
steps = 4
a = 0.25
Vth = 0.5  #  V_threshold
aa = Vth
tau = 0.25  # exponential decay coefficient
conduct = 0.5 # time-dependent synaptic weight
linear_decay = Vth/(steps * 2)  #linear decay coefficient

gamma_SG = 1.
class SpikeAct_extended(torch.autograd.Function):
    '''
    solving the non-differentiable term of the Heavisde function
    '''
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        # if input = u > Vth then output = 1
        output = torch.gt(input, 0.)
        return output.float()

    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_tensors[0]
        grad_input = grad_output.clone()

        # hu is an approximate func of df/du in linear formulation
        hu = abs(input) < 0.5
        hu = hu.float()

        # arctan surrogate function
        # hu =  1 / ((input * torch.pi) ** 2 + 1)

        # triangles
        # hu = (1 / gamma_SG) * (1 / gamma_SG) * ((gamma_SG - input.abs()).clamp(min=0))

        return grad_input * hu

class ArchAct(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = torch.gt(input, 0.5)
        return output.float()
    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_tensors[0]
        grad_input = grad_output.clone()
        return grad_input

class LIFSpike(nn.Module):
    '''
        Layer-wise gated LIF
    '''
    def __init__(self, **kwargs):
        super(LIFSpike, self).__init__()
        self.T = kwargs['t']
        self.soft_mode = kwargs['soft_mode']
        self.static_gate = kwargs['static_gate']
        self.static_param = kwargs['static_param']
        self.time_wise = kwargs['time_wise']

        self.alpha, self.beta, self.gamma = [nn.Parameter(torch.tensor(math.log(1 / ((i - 0.5)*0.5+0.5) - 1), dtype=torch.float))
                                                 for i in kwargs['gate']]
        if self.static_param:
            self.tau, self.Vth, self.leak, self.conduct = [torch.tensor(- math.log(1 / i - 1), dtype=torch.float)
                                                     for i in kwargs['param']]
            self.reVth = self.Vth
        else:
            if self.time_wise:
                self.tau, self.Vth, self.leak, self.conduct = [nn.Parameter(- math.log(1 / i - 1) * torch.ones(self.T, dtype=torch.float))
                                                               for i in kwargs['param']]
                self.reVth = nn.Parameter(- math.log(1 / kwargs['param'][1] - 1) * torch.ones(self.T, dtype=torch.float))

            else:
                self.tau, self.Vth, self.leak = [nn.Parameter(torch.tensor(- math.log(1 / i - 1), dtype=torch.float))
                                      for i in kwargs['param'][:-1]]
                self.reVth = nn.Parameter(torch.tensor(- math.log(1 / kwargs['param'][1] - 1), dtype=torch.float))

                self.conduct = [nn.Parameter(- math.log(1 / i - 1) * torch.ones(self.T, dtype=torch.float))
                                           for i in kwargs['param'][3:]][0]

    def forward(self, x):
        u = torch.zeros(x.shape[1:], device=x.device)
        out = torch.zeros(x.shape, device=x.device)
        for step in range(self.T):
            u, out[step] = self.extended_state_update(u, out[max(step - 1, 0)], x[step],
                                                      tau=self.tau[step].sigmoid() if self.time_wise else self.tau.sigmoid(),
                                                      Vth=self.Vth[step].sigmoid() if self.time_wise else self.Vth.sigmoid(),
                                                      leak=self.leak[step].sigmoid() if self.time_wise else self.leak.sigmoid(),
                                                      conduct=self.conduct[step].sigmoid() if not self.static_param else self.conduct.sigmoid(),
                                                      reVth=self.reVth[step].sigmoid() if self.time_wise else self.reVth.sigmoid())
        return out
    #
    # def state_update(self, u_t_n1, o_t_n1, W_mul_o_t_n1, tau):
    #     u_t1_n1 = tau * u_t_n1 * (1 - o_t_n1) + W_mul_o_t_n1
    #     o_t1_n1 = spikeAct(u_t1_n1)
    #     return u_t1_n1, o_t1_n1

    def extended_state_update(self, u_t_n1, o_t_n1, W_mul_o_t_n1, tau, Vth, leak, conduct, reVth):
        if self.static_gate:
            al, be, ga = self.alpha.clone().detach().gt(0.).float(), self.beta.clone().detach().gt(0.).float(), self.gamma.clone().detach().gt(0.).float()
        else:
            al, be, ga = self.alpha.sigmoid(), self.beta.sigmoid(), self.gamma.sigmoid()
        # I_t1 = W_mul_o_t_n1 + be * I_t0 * self.conduct.sigmoid()#原先
        I_t1 = W_mul_o_t_n1 * (1 - be * (1 - conduct))
        u_t1_n1 = ((1 - al * (1 - tau)) * u_t_n1 * (1 - ga * o_t_n1.clone()) - (1 - al) * leak) + \
                  I_t1 - \
                  (1 - ga) * reVth * o_t_n1.clone()
        o_t1_n1 = SpikeAct_extended.apply(u_t1_n1 - Vth)
        return u_t1_n1, o_t1_n1

    def _initialize_params(self, **kwargs):
        self.mid_gate_mode = True
        self.tau.copy_(torch.tensor(- math.log(1 / kwargs['param'][0] - 1), dtype=torch.float, device=self.tau.device))
        self.Vth.copy_(torch.tensor(- math.log(1 / kwargs['param'][1] - 1), dtype=torch.float, device=self.Vth.device))
        self.reVth.copy_(torch.tensor(- math.log(1 / kwargs['param'][1] - 1), dtype=torch.float, device=self.reVth.device))

        self.leak.copy_(- math.log(1 / kwargs['param'][2] - 1) * torch.ones(self.T, dtype=torch.float, device=self.leak.device))
        self.conduct.copy_(- math.log(1 / kwargs['param'][3] - 1) * torch.ones(self.T, dtype=torch.float, device=self.conduct.device))


class LIFSpike_CW(nn.Module):
    '''
    gated spiking neuron
    '''
    def __init__(self, inplace, **kwargs):
        super(LIFSpike_CW, self).__init__()
        self.T = kwargs['t']
        self.soft_mode = kwargs['soft_mode']
        self.static_gate = kwargs['static_gate']
        self.static_param = kwargs['static_param']
        self.time_wise = kwargs['time_wise']
        self.plane = inplace
        #c
        self.alpha, self.beta, self.gamma = [nn.Parameter(- math.log(1 / ((i - 0.5)*0.5+0.5) - 1) * torch.ones(self.plane, dtype=torch.float))
                                                 for i in kwargs['gate']]

        self.tau, self.Vth, self.leak = [nn.Parameter(- math.log(1 / i - 1) * torch.ones(self.plane, dtype=torch.float))
                              for i in kwargs['param'][:-1]]
        self.reVth = nn.Parameter(- math.log(1 / kwargs['param'][1] - 1) * torch.ones(self.plane, dtype=torch.float))
        #t, c
        self.conduct = [nn.Parameter(- math.log(1 / i - 1) * torch.ones((self.T, self.plane), dtype=torch.float))
                                   for i in kwargs['param'][3:]][0]

    def forward(self, x): #t, b, c, h, w
        u = torch.zeros(x.shape[1:], device=x.device)
        out = torch.zeros(x.shape, device=x.device)
        for step in range(self.T):
            u, out[step] = self.extended_state_update(u, out[max(step - 1, 0)], x[step],
                                                      tau=self.tau.sigmoid(),
                                                      Vth=self.Vth.sigmoid(),
                                                      leak=self.leak.sigmoid(),
                                                      conduct=self.conduct[step].sigmoid(),
                                                      reVth=self.reVth.sigmoid())
        return out

    #[b, c, h, w]  * [c]
    def extended_state_update(self, u_t_n1, o_t_n1, W_mul_o_t_n1, tau, Vth, leak, conduct, reVth):
        # print(W_mul_o_t_n1.shape, self.alpha[None, :, None, None].sigmoid().shape)
        if self.static_gate:
            if self.soft_mode:
                al, be, ga = self.alpha.view(1, -1, 1, 1).clone().detach().sigmoid(), self.beta.view(1, -1, 1, 1).clone().detach().sigmoid(), self.gamma.view(1, -1, 1, 1).clone().detach().sigmoid()
            else:
                al, be, ga = self.alpha.view(1, -1, 1, 1).clone().detach().gt(0.).float(), self.beta.view(1, -1, 1, 1).clone().detach().gt(0.).float(), self.gamma.view(1, -1, 1, 1).clone().detach().gt(0.).float()
        else:
            if self.soft_mode:
                al, be, ga = self.alpha.view(1, -1, 1, 1).sigmoid(), self.beta.view(1, -1, 1, 1).sigmoid(), self.gamma.view(1, -1, 1, 1).sigmoid()
            else:
                al, be, ga = ArchAct.apply(self.alpha.view(1, -1, 1, 1).sigmoid()), ArchAct.apply(self.beta.view(1, -1, 1, 1).sigmoid()), ArchAct.apply(self.gamma.view(1, -1, 1, 1).sigmoid())

        # I_t1 = W_mul_o_t_n1 + be * I_t0 * self.conduct.sigmoid()#原先
        I_t1 = W_mul_o_t_n1 * (1 - be * (1 - conduct[None, :, None, None]))
        u_t1_n1 = ((1 - al * (1 - tau[None, :, None, None])) * u_t_n1 * (1 - ga * o_t_n1.clone()) - (1 - al) * leak[None, :, None, None]) + \
                  I_t1 - (1 - ga) * reVth[None, :, None, None] * o_t_n1.clone()
        o_t1_n1 = SpikeAct_extended.apply(u_t1_n1 - Vth[None, :, None, None])
        return u_t1_n1, o_t1_n1

    def _initialize_params(self, **kwargs):
        self.mid_gate_mode = True
        self.tau.copy_(torch.tensor(- math.log(1 / kwargs['param'][0] - 1), dtype=torch.float, device=self.tau.device))
        self.Vth.copy_(torch.tensor(- math.log(1 / kwargs['param'][1] - 1), dtype=torch.float, device=self.Vth.device))
        self.reVth.copy_(torch.tensor(- math.log(1 / kwargs['param'][1] - 1), dtype=torch.float, device=self.reVth.device))

        self.leak.copy_(- math.log(1 / kwargs['param'][2] - 1) * torch.ones(self.T, dtype=torch.float, device=self.leak.device))
        self.conduct.copy_(- math.log(1 / kwargs['param'][3] - 1) * torch.ones(self.T, dtype=torch.float, device=self.conduct.device))

class OscLIFSpike_CW(LIFSpike_CW):
    def forward(self, x): #t, b, c, h, w
        u = torch.zeros(x.shape[1:], device=x.device)
        out = torch.zeros(x.shape, device=x.device)
        sign = 1.
        for step in range(self.T):
            u, out[step] = self.extended_state_update(u, out[max(step - 1, 0)], x[step],
                                                      tau=self.tau.sigmoid(),
                                                      Vth=self.Vth.sigmoid(),
                                                      leak=sign * self.leak.sigmoid(),
                                                      conduct=self.conduct[step].sigmoid(),
                                                      reVth=self.reVth.sigmoid())
            sign = - sign
        return out

class OscLIF2Spike_CW(LIFSpike_CW):
    def forward(self, x): #t, b, c, h, w
        u = torch.zeros(x.shape[1:], device=x.device)
        out = torch.zeros(x.shape, device=x.device)
        sign = 1.
        for step in range(self.T):
            u, out[step] = self.extended_state_update(u, out[max(step - 1, 0)], x[step],
                                                      tau=sign * self.tau.sigmoid(),
                                                      Vth=self.Vth.sigmoid(),
                                                      leak=self.leak.sigmoid(),
                                                      conduct=self.conduct[step].sigmoid(),
                                                      reVth=self.reVth.sigmoid())
            sign = - sign
        return out

class OscLIF3Spike_CW(LIFSpike_CW):
    def forward(self, x): #t, b, c, h, w
        u = torch.zeros(x.shape[1:], device=x.device)
        out = torch.zeros(x.shape, device=x.device)
        sign = 1.
        for step in range(self.T):
            u, out[step] = self.extended_state_update(u, out[max(step - 1, 0)], x[step],
                                                      tau=self.tau.sigmoid(),
                                                      Vth=self.Vth.sigmoid(),
                                                      leak=self.leak.sigmoid(),
                                                      conduct=self.conduct[step].sigmoid(),
                                                      reVth=self.reVth.sigmoid(), sign=sign)
            sign = - sign
        return out

    #[b, c, h, w]  * [c]
    def extended_state_update(self, u_t_n1, o_t_n1, W_mul_o_t_n1, tau, Vth, leak, conduct, reVth, sign):
        # print(W_mul_o_t_n1.shape, self.alpha[None, :, None, None].sigmoid().shape)
        if self.static_gate:
            if self.soft_mode:
                al, be, ga = self.alpha.view(1, -1, 1, 1).clone().detach().sigmoid(), self.beta.view(1, -1, 1, 1).clone().detach().sigmoid(), self.gamma.view(1, -1, 1, 1).clone().detach().sigmoid()
            else:
                al, be, ga = self.alpha.view(1, -1, 1, 1).clone().detach().gt(0.).float(), self.beta.view(1, -1, 1, 1).clone().detach().gt(0.).float(), self.gamma.view(1, -1, 1, 1).clone().detach().gt(0.).float()
        else:
            if self.soft_mode:
                al, be, ga = self.alpha.view(1, -1, 1, 1).sigmoid(), self.beta.view(1, -1, 1, 1).sigmoid(), self.gamma.view(1, -1, 1, 1).sigmoid()
            else:
                al, be, ga = ArchAct.apply(self.alpha.view(1, -1, 1, 1).sigmoid()), ArchAct.apply(self.beta.view(1, -1, 1, 1).sigmoid()), ArchAct.apply(self.gamma.view(1, -1, 1, 1).sigmoid())

        # I_t1 = W_mul_o_t_n1 + be * I_t0 * self.conduct.sigmoid()#原先
        I_t1 = W_mul_o_t_n1 * (1 - be * (1 - conduct[None, :, None, None]))
        u_t1_n1 = ((1 - sign * al * (1 - tau[None, :, None, None])) * u_t_n1 * (1 - ga * o_t_n1.clone()) - (1 - al) * leak[None, :, None, None]) + \
                  I_t1 - (1 - ga) * reVth[None, :, None, None] * o_t_n1.clone()
        o_t1_n1 = SpikeAct_extended.apply(u_t1_n1 - Vth[None, :, None, None])
        return u_t1_n1, o_t1_n1

class OscLIF4Spike_CW(LIFSpike_CW):
    def forward(self, x): #t, b, c, h, w
        u = torch.zeros(x.shape[1:], device=x.device)
        out = torch.zeros(x.shape, device=x.device)
        sign = 1.
        for step in range(self.T):
            u, out[step] = self.extended_state_update(u, out[max(step - 1, 0)], x[step],
                                                      tau=self.tau.sigmoid(),
                                                      Vth=self.Vth.sigmoid(),
                                                      leak=self.leak.sigmoid(),
                                                      conduct=self.conduct[step].sigmoid(),
                                                      reVth=self.reVth.sigmoid(), sign=sign)
            sign = - sign
        return out

    #[b, c, h, w]  * [c]
    def extended_state_update(self, u_t_n1, o_t_n1, W_mul_o_t_n1, tau, Vth, leak, conduct, reVth, sign):
        # print(W_mul_o_t_n1.shape, self.alpha[None, :, None, None].sigmoid().shape)
        if self.static_gate:
            if self.soft_mode:
                al, be, ga = self.alpha.view(1, -1, 1, 1).clone().detach().sigmoid(), self.beta.view(1, -1, 1, 1).clone().detach().sigmoid(), self.gamma.view(1, -1, 1, 1).clone().detach().sigmoid()
            else:
                al, be, ga = self.alpha.view(1, -1, 1, 1).clone().detach().gt(0.).float(), self.beta.view(1, -1, 1, 1).clone().detach().gt(0.).float(), self.gamma.view(1, -1, 1, 1).clone().detach().gt(0.).float()
        else:
            if self.soft_mode:
                al, be, ga = self.alpha.view(1, -1, 1, 1).sigmoid(), self.beta.view(1, -1, 1, 1).sigmoid(), self.gamma.view(1, -1, 1, 1).sigmoid()
            else:
                al, be, ga = ArchAct.apply(self.alpha.view(1, -1, 1, 1).sigmoid()), ArchAct.apply(self.beta.view(1, -1, 1, 1).sigmoid()), ArchAct.apply(self.gamma.view(1, -1, 1, 1).sigmoid())

        # I_t1 = W_mul_o_t_n1 + be * I_t0 * self.conduct.sigmoid()#原先
        I_t1 = W_mul_o_t_n1 * (1 - be * (1 - conduct[None, :, None, None]))
        u_t1_n1 = ((1 - al * (1 - tau[None, :, None, None])) * u_t_n1 * (1 - ga * o_t_n1.clone()) - (1 - al) * leak[None, :, None, None]) - (1 - ga) * reVth[None, :, None, None] * o_t_n1.clone()

        if sign == 1.:
            pass
        elif sign == -1.:
            delta_u = u_t1_n1 - u_t_n1
            u_t1_n1 = u_t_n1 - delta_u
            # if sign==1, then u_t1_n1 = u_t_n1 + delta_u = u_t_n1 + u_t1_n1 - u_t_n1 = u_t1_n1
        else:
            raise NotImplementedError(sign)

        u_t1_n1 = u_t1_n1 + I_t1
        o_t1_n1 = SpikeAct_extended.apply(u_t1_n1 - Vth[None, :, None, None])
        return u_t1_n1, o_t1_n1


class OscLIF5Spike_CW(LIFSpike_CW):
    '''
    gated spiking neuron
    '''

    def forward(self, x): #t, b, c, h, w
        u = torch.zeros(x.shape[1:], device=x.device)
        out = torch.zeros(x.shape, device=x.device)
        sign = 1.
        for step in range(self.T):
            u, out[step] = self.extended_state_update(u, out[max(step - 1, 0)], x[step],
                                                      tau=self.tau.sigmoid(),
                                                      Vth=self.Vth.sigmoid(),
                                                      leak=self.leak.sigmoid(),
                                                      conduct=self.conduct[step].sigmoid(),
                                                      reVth=self.reVth.sigmoid(), sign=sign)
            sign = - sign
        return out

    #[b, c, h, w]  * [c]
    def extended_state_update(self, u_t_n1, o_t_n1, W_mul_o_t_n1, tau, Vth, leak, conduct, reVth, sign):
        # print(W_mul_o_t_n1.shape, self.alpha[None, :, None, None].sigmoid().shape)
        if self.static_gate:
            if self.soft_mode:
                al, be, ga = self.alpha.view(1, -1, 1, 1).clone().detach().sigmoid(), self.beta.view(1, -1, 1, 1).clone().detach().sigmoid(), self.gamma.view(1, -1, 1, 1).clone().detach().sigmoid()
            else:
                al, be, ga = self.alpha.view(1, -1, 1, 1).clone().detach().gt(0.).float(), self.beta.view(1, -1, 1, 1).clone().detach().gt(0.).float(), self.gamma.view(1, -1, 1, 1).clone().detach().gt(0.).float()
        else:
            if self.soft_mode:
                al, be, ga = self.alpha.view(1, -1, 1, 1).sigmoid(), self.beta.view(1, -1, 1, 1).sigmoid(), self.gamma.view(1, -1, 1, 1).sigmoid()
            else:
                al, be, ga = ArchAct.apply(self.alpha.view(1, -1, 1, 1).sigmoid()), ArchAct.apply(self.beta.view(1, -1, 1, 1).sigmoid()), ArchAct.apply(self.gamma.view(1, -1, 1, 1).sigmoid())

        # I_t1 = W_mul_o_t_n1 + be * I_t0 * self.conduct.sigmoid()#原先
        I_t1 = W_mul_o_t_n1 * (1 - be * (1 - conduct[None, :, None, None]))
        u_t1_n1 = ((1 - sign * al * (1 - tau[None, :, None, None])) * u_t_n1 * (1 - ga * o_t_n1.clone()) - sign * (1 - al) * leak[None, :, None, None]) + \
                  I_t1 - (1 - ga) * reVth[None, :, None, None] * o_t_n1.clone()
        o_t1_n1 = SpikeAct_extended.apply(u_t1_n1 - Vth[None, :, None, None])
        return u_t1_n1, o_t1_n1

    def _initialize_params(self, **kwargs):
        self.mid_gate_mode = True
        self.tau.copy_(torch.tensor(- math.log(1 / kwargs['param'][0] - 1), dtype=torch.float, device=self.tau.device))
        self.Vth.copy_(torch.tensor(- math.log(1 / kwargs['param'][1] - 1), dtype=torch.float, device=self.Vth.device))
        self.reVth.copy_(torch.tensor(- math.log(1 / kwargs['param'][1] - 1), dtype=torch.float, device=self.reVth.device))

        self.leak.copy_(- math.log(1 / kwargs['param'][2] - 1) * torch.ones(self.T, dtype=torch.float, device=self.leak.device))
        self.conduct.copy_(- math.log(1 / kwargs['param'][3] - 1) * torch.ones(self.T, dtype=torch.float, device=self.conduct.device))


class LIFSpike_CW_softsimple(nn.Module):
    '''
        Coarsely fused LIF, referred to as GLIF_f in 'GLIF: A Unified Gated Leaky Integrate-and-Fire Neuron for Spiking Neural Networks'
    '''
    def __init__(self, inplace, **kwargs):
        super(LIFSpike_CW_softsimple, self).__init__()
        self.T = kwargs['t']
        self.soft_mode = kwargs['soft_mode']
        self.static_gate = kwargs['static_gate']
        self.static_param = kwargs['static_param']
        self.time_wise = kwargs['time_wise']
        self.plane = inplace
        #c
        self.gamma = nn.Parameter(- math.log(1 / ((kwargs['gate'][-1] - 0.5)*0.5+0.5) - 1) * torch.ones(self.plane, dtype=torch.float))

        self.tau, self.Vth, self.leak = [nn.Parameter(- math.log(1 / i - 1) * torch.ones(self.plane, dtype=torch.float))
                              for i in kwargs['param'][:-1]]
        self.reVth = nn.Parameter(- math.log(1 / kwargs['param'][1] - 1) * torch.ones(self.plane, dtype=torch.float))
        #t, c
        self.conduct = [nn.Parameter(- math.log(1 / i - 1) * torch.ones((self.T, self.plane), dtype=torch.float))
                                   for i in kwargs['param'][3:]][0]

    def forward(self, x): #t, b, c, h, w
        u = torch.zeros(x.shape[1:], device=x.device)
        out = torch.zeros(x.shape, device=x.device)
        for step in range(self.T):
            u, out[step] = self.extended_state_update(u, out[max(step - 1, 0)], x[step],
                                                      tau=self.tau.sigmoid(),
                                                      Vth=self.Vth.sigmoid(),
                                                      leak=self.leak.sigmoid(),
                                                      conduct=self.conduct[step].sigmoid(),
                                                      reVth=self.reVth.sigmoid())
        return out

    #[b, c, h, w]  * [c]
    def extended_state_update(self, u_t_n1, o_t_n1, W_mul_o_t_n1, tau, Vth, leak, conduct, reVth):

        I_t1 = W_mul_o_t_n1 * conduct[None, :, None, None]
        u_t1_n1 = ((tau[None, :, None, None]) * u_t_n1 * (1 - o_t_n1.clone()) - leak[None, :, None, None]) + \
                  I_t1 - \
                  reVth[None, :, None, None] * o_t_n1.clone()
        o_t1_n1 = SpikeAct_extended.apply(u_t1_n1 - Vth[None, :, None, None])
        return u_t1_n1, o_t1_n1

    def _initialize_params(self, **kwargs):
        self.mid_gate_mode = True
        self.tau.copy_(torch.tensor(- math.log(1 / kwargs['param'][0] - 1), dtype=torch.float, device=self.tau.device))
        self.Vth.copy_(torch.tensor(- math.log(1 / kwargs['param'][1] - 1), dtype=torch.float, device=self.Vth.device))
        self.reVth.copy_(torch.tensor(- math.log(1 / kwargs['param'][1] - 1), dtype=torch.float, device=self.reVth.device))

        self.leak.copy_(- math.log(1 / kwargs['param'][2] - 1) * torch.ones(self.T, dtype=torch.float, device=self.leak.device))
        self.conduct.copy_(- math.log(1 / kwargs['param'][3] - 1) * torch.ones(self.T, dtype=torch.float, device=self.conduct.device))

    def gumbel_on(self):
        self.static_gate = False

    def gumbel_off(self):
        self.static_gate = True


class tdBatchNorm(nn.BatchNorm2d):
    '''
    Implementation of tdBN in 'Going Deeper With Directly-Trained Larger Spiking Neural Networks '
    '''
    def __init__(self, num_features, eps=1e-05, momentum=0.1, alpha=1, affine=True, track_running_stats=True, Vth=Vth):
        super(tdBatchNorm, self).__init__(
            num_features, eps, momentum, affine, track_running_stats)
        self.alpha = alpha
        self.register_buffer('Vth', torch.tensor(Vth, dtype=torch.float))

    def forward(self, input):
        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        # calculate running estimates
        if self.training:
            mean = input.mean([0, 2, 3])   #T*Batch, channel, height, width
            # use biased var in train
            var = input.var([0, 2, 3], unbiased=False)
            n = input.numel() / input.size(1)
            with torch.no_grad():
                self.running_mean = exponential_average_factor * mean\
                    + (1 - exponential_average_factor) * self.running_mean
                # update running_var with unbiased var
                self.running_var = exponential_average_factor * var * n / (n - 1)\
                    + (1 - exponential_average_factor) * self.running_var
        else:
            mean = self.running_mean
            var = self.running_var

        input = self.alpha * self.Vth * (input - mean[None, :, None, None]) / (torch.sqrt(var[None, :, None, None] + self.eps))
        if self.affine:
            input = input * self.weight[None, :, None, None] + self.bias[None, :, None, None]
        return input


def get_initial_dict(T: int):
    # from https://github.com/Ikarosy/Gated-LIF/blob/60638b28ff8b688b51867e7dd44f1e065d4892b0/train.py#L198
    initial_dict = {'gate': [0.6, 0.8, 0.6], 'param': [tau, Vth, linear_decay, conduct],
                       't': T, 'static_gate': True, 'static_param': False, 'time_wise': True, 'soft_mode': False}
    initial_dict['gate'] = [0.6, 0.8, 0.6]
    initial_dict['static_gate'] = False
    initial_dict['static_param'] = False
    initial_dict['time_wise'] = False
    initial_dict['soft_mode'] = False
    return initial_dict



# class Dropout(nn.Module):
#     def __init__(self, p=0.5):
#         super(Dropout, self).__init__()
#         assert 0 <= p < 1.
#         self.p = p
#
#     def forward(self, x): # T, B, C, H, W
#         if self.training:
#             mask = F.dropout(torch.ones(x.shape[1:], device=x.device), self.p, training=True)
#             out = torch.zeros(x.shape, device=x.device)
#             for step in range(x.shape[0]):
#                 out[step] = x[step] * mask
#             return out
#         else:
#             return x


if __name__ == "__main__":
    channels = 2
    T = 4
    print(list(LIFSpike_CW(channels, **get_initial_dict(T)).parameters()))
    exit()
    test_data = torch.rand(1, 1, 3, 3)
    test_data, _ = torch.broadcast_tensors(test_data, torch.zeros((2,) + test_data.shape))
    # test_data = test_data.permute(1, 2, 3, 4, 0)
    print(test_data)

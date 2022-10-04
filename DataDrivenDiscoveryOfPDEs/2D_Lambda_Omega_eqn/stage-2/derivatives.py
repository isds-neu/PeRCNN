import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import scipy.io as scio

torch.set_default_dtype(torch.float32)

dx_2d_op  = [[[[    0,   0,  1/12,   0,    0],
               [    0,   0, -8/12,   0,    0],
               [    0,   0,     0,   0,    0],
               [    0,   0,  8/12,   0,    0],
               [    0,   0, -1/12,   0,    0]]]]

dy_2d_op  = [[[[    0,     0,    0,     0,      0],
               [    0,     0,    0,     0,      0],
               [ 1/12, -8/12,    0,  8/12,  -1/12],
               [    0,     0,    0,     0,      0],
               [    0,     0,    0,     0,      0]]]]

lap_2d_op = [[[[    0,   0, -1/12,   0,     0],
               [    0,   0,   4/3,   0,     0],
               [-1/12, 4/3,   - 5, 4/3, -1/12],
               [    0,   0,   4/3,   0,     0],
               [    0,   0, -1/12,   0,     0]]]]

"""
    dx = 0.2
    dt = 0.0125  
"""


class Conv1dDerivative(nn.Module):
    def __init__(self, DerFilter, resol, kernel_size=3, name=''):
        super(Conv1dDerivative, self).__init__()

        self.resol = resol  # $\delta$*constant in the finite difference
        self.name = name
        self.input_channels = 1
        self.output_channels = 1
        self.kernel_size = kernel_size

        self.padding = int((kernel_size - 1) / 2)
        self.filter = nn.Conv1d(self.input_channels, self.output_channels, self.kernel_size,
            1, padding=0, bias=False)

        # Fixed gradient operator
        self.filter.weight = nn.Parameter(torch.tensor(DerFilter, dtype=torch.float32), requires_grad=False)

    def forward(self, input):
        derivative = self.filter(input)
        return derivative / self.resol


class Conv2dDerivative(nn.Module):
    def __init__(self, DerFilter, resol, kernel_size=5, name=''):
        '''
        :param DerFilter: constructed derivative filter, e.g. Laplace filter
        :param resol: resolution of the filter, used to divide the output, e.g. c*dt, c*dx or c*dx^2
        :param kernel_size:
        :param name: optional name for the operator
        '''
        super(Conv2dDerivative, self).__init__()
        self.resol = resol  # constant in the finite difference
        self.name = name
        self.input_channels = 1
        self.output_channels = 1
        self.kernel_size = kernel_size

        self.padding = int((kernel_size - 1) / 2)
        self.filter = nn.Conv2d(self.input_channels, self.output_channels, self.kernel_size, 
                                1, padding=0, bias=False)
        # Fixed gradient operator
        self.filter.weight = nn.Parameter(torch.tensor(DerFilter, dtype=torch.float32), requires_grad=False)

    def forward(self, input):
        derivative = self.filter(input)
        return derivative / self.resol



class Loss_generator(nn.Module):
    ''' Loss generator for physics loss '''

    def __init__(self, dt = (0.0125), dx = (0.2)):

        '''
        Construct the derivatives, X = Width, Y = Height      

        '''

        self.dt = dt
        self.dx = dx
        self.dy = dx
       
        super(Loss_generator, self).__init__()

        # spatial derivative operator
        self.laplace = Conv2dDerivative(
            DerFilter = lap_2d_op,
            resol = (self.dx**2),
            kernel_size = 5,
            name = 'laplace_operator').cuda()

        # temporal derivative operator
        self.dt = Conv1dDerivative(
            DerFilter = [[[-1, 1, 0]]],
            resol = (self.dt*1),
            kernel_size = 3,
            name = 'partial_t').cuda()

        # Spatial derivative operator
        self.dx = Conv2dDerivative(
            DerFilter = dx_2d_op,
            resol = (self.dx),
            kernel_size = 5,
            name = 'dx_operator').cuda()

        # Spatial derivative operator
        self.dy = Conv2dDerivative(
            DerFilter = dy_2d_op,
            resol = (self.dy),
            kernel_size = 5,
            name = 'dy_operator').cuda()


    def get_library(self, output):
        '''
        Calculate the physics loss

        Args:
        -----
        output: tensor, dim:
            shape: [time, channel, height, width]

        Returns:
        --------
        f_u: float
            physics loss of u

        f_v: float
            physics loss of v
        '''

        start = 0
        end = -2

        # spatial derivatives
        laplace_u = self.laplace(output[start:end, 0:1, :, :])  # 201x1x128x128
        laplace_v = self.laplace(output[start:end, 1:2, :, :])  # 201x1x128x128

        u_x = self.dx(output[start:end, 0:1, :, :])  # 201x1x128x128
        u_y = self.dy(output[start:end, 0:1, :, :])  # 201x1x128x128

        v_x = self.dx(output[start:end, 1:2, :, :])  # 201x1x128x128
        v_y = self.dy(output[start:end, 1:2, :, :])  # 201x1x128x128

        # temporal derivatives - u
        u = output[:, 0:1, 2:-2, 2:-2]
        lent = u.shape[0]
        lenx = u.shape[3]
        leny = u.shape[2]
        u_conv1d = u.permute(2, 3, 1, 0)  # [height(Y), width(X), c, step]
        u_conv1d = u_conv1d.reshape(lenx*leny, 1, lent)
        u_t = self.dt(u_conv1d)  # lent-2 due to no-padding
        u_t = u_t.reshape(leny, lenx, 1, lent-2)
        u_t = u_t.permute(3, 2, 0, 1)  # [step-2, c, height(Y), width(X)]

        # temporal derivatives - v
        v = output[:, 1:2, 2:-2, 2:-2]
        v_conv1d = v.permute(2, 3, 1, 0)  # [height(Y), width(X), c, step]
        v_conv1d = v_conv1d.reshape(lenx*leny, 1, lent)
        v_t = self.dt(v_conv1d)  # lent-2 due to no-padding
        v_t = v_t.reshape(leny, lenx, 1, lent-2)
        v_t = v_t.permute(3, 2, 0, 1)  # [step-2, c, height(Y), width(X)]

        u = output[start:end, 0:1, 2:-2, 2:-2]  # [step, c, height(Y), width(X)]
        v = output[start:end, 1:2, 2:-2, 2:-2]  # [step, c, height(Y), width(X)]

        # make sure the dimensions consistent
        assert laplace_u.shape == u_t.shape
        assert u_t.shape == v_t.shape
        assert laplace_u.shape == u.shape
        assert laplace_v.shape == v.shape

        # lambda-omega RD eqn
        beta = 1.0
        mu_u, mu_v = 0.1, 0.1

        f_u = u_t - (mu_u*laplace_u + (1 - u**2 - v**2)*u + beta*(u**2+v**2)*v)
        f_v = v_t - (mu_v*laplace_v + (1 - u**2 - v**2)*v - beta*(u**2+v**2)*u)

        ones = torch.ones_like(u)

        library = {'f_u': f_u, 'f_v': f_v, 'ones': ones, 'u': u, 'v': v, 'u_t': u_t, 'v_t': v_t,
                   'u_x': u_x, 'u_y': u_y, 'v_x': v_x, 'v_y': v_y, 'lap_u': laplace_u, 'lap_v': laplace_v}

        return library


    def get_residual_mse(self, output):
        '''calculate the phycis loss'''

        mse_loss = nn.MSELoss()

        output = torch.cat((output[:, :, :, -2:], output, output[:, :, :, 0:3]), dim=3)
        output = torch.cat((output[:, :, -2:, :], output, output[:, :, 0:3, :]), dim=2)

        library = self.get_library(output)

        f_u, f_v = library['f_u'], library['f_v']

        mse_u = mse_loss(f_u, torch.zeros_like(f_u).cuda())
        mse_v = mse_loss(f_v, torch.zeros_like(f_v).cuda())

        return mse_u, mse_v


def add_noise(truth, pec=0.05):  # BxCx101x101
    from torch.distributions import normal
    assert truth.shape[1]==2
    uv = [truth[:,0:1,:,:], truth[:,1:2,:,:]]
    uv_noi = []
    for truth in uv:
        n_distr = normal.Normal(0.0, 1.0)
        R = n_distr.sample(truth.shape)
        std_R = torch.std(R)          # std of samples
        std_T = torch.std(truth)
        noise = R*std_T/std_R*pec
        uv_noi.append(truth+noise)
    return torch.cat(uv_noi, dim=1)


def to_numpy_float64(library):
    for key in library.keys():
        library[key] = library[key].detach().cpu().double().numpy().flatten()[:, None]
    return library


if __name__ == '__main__':

    ################# prepare the input dataset ####################
    time_steps = 400
    dt = 0.0125
    dx = 0.2
    dy = 0.2

    ################### define the Initial conditions ####################
    import scipy.io as sio
    UV = sio.loadmat('../data/uv_2x1602x100x100_Euler_[dt=0.0125,HighOrderLap].mat')['uv']
    UV = np.swapaxes(UV, 0, 1)
    truth_clean = torch.from_numpy(UV[100:1601]).float()  # [1801, 2, 100 X, 100 Y]
    # Add noise 10%
    UV = add_noise(truth_clean, pec=0.0)

    # get the ground truth (noisy) for training
    truth = UV[:time_steps+1]                                # [401, 2, 100, 100]

    # output.shape = Tx2x50x50
    output = UV.cuda()

    loss_generator = Loss_generator()
    mse_u, mse_v = loss_generator.get_residual_mse(output)






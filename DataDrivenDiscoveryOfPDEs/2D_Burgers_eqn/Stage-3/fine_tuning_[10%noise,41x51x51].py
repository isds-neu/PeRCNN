import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import StepLR
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import scipy.io as sio
import time
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
torch.set_default_dtype(torch.float64)

torch.manual_seed(66)
np.random.seed(66)

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
               [-1/12, 4/3,    -5, 4/3, -1/12],
               [    0,   0,   4/3,   0,     0],
               [    0,   0, -1/12,   0,     0]]]]

class upscaler(nn.Module):
    ''' Upscaler to convert low-res to high-res initial state '''

    def __init__(self):
        super(upscaler, self).__init__()
        self.layers = []
        self.up0 = nn.ConvTranspose2d(2, 16, kernel_size=5, padding=5 // 2, stride=2, output_padding=1, bias=True)
        # self.layers.append(torch.nn.ReLU())
        self.tanh = torch.nn.Tanh()
        # self.up1 = nn.ConvTranspose2d(16, 16, kernel_size=5, padding=5 // 2, stride=2, output_padding=1, bias=True)
        self.out = nn.Conv2d(16, 2, 1, 1, padding=0, bias=True)
        self.convnet = torch.nn.Sequential(self.up0, self.tanh, self.out)

    def forward(self, h):
        return self.convnet(h)

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
        self.input_padding = kernel_size//2

        self.padding = int((kernel_size - 1) / 2)
        self.filter = nn.Conv2d(self.input_channels, self.output_channels, self.kernel_size,
                                1, padding=self.input_padding, padding_mode='circular', bias=False)
        # Fixed gradient operator
        self.filter.weight.data = torch.tensor(DerFilter, dtype=torch.float64)
        self.filter.weight.requires_grad = False


    def forward(self, input):
        derivative = self.filter(input)
        return derivative / self.resol


class RCNNCell(nn.Module):
    ''' Recurrent convolutional neural network Cell '''
   
    def __init__(self, input_channels, hidden_channels, output_channels, input_kernel_size,
                 input_stride, input_padding):
        
        '''       
        input dimension -> spatial dimension -> size changable
        hidden dimension -> temporal dimension -> size fixed

        Args:
        -----------
        input_channels: int
            Number of channels of input tensor

        hidden_channels: int
            Number of channels of hidden state 

        input_kernel_size: int
            Size of the convolutional kernel for input tensor

        input_stride: int
            Convolution stride, only for input
            b/c we need to keep the hidden state have same dimension

        input_padding: int
            Convolution padding, only for input
            b/c we need to keep the hidden state have same dimension
        '''

        super(RCNNCell, self).__init__()

        # the initial parameters
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.output_channels = output_channels
        self.input_kernel_size = 5
        self.input_stride = input_stride
        self.input_padding = self.input_kernel_size//2

        self.nu_u = torch.nn.Parameter(torch.tensor(0.005194, dtype=torch.float64), requires_grad=True)   # lap_u
        self.nu_v = torch.nn.Parameter(torch.tensor(0.005310, dtype=torch.float64), requires_grad=True)   # lap_v

        self.C1_u = torch.nn.Parameter(torch.tensor(-0.99013, dtype=torch.float64), requires_grad=True)  # uu_x
        self.C2_u = torch.nn.Parameter(torch.tensor(-0.99245, dtype=torch.float64), requires_grad=True)  # vu_y

        self.C1_v = torch.nn.Parameter(torch.tensor(-0.97758, dtype=torch.float64), requires_grad=True)  # uv_x
        self.C2_v = torch.nn.Parameter(torch.tensor(-0.97639, dtype=torch.float64), requires_grad=True)  # vv_y

        self.dx = 1/100
        self.dy = 1/100
        self.dt = 0.00025

        # laplace operator
        self.laplace_op = Conv2dDerivative( DerFilter = lap_2d_op,
                                            resol = (self.dx**2),
                                            kernel_size = 5,
                                            name = 'laplace_operator').cuda()

        # dx operator
        self.dx_op = Conv2dDerivative(  DerFilter = dx_2d_op,
                                        resol = (self.dx),
                                        kernel_size = 5,
                                        name = 'dx_operator').cuda()

        # dy operator
        self.dy_op = Conv2dDerivative( DerFilter = dy_2d_op,
                                       resol = (self.dy),
                                       kernel_size = 5,
                                       name = 'dy_operator').cuda()

    def f_rhs(self, u, v):
        f_u = self.nu_u*self.laplace_op(u) + self.C1_u*u*self.dx_op(u) + self.C2_u*v*self.dy_op(u)
        f_v = self.nu_v*self.laplace_op(v) + self.C1_v*u*self.dx_op(v) + self.C2_v*v*self.dy_op(v)
        return f_u, f_v

    def forward_rk4(self, h):
        '''
        Calculate the updated gates forward.

        Args:
        -----------
        x: tensor
            Input, shape:[batch, channel, height, width]

        h: tensor
            Previous hidden state, shape: [batch, channel, height, width]

        Returns:
        --------
        ch: tensor
            updated hidden state, shape: [batch, channel, height, width]

        '''
        u0 = h[:, 0:1, ...]
        v0 = h[:, 1:2, ...]

        # Stage 1
        k1_u, k1_v = self.f_rhs(u0, v0)

        u1 = u0 + k1_u * self.dt / 2.0
        v1 = v0 + k1_v * self.dt / 2.0

        # Stage 2
        k2_u, k2_v = self.f_rhs(u1, v1)

        u2 = u0 + k2_u * self.dt / 2.0
        v2 = v0 + k2_v * self.dt / 2.0

        # Stage 3
        k3_u, k3_v = self.f_rhs(u2, v2)

        u3 = u0 + k3_u * self.dt
        v3 = v0 + k3_v * self.dt

        # Final stage
        k4_u, k4_v = self.f_rhs(u3, v3)

        u_next = u0 + self.dt * (k1_u + 2 * k2_u + 2 * k3_u + k4_u) / 6.0
        v_next = v0 + self.dt * (k1_v + 2 * k2_v + 2 * k3_v + k4_v) / 6.0

        ch = torch.cat((u_next, v_next), dim=1)

        return ch, ch


    def forward(self, h):

        u0 = h[:, 0:1, ...]
        v0 = h[:, 1:2, ...]

        f_u, f_v = self.f_rhs(u0, v0)

        u_next = u0 + self.dt * f_u
        v_next = v0 + self.dt * f_v

        ch = torch.cat((u_next, v_next), dim=1)

        return ch, ch


    def init_hidden_tensor(self, prev_state):
        ''' 
            Initial hidden state with h from previous batch
            shape: [batch, channel, height, width]
        '''
        return prev_state.cuda()

    def show_coef(self):
        from prettytable import PrettyTable
        table = PrettyTable()
        table.field_names = ['\\', r"$\nu_u$", r"$\nu_v$", r"$Cu_1$", r"$Cu_2$", r"$Cv_1$", r"$Cv_2$", ]
        nu_u, nu_v, Cu_1, Cu_2, Cv_1, Cv_2 \
           = self.nu_u.item(), self.nu_v.item(), self.C1_u.item(), self.C2_u.item(), \
             self.C1_v.item(), self.C2_v.item()
        table.add_row(["True",       0.005, 0.005, -1, -1, -1, -1])
        table.add_row(["Identified", nu_u, nu_v, Cu_1, Cu_2, Cv_1, Cv_2, ])
        print(table)
        # return mu_u, mu_v, C_1, C_2, C_F, C_k

class RCNN(nn.Module):

    ''' Recurrent convolutional neural network layer '''

    def __init__(self, input_channels, hidden_channels, output_channels, init_state_low, input_kernel_size,
        input_stride, input_padding, step=1, effective_step=[1]):
        
        '''        
        Args:
        -----------
        input_channels: int
            Number of channels of input tensor

        hidden_channels: int
            Number of channels of hidden state 

        input_kernel_size: int
            Size of the convolutional kernel for input 

        input_stride: int
            Convolution stride, only for input
            b/c we need to keep the hidden state have same dimension

        input_padding: input
            Convolution padding, only for input
            b/c we need to keep the hidden state have same dimension
        
        step: int
            time steps

        effective_step: list
        '''

        super(RCNN, self).__init__()
        
        # input channels of layer includes input_channels and hidden_channels of cells 
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.output_channels = output_channels
        self.input_kernel_size = input_kernel_size
        self.input_stride = input_stride
        self.input_padding = input_padding
        self.step = step
        self.effective_step = effective_step
        # self.init_state = torch.nn.Parameter(torch.tensor(ini_state, dtype=torch.float64).cuda(), requires_grad=True)
        self._all_layers = []
        self.init_state_low = init_state_low
        self.init_state = []

        self.UpconvBlock = upscaler().cuda()

        name = 'crnn_cell'
        cell = RCNNCell(
            input_channels = self.input_channels,
            hidden_channels = self.hidden_channels,
            output_channels = self.output_channels,
            input_kernel_size = self.input_kernel_size,
            input_stride = self.input_stride,
            input_padding = self.input_padding)

        setattr(self, name, cell)
        self._all_layers.append(cell)


    def forward(self):
        '''
        RCNN temporal propogation 
        "internal" is horizontal transmission (h), only has the length of num_layers
        
        Args:
        -----
        input: tensor, shape: [time, batch, channel, height, width]
            input tensor - X
        
        init_state_low: tensor, shape: [batch, channel, height, width]
            initial state
        
        Returns:
        --------
        outputs: list
            output results list, vertical output (h)
        
        second_last_state: list, length = # layers
            the second last state  
        
        '''

        self.init_state = self.UpconvBlock(self.init_state_low)
        internal_state = []
        outputs = [self.init_state]
        second_last_state = []

        for step in range(self.step):
            name = 'crnn_cell'
            # all cells are initialized in the first step
            if step == 0:
                h = self.init_state
                internal_state = h

            # forward
            h = internal_state
            # hidden state + output
            h, o = getattr(self, name)(h)
            internal_state = h

            if step == (self.step - 2):
                #  last output is a dummy for central FD
                second_last_state = internal_state.clone()

            # after many layers output the result save at time step t
            if step in self.effective_step:
                outputs.append(o)

        return outputs, second_last_state


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
        self.filter.weight = nn.Parameter(torch.tensor(DerFilter, dtype=torch.float64), requires_grad=False)

    def forward(self, input):
        derivative = self.filter(input)
        return derivative / self.resol


class loss_generator(nn.Module):
    ''' Loss generator for physics loss '''

    def __init__(self, dt=(0.00025), dx=(1.0 / 100)):
        '''
        Construct the derivatives, X = Width, Y = Height

        '''

        self.dt = dt
        self.dx = dx

        super(loss_generator, self).__init__()

        # spatial derivative operator
        self.laplace = Conv2dDerivative(
            DerFilter=lap_2d_op,
            resol=(dx ** 2),
            kernel_size=5,
            name='laplace_operator').cuda()

        # temporal derivative operator
        self.dt = Conv1dDerivative(
            DerFilter=[[[-1, 1, 0]]],
            resol=(dt * 1),
            kernel_size=3,
            name='partial_t').cuda()

        # Spatial derivative operator
        self.dx = Conv2dDerivative(
            DerFilter=dx_2d_op,
            resol=(dx),
            kernel_size=5,
            name='dx_operator').cuda()

        # Spatial derivative operator
        self.dy = Conv2dDerivative(
            DerFilter=dy_2d_op,
            resol=(dy),
            kernel_size=5,
            name='dy_operator').cuda()

    def get_phy_Loss(self, output):  # Staggered version
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

        # spatial derivatives
        laplace_u = self.laplace(output[0:-2, 0:1, :, :])  # 201x1x128x128
        laplace_v = self.laplace(output[0:-2, 1:2, :, :])  # 201x1x128x128

        u_x = self.dx(output[0:-2, 0:1, :, :])  # 201x1x128x128
        u_y = self.dy(output[0:-2, 0:1, :, :])  # 201x1x128x128

        v_x = self.dx(output[0:-2, 1:2, :, :])  # 201x1x128x128
        v_y = self.dy(output[0:-2, 1:2, :, :])  # 201x1x128x128

        # temporal derivatives - u
        u = output[:, 0:1, :, :]
        lent = u.shape[0]
        lenx = u.shape[3]
        leny = u.shape[2]
        u_conv1d = u.permute(2, 3, 1, 0)  # [height(Y), width(X), c, step]
        u_conv1d = u_conv1d.reshape(lenx * leny, 1, lent)
        u_t = self.dt(u_conv1d)  # lent-2 due to no-padding
        u_t = u_t.reshape(leny, lenx, 1, lent - 2)
        u_t = u_t.permute(3, 2, 0, 1)  # [step-2, c, height(Y), width(X)]

        # temporal derivatives - v
        v = output[:, 1:2, :, :]
        v_conv1d = v.permute(2, 3, 1, 0)  # [height(Y), width(X), c, step]
        v_conv1d = v_conv1d.reshape(lenx * leny, 1, lent)
        v_t = self.dt(v_conv1d)  # lent-2 due to no-padding
        v_t = v_t.reshape(leny, lenx, 1, lent - 2)
        v_t = v_t.permute(3, 2, 0, 1)  # [step-2, c, height(Y), width(X)]

        u = output[0:-2, 0:1, :, :]  # [step, c, height(Y), width(X)]
        v = output[0:-2, 1:2, :, :]  # [step, c, height(Y), width(X)]

        # make sure the dimensions consistent
        assert laplace_u.shape == u_t.shape
        assert u_t.shape == v_t.shape
        assert laplace_u.shape == u.shape
        assert laplace_v.shape == v.shape

        # Burger's eqn
        nu = 1 / 200

        f_u = (u_t - nu * laplace_u + u * u_x + v * u_y) / 1
        f_v = (v_t - nu * laplace_v + u * v_x + v * v_y) / 1

        return f_u, f_v

def get_ic_loss(model):
    Upconv = model.UpconvBlock
    init_state_low = model.init_state_low
    init_state_low = torch.cat((init_state_low, init_state_low[:, :, :, 0:1]), dim=3)
    init_state_low = torch.cat((init_state_low, init_state_low[:, :, 0:1, :]), dim=2)
    init_state_bicubic = F.interpolate(init_state_low, (101, 101), mode='bicubic', align_corners=True)
    mse_loss = nn.MSELoss()
    init_state_pred = Upconv(model.init_state_low)
    loss_ic = mse_loss(init_state_pred, init_state_bicubic[:, :, :-1, :-1])
    return loss_ic



def loss_gen(output, loss_func):
    '''calculate the phycis loss'''
    
    # Padding x axis due to periodic boundary condition
    # shape: [27, 2, 128, 131]

    # get physics loss
    mse_loss = nn.MSELoss()
    f_u, f_v = loss_func.get_phy_Loss(output)
    loss = mse_loss(f_u, torch.zeros_like(f_u).cuda()) + mse_loss(
        f_v, torch.zeros_like(f_v).cuda())
    return loss

def pretrain_upscaler(Upconv, init_state_low, epoch=4000, plotFlag=True):
    '''
    :param Upconv: upscalar model
    :param init_state_low: low resolution measurement
    :return:
    '''
    init_state_51 = torch.cat((init_state_low, init_state_low[:, :, :, 0:1]), dim=3)
    init_state_51 = torch.cat((init_state_51, init_state_51[:, :, 0:1, :]), dim=2)
    init_state_bicubic = F.interpolate(init_state_51, (101, 101), mode='bicubic', align_corners=True)
    optimizer = optim.Adam(Upconv.parameters(), lr = 0.02)
    scheduler = StepLR(optimizer, step_size=100, gamma=0.97)
    mse_loss = nn.MSELoss()
    for epoch in range(epoch):
        optimizer.zero_grad()
        init_state_pred = Upconv(init_state_low)
        loss = mse_loss(init_state_pred, init_state_bicubic[:, :, :-1, :-1])
        loss.backward(retain_graph=True)
        print('[%d] loss: %.9f' % ((epoch+1), loss.item()))
        optimizer.step()
        scheduler.step()
    if plotFlag:
        init_state_upconv = Upconv(init_state_low)
        x = np.linspace(0, 1, 100)
        y = np.linspace(0, 1, 100)
        x_star, y_star = np.meshgrid(x, y)
        u_bicubic = init_state_bicubic[0, 0, :-1, :-1].detach().cpu().numpy()
        v_bicubic = init_state_bicubic[0, 1, :-1, :-1].detach().cpu().numpy()
        u_upconv  = init_state_upconv[0, 0, :, :].detach().cpu().numpy()
        v_upconv  = init_state_upconv[0, 1, :, :].detach().cpu().numpy()
        u_low_res = np.kron(init_state_low[0, 0, ...].detach().cpu().numpy(), np.ones((2, 2)))
        v_low_res = np.kron(init_state_low[0, 1, ...].detach().cpu().numpy(), np.ones((2, 2)))
        fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(10, 6))
        fig.subplots_adjust(hspace=0.2, wspace=0.2)
        #
        cf = ax[0, 0].scatter(x_star, y_star, c=u_bicubic-u_upconv, alpha=0.9, edgecolors='none', cmap='RdBu', marker='s', s=4, vmin=-.5, vmax=.5)
        ax[0, 0].axis('square')
        ax[0, 0].set_xlim([0, 1])
        ax[0, 0].set_ylim([0, 1])
        ax[0, 0].set_xticks([])
        ax[0, 0].set_yticks([])
        ax[0, 0].set_title('u-u Bicubic')
        fig.colorbar(cf, ax=ax[0, 0], fraction=0.046, pad=0.04)
        #
        cf = ax[0, 1].scatter(x_star, y_star, c=u_upconv, alpha=0.9, edgecolors='none', cmap='RdYlBu', marker='s', s=4, vmin=-2, vmax=1.6)
        ax[0, 1].axis('square')
        ax[0, 1].set_xlim([0, 1])
        ax[0, 1].set_ylim([0, 1])
        ax[0, 1].set_xticks([])
        ax[0, 1].set_yticks([])
        ax[0, 1].set_title('u-UpConv')
        fig.colorbar(cf, ax=ax[0, 1], fraction=0.046, pad=0.04)
        #
        cf = ax[0, 2].scatter(x_star, y_star, c=u_low_res, alpha=0.9, edgecolors='none', cmap='RdYlBu', marker='s', s=4, vmin=-2, vmax=1.6)
        ax[0, 2].axis('square')
        ax[0, 2].set_xlim([0, 1])
        ax[0, 2].set_ylim([0, 1])
        ax[0, 2].set_xticks([])
        ax[0, 2].set_yticks([])
        ax[0, 2].set_title('u-LowRes')
        fig.colorbar(cf, ax=ax[0, 2], fraction=0.046, pad=0.04)
        #
        cf = ax[1, 0].scatter(x_star, y_star, c=v_bicubic-v_upconv, alpha=0.9, edgecolors='none', cmap='RdBu', marker='s', s=4, vmin=-.5, vmax=0.5)
        ax[1, 0].axis('square')
        ax[1, 0].set_xlim([0, 1])
        ax[1, 0].set_ylim([0, 1])
        ax[1, 0].set_xticks([])
        ax[1, 0].set_yticks([])
        ax[1, 0].set_title('v-Bicubic')
        fig.colorbar(cf, ax=ax[1, 0], fraction=0.046, pad=0.04)
        # #
        cf = ax[1, 1].scatter(x_star, y_star, c=v_upconv, alpha=0.9, edgecolors='none', cmap='RdYlBu', marker='s', s=4, vmin=-2.8, vmax=0.5)
        ax[1, 1].axis('square')
        ax[1, 1].set_xlim([0, 1])
        ax[1, 1].set_ylim([0, 1])
        ax[1, 1].set_xticks([])
        ax[1, 1].set_yticks([])
        ax[1, 1].set_title('v-UpConv')
        fig.colorbar(cf, ax=ax[1, 1], fraction=0.046, pad=0.04)
        #
        cf = ax[1, 2].scatter(x_star, y_star, c=v_low_res, alpha=0.9, edgecolors='none', cmap='RdYlBu', marker='s', s=4, vmin=-2.8, vmax=0.5)
        ax[1, 2].axis('square')
        ax[1, 2].set_xlim([0, 1])
        ax[1, 2].set_ylim([0, 1])
        ax[1, 2].set_xticks([])
        ax[1, 2].set_yticks([])
        ax[1, 2].set_title('v-LowRes')
        fig.colorbar(cf, ax=ax[1, 2], fraction=0.046, pad=0.04)
        #
        plt.savefig('./pretrained_upscalar.png', dpi=200)
        plt.close('all')

def train(model, truth, n_iters, time_batch_size, learning_rate, dt, dx, restart=True):  # restart=False
    train_loss_list = []
    best_loss = 10000
    if restart:
        model, optimizer, scheduler = load_model(model)
    else:
        optimizer = optim.Adam(model.parameters(), lr = learning_rate)
        scheduler = StepLR(optimizer, step_size=100, gamma=0.95)
    loss_func = loss_generator(dt, dx)
    for epoch in range(n_iters):
        # input: [time batch, channel, height, width]
        optimizer.zero_grad()
        num_time_batch = 1
        batch_loss, phy_loss, ic_loss, data_loss, val_loss = [0] * 5
        for time_batch_id in range(num_time_batch):
            if time_batch_id == 0:
                pass
            else:
                pass
            output, second_last_state = model()                # output is a list
            output = torch.cat(tuple(output), dim=0)
            mse_loss = nn.MSELoss()
            # Downsample the ground truth as measurement, noise is already added while loading data
            pred, gt = output[0:-1:5, :, ::2, ::2], truth[::5, :, ::2, ::2].cuda()
            idx = int(pred.shape[0] * 0.9)
            pred_tra, pred_val = pred[:idx], pred[idx:]  # prediction
            gt_tra, gt_val = gt[:idx], gt[idx:]          # ground truth
            loss_data = mse_loss(pred_tra, gt_tra)       # data loss
            loss_valid = mse_loss(pred_val, gt_val)
            loss_ic   = get_ic_loss(model)
            # get physics loss (for validation, not used for training)
            loss_phy = loss_gen(output, loss_func)
            loss = 1*loss_data + 1.0*loss_ic    # + loss_phy
            loss.backward(retain_graph=True)
            batch_loss += loss.item()
            phy_loss, ic_loss, data_loss, val_loss = loss_phy.item(), loss_ic.item(), loss_data.item(), loss_valid.item()
            # state_detached = second_last_state.detach()
        optimizer.step()
        scheduler.step()
        # print loss in each epoch
        print('[%d/%d %d%%] loss: %.11f, ic_loss: %.11f, data_loss: %.11f, val_loss: %.7f, phy_loss: %.11f' % ((epoch+1), n_iters, ((epoch+1)/n_iters*100.0),
            batch_loss, ic_loss, data_loss, val_loss, phy_loss))
        train_loss_list.append(batch_loss)
        # Save checkpoint if hit best_loss (check every 10 epoch)
        if val_loss < best_loss and epoch%10==0:
            best_loss = val_loss
            for param_group in optimizer.param_groups:
                print(param_group['lr'])
            print('save model!!!')
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, './model/checkpoint.pt')
            # Record coef. history
            model.crnn_cell.show_coef()
    return train_loss_list


def save_model(model, model_name, save_path):
    ''' save the model '''
    torch.save(model.state_dict(), save_path + model_name + '.pt')

def load_model(model):
    # Load model and optimizer state
    checkpoint = torch.load('./model/checkpoint.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer = optim.Adam(model.parameters(), lr=0.0)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.97)
    return model, optimizer, scheduler

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

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def postProcess(output, truth, low_res, xmin, xmax, ymin, ymax, num, fig_save_path):
    ''' num: Number of time step
      '''
    x = np.linspace(0, 1, 101)
    y = np.linspace(0, 1, 101)
    x_star, y_star = np.meshgrid(x, y)
    u_low_res, v_low_res = low_res[num, 0, ...], low_res[num, 1, ...]
    u_low_res, v_low_res = np.kron(u_low_res.detach().cpu().numpy(), np.ones((2, 2))), \
                           np.kron(v_low_res.detach().cpu().numpy(), np.ones((2, 2)))
    u_low_res, v_low_res = np.concatenate((u_low_res, u_low_res[:, 0:1]), axis=1), \
                           np.concatenate((v_low_res, v_low_res[:, 0:1]), axis=1)
    u_low_res, v_low_res = np.concatenate((u_low_res, u_low_res[0:1, :]), axis=0), \
                           np.concatenate((v_low_res, v_low_res[0:1, :]), axis=0)
    u_star, v_star = truth[num, 0, ...].numpy(), truth[num, 1, ...].numpy()
    u_pred, v_pred = output[num, 0, :, :].detach().cpu().numpy(), \
                     output[num, 1, :, :].detach().cpu().numpy()
    #
    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(11, 7))
    fig.subplots_adjust(hspace=0.25, wspace=0.25)
    #
    cf = ax[0, 0].scatter(x_star, y_star, c=u_pred-u_star, alpha=1.0, edgecolors='none', cmap='RdBu', marker='s', s=4, vmin=-.5, vmax=.5)
    ax[0, 0].axis('square')
    ax[0, 0].set_xlim([xmin, xmax])
    ax[0, 0].set_ylim([ymin, ymax])
    ax[0, 0].set_xticks([])
    ax[0, 0].set_yticks([])
    ax[0, 0].set_title('u-u (PeCRNN)')
    fig.colorbar(cf, ax=ax[0, 0], fraction=0.046, pad=0.04)
    #
    cf = ax[0, 1].scatter(x_star, y_star, c=u_star, alpha=1.0, edgecolors='none', cmap='RdYlBu', marker='s', s=4, vmin=-2, vmax=1.6)
    ax[0, 1].axis('square')
    ax[0, 1].set_xlim([xmin, xmax])
    ax[0, 1].set_ylim([ymin, ymax])
    ax[0, 1].set_xticks([])
    ax[0, 1].set_yticks([])
    ax[0, 1].set_title('u (Ref.)')
    fig.colorbar(cf, ax=ax[0, 1], fraction=0.046, pad=0.04)
    #
    cf = ax[0, 2].scatter(x_star, y_star, c=u_low_res, alpha=1.0, edgecolors='none', cmap='RdYlBu', marker='s', s=4, vmin=-2, vmax=1.6)
    ax[0, 2].axis('square')
    ax[0, 2].set_xlim([xmin, xmax])
    ax[0, 2].set_ylim([ymin, ymax])
    ax[0, 2].set_xticks([])
    ax[0, 2].set_yticks([])
    ax[0, 2].set_title('u (Meas.)')
    fig.colorbar(cf, ax=ax[0, 2], fraction=0.046, pad=0.04)
    #
    cf = ax[1, 0].scatter(x_star, y_star, c=v_pred-v_star, alpha=1.0, edgecolors='none', cmap='RdBu', marker='s', s=4, vmin=-.5, vmax=0.5)
    ax[1, 0].axis('square')
    ax[1, 0].set_xlim([xmin, xmax])
    ax[1, 0].set_ylim([ymin, ymax])
    ax[1, 0].set_xticks([])
    ax[1, 0].set_yticks([])
    ax[1, 0].set_title('v-v (PeCRNN)')
    fig.colorbar(cf, ax=ax[1, 0], fraction=0.046, pad=0.04)
    #
    cf = ax[1, 1].scatter(x_star, y_star, c=v_star, alpha=1.0, edgecolors='none', cmap='RdYlBu', marker='s', s=4, vmin=-2.8, vmax=0.5)
    ax[1, 1].axis('square')
    ax[1, 1].set_xlim([xmin, xmax])
    ax[1, 1].set_ylim([ymin, ymax])
    ax[1, 1].set_xticks([])
    ax[1, 1].set_yticks([])
    ax[1, 1].set_title('v (Ref.)')
    fig.colorbar(cf, ax=ax[1, 1], fraction=0.046, pad=0.04)
    #
    cf = ax[1, 2].scatter(x_star, y_star, c=v_low_res, alpha=1.0, edgecolors='none', cmap='RdYlBu', marker='s', s=4, vmin=-2.8, vmax=0.5)
    ax[1, 2].axis('square')
    ax[1, 2].set_xlim([xmin, xmax])
    ax[1, 2].set_ylim([ymin, ymax])
    ax[1, 2].set_xticks([])
    ax[1, 2].set_yticks([])
    ax[1, 2].set_title('v (Meas.)')
    fig.colorbar(cf, ax=ax[1, 2], fraction=0.046, pad=0.04)
    #
    plt.savefig(fig_save_path + 'uv_comparison_'+str(num).zfill(3)+'.png')
    plt.close('all')

def postProcess_v2(output, truth, low_res, xmin, xmax, ymin, ymax, num, fig_save_path):
    ''' num: Number of time step
      '''
    x = np.linspace(0, 1, 101)
    y = np.linspace(0, 1, 101)
    x_star, y_star = np.meshgrid(x, y)
    u_low_res, v_low_res = low_res[num, 0, ...], low_res[num, 1, ...]
    u_low_res, v_low_res = np.kron(u_low_res.detach().cpu().numpy(), np.ones((2, 2))), \
                           np.kron(v_low_res.detach().cpu().numpy(), np.ones((2, 2)))
    u_low_res, v_low_res = np.concatenate((u_low_res, u_low_res[:, 0:1]), axis=1), \
                           np.concatenate((v_low_res, v_low_res[:, 0:1]), axis=1)
    u_low_res, v_low_res = np.concatenate((u_low_res, u_low_res[0:1, :]), axis=0), \
                           np.concatenate((v_low_res, v_low_res[0:1, :]), axis=0)
    u_star, v_star = truth[num, 0, ...], truth[num, 1, ...]
    u_pred, v_pred = output[num, 0, :, :].detach().cpu().numpy(), \
                     output[num, 1, :, :].detach().cpu().numpy()
    #
    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(11, 7))
    fig.subplots_adjust(hspace=0.25, wspace=0.25)
    #
    cf = ax[0, 0].scatter(x_star, y_star, c=u_pred, alpha=1.0, edgecolors='none', cmap='RdYlBu', marker='s', s=4, vmin=-2, vmax=1.6)
    ax[0, 0].axis('square')
    ax[0, 0].set_xlim([xmin, xmax])
    ax[0, 0].set_ylim([ymin, ymax])
    ax[0, 0].set_xticks([])
    ax[0, 0].set_yticks([])
    ax[0, 0].set_title('u (PeCRNN)')
    fig.colorbar(cf, ax=ax[0, 0], fraction=0.046, pad=0.04)
    #
    cf = ax[0, 1].scatter(x_star, y_star, c=u_star, alpha=1.0, edgecolors='none', cmap='RdYlBu', marker='s', s=4, vmin=-2, vmax=1.6)
    ax[0, 1].axis('square')
    ax[0, 1].set_xlim([xmin, xmax])
    ax[0, 1].set_ylim([ymin, ymax])
    ax[0, 1].set_xticks([])
    ax[0, 1].set_yticks([])
    ax[0, 1].set_title('u (Ref.)')
    fig.colorbar(cf, ax=ax[0, 1], fraction=0.046, pad=0.04)
    #
    cf = ax[0, 2].scatter(x_star, y_star, c=u_low_res, alpha=1.0, edgecolors='none', cmap='RdYlBu', marker='s', s=4, vmin=-2, vmax=1.6)
    ax[0, 2].axis('square')
    ax[0, 2].set_xlim([xmin, xmax])
    ax[0, 2].set_ylim([ymin, ymax])
    ax[0, 2].set_xticks([])
    ax[0, 2].set_yticks([])
    ax[0, 2].set_title('u (Meas.)')
    fig.colorbar(cf, ax=ax[0, 2], fraction=0.046, pad=0.04)
    #
    cf = ax[1, 0].scatter(x_star, y_star, c=v_pred, alpha=1.0, edgecolors='none', cmap='RdYlBu', marker='s', s=4, vmin=-2.8, vmax=0.5)
    ax[1, 0].axis('square')
    ax[1, 0].set_xlim([xmin, xmax])
    ax[1, 0].set_ylim([ymin, ymax])
    ax[1, 0].set_xticks([])
    ax[1, 0].set_yticks([])
    ax[1, 0].set_title('v (PeCRNN)')
    fig.colorbar(cf, ax=ax[1, 0], fraction=0.046, pad=0.04)
    #
    cf = ax[1, 1].scatter(x_star, y_star, c=v_star, alpha=1.0, edgecolors='none', cmap='RdYlBu', marker='s', s=4, vmin=-2.8, vmax=0.5)
    ax[1, 1].axis('square')
    ax[1, 1].set_xlim([xmin, xmax])
    ax[1, 1].set_ylim([ymin, ymax])
    ax[1, 1].set_xticks([])
    ax[1, 1].set_yticks([])
    ax[1, 1].set_title('v (Ref.)')
    fig.colorbar(cf, ax=ax[1, 1], fraction=0.046, pad=0.04)
    #
    cf = ax[1, 2].scatter(x_star, y_star, c=v_low_res, alpha=1.0, edgecolors='none', cmap='RdYlBu', marker='s', s=4, vmin=-2.8, vmax=0.5)
    ax[1, 2].axis('square')
    ax[1, 2].set_xlim([xmin, xmax])
    ax[1, 2].set_ylim([ymin, ymax])
    ax[1, 2].set_xticks([])
    ax[1, 2].set_yticks([])
    ax[1, 2].set_title('v (Meas.)')
    fig.colorbar(cf, ax=ax[1, 2], fraction=0.046, pad=0.04)
    #
    plt.savefig(fig_save_path + 'uv_comparison_'+str(num).zfill(3)+'.png')
    plt.close('all')

def plot_IC(model, ic_GT, init_state_low, fig_save_path):
    xmin, xmax = 0, 1
    ymin, ymax = 0, 1
    init_state_bicubic = F.interpolate(init_state_low, (100, 100), mode='bicubic')
    init_state_upconv = model.UpconvBlock(init_state_low)

    x = np.linspace(0, 1, 101)
    y = np.linspace(0, 1, 101)
    x_star, y_star = np.meshgrid(x[:-1], y[:-1])

    # Ground truth
    u_star, v_star = ic_GT[0, 0, ...].numpy(), ic_GT[0, 1, ...].numpy()
    # IC generator
    u_Genr, v_Genr = init_state_upconv[0, 0, :, :].detach().cpu().numpy(), init_state_upconv[0, 1, :, :].detach().cpu().numpy()
    # Interp IC
    u_Intrp, v_Intrp = init_state_bicubic[0, 0, ...].detach().cpu().numpy(), init_state_bicubic[0, 1, ...].detach().cpu().numpy()
    #
    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(11, 7))
    fig.subplots_adjust(hspace=0.25, wspace=0.25)
    #
    cf = ax[0, 0].scatter(x_star, y_star, c=u_Genr-u_star, alpha=1.0, edgecolors='none', cmap='RdBu', marker='s', s=4, vmin=-.5, vmax=.5)
    ax[0, 0].axis('square')
    ax[0, 0].set_xlim([xmin, xmax])
    ax[0, 0].set_ylim([ymin, ymax])
    ax[0, 0].set_xticks([])
    ax[0, 0].set_yticks([])
    ax[0, 0].set_title('Gen-GT')
    fig.colorbar(cf, ax=ax[0, 0], fraction=0.046, pad=0.04)
    #
    cf = ax[0, 1].scatter(x_star, y_star, c=u_star, alpha=1.0, edgecolors='none', cmap='RdYlBu', marker='s', s=4, vmin=-2, vmax=1.6)
    ax[0, 1].axis('square')
    ax[0, 1].set_xlim([xmin, xmax])
    ax[0, 1].set_ylim([ymin, ymax])
    ax[0, 1].set_xticks([])
    ax[0, 1].set_yticks([])
    ax[0, 1].set_title('u (Ref.)')
    fig.colorbar(cf, ax=ax[0, 1], fraction=0.046, pad=0.04)
    #
    cf = ax[0, 2].scatter(x_star, y_star, c=u_Intrp-u_star, alpha=1.0, edgecolors='none', cmap='RdBu', marker='s', s=4, vmin=-0.5, vmax=.5)
    ax[0, 2].axis('square')
    ax[0, 2].set_xlim([xmin, xmax])
    ax[0, 2].set_ylim([ymin, ymax])
    ax[0, 2].set_xticks([])
    ax[0, 2].set_yticks([])
    ax[0, 2].set_title('Intrp-GT')
    fig.colorbar(cf, ax=ax[0, 2], fraction=0.046, pad=0.04)
    #
    cf = ax[1, 0].scatter(x_star, y_star, c=v_Genr-v_star, alpha=1.0, edgecolors='none', cmap='RdBu', marker='s', s=4, vmin=-.5, vmax=0.5)
    ax[1, 0].axis('square')
    ax[1, 0].set_xlim([xmin, xmax])
    ax[1, 0].set_ylim([ymin, ymax])
    ax[1, 0].set_xticks([])
    ax[1, 0].set_yticks([])
    ax[1, 0].set_title('Gen-GT')
    fig.colorbar(cf, ax=ax[1, 0], fraction=0.046, pad=0.04)
    #
    cf = ax[1, 1].scatter(x_star, y_star, c=v_star, alpha=1.0, edgecolors='none', cmap='RdYlBu', marker='s', s=4, vmin=-2.8, vmax=0.5)
    ax[1, 1].axis('square')
    ax[1, 1].set_xlim([xmin, xmax])
    ax[1, 1].set_ylim([ymin, ymax])
    ax[1, 1].set_xticks([])
    ax[1, 1].set_yticks([])
    ax[1, 1].set_title('v (Ref.)')
    fig.colorbar(cf, ax=ax[1, 1], fraction=0.046, pad=0.04)
    #
    cf = ax[1, 2].scatter(x_star, y_star, c=v_Intrp-v_star, alpha=1.0, edgecolors='none', cmap='RdBu', marker='s', s=4, vmin=-.5, vmax=0.5)
    ax[1, 2].axis('square')
    ax[1, 2].set_xlim([xmin, xmax])
    ax[1, 2].set_ylim([ymin, ymax])
    ax[1, 2].set_xticks([])
    ax[1, 2].set_yticks([])
    ax[1, 2].set_title('Intrp-GT')
    fig.colorbar(cf, ax=ax[1, 2], fraction=0.046, pad=0.04)
    #
    plt.savefig(fig_save_path + 'IC_comparison.png')
    plt.close('all')

def summary_parameters(model):
    for i in model.parameters():
        print(i.shape)

def show_trainable(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.data)

if __name__ == '__main__':

    ################# prepare the input dataset ####################
    time_steps = 200    # 400 for training, 1200 for inference
    dt = 0.00025
    dx = 1.0/100
    dy = 1.0/100

    import scipy.io as sio
    UV = sio.loadmat('../data/Burgers_2001x2x100x100_[FWE,dt=00025].mat')['uv']
    truth_clean = torch.from_numpy(UV[100:1901]).float()  # [1801, 2, 100 X, 100 Y]
    # Add noise 5%
    UV = add_noise(truth_clean, pec=0.1)
    # Retrieve initial condition
    IC = UV[0:1, :, :, :]

    # get the ground truth (noisy) for training
    truth = UV[:time_steps+1]

    time_batch_size = time_steps
    steps = time_batch_size + 1
    effective_step = list(range(0, steps))
    n_iters = 2000
    learning_rate = 5e-4
    save_path = './model/'                      # [201, 2, 100, 100]

    # Low-res initial condition
    U0_low = IC[0, 0, ::2, ::2]
    V0_low = IC[0, 1, ::2, ::2]
    h0 = torch.cat((U0_low[None, None, ...], V0_low[None, None, ...]), dim=1)
    init_state_low = h0.cuda()

    model = RCNN(
                input_channels = 2,
                hidden_channels = 5,
                output_channels = 2,
                init_state_low = init_state_low,
                input_kernel_size = 5,
                input_stride = 1,
                input_padding = 1,
                step = steps,
                effective_step = effective_step).cuda()

    # train the model
    start = time.time()
    pretrain_upscaler(model.UpconvBlock, init_state_low, epoch=5000, plotFlag=True)
    train_loss_list = train(model, truth, n_iters, time_batch_size, learning_rate, dt, dx, restart=False)
    end = time.time()

    print('The training time is: ', (end-start))

    with torch.no_grad():
        # Do the forward inference
        output, _ = model()
        output = torch.cat(tuple(output), dim=0)



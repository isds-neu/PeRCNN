import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import StepLR
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import scipy.io as scio
import time
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
torch.set_default_dtype(torch.float32)

torch.manual_seed(66)
np.random.seed(66)

lap_2d_op = [[[[    0,   0, -1/12,   0,     0],
               [    0,   0,   4/3,   0,     0],
               [-1/12, 4/3,   - 5, 4/3, -1/12],
               [    0,   0,   4/3,   0,     0],
               [    0,   0, -1/12,   0,     0]]]]

class upscaler(nn.Module):
    ''' Upscaler to convert low-res to high-res initial state '''

    def __init__(self):
        super(upscaler, self).__init__()
        self.layers = []
        self.layers.append(
            nn.ConvTranspose2d(2, 8, kernel_size=5, padding=5 // 2, stride=2, output_padding=1, bias=True))
        self.layers.append(torch.nn.Sigmoid())
        self.layers.append(
            nn.ConvTranspose2d(8, 8, kernel_size=5, padding=5 // 2, stride=2, output_padding=1, bias=True))
        self.layers.append(nn.Conv2d(8, 2, 1, 1, padding=0, bias=True))
        self.convnet = torch.nn.Sequential(*self.layers)

    def forward(self, h):
        return self.convnet(h)

class RCNNCell(nn.Module):
    ''' Recurrent convolutional neural network Cell '''
   
    def __init__(self, input_channels, hidden_channels, input_kernel_size):

        super(RCNNCell, self).__init__()

        # the initial parameters
        self.input_channels = input_channels  # no use, always 2
        self.hidden_channels = hidden_channels
        self.input_kernel_size = 5
        self.input_stride = 1

        self.dx = 0.01
        self.dt = 0.5
        self.mu_up = 3.89e-5  # upper bound for the diffusion coefficient
        # Design the laplace_u term # [-1, 1]
        np.random.seed(1234)
        self.CA = torch.nn.Parameter(torch.tensor((np.random.rand()-0.5)*2, dtype=torch.float32), requires_grad=True)
        self.CB = torch.nn.Parameter(torch.tensor((np.random.rand()-0.5)*2, dtype=torch.float32), requires_grad=True)

        # padding_mode='replicate' not working for the test
        self.W_laplace = nn.Conv2d(1, 1, self.input_kernel_size, self.input_stride, padding=0, bias=False)
        self.W_laplace.weight.data = 1/self.dx**2*torch.tensor(lap_2d_op, dtype=torch.float32)
        self.W_laplace.weight.requires_grad = False

        # Nonlinear term for u (up to 3rd order)
        self.Wh1_u = nn.Conv2d(in_channels=2, out_channels=hidden_channels, kernel_size=1,
                               stride=self.input_stride, padding=0, bias=True, )
        self.Wh2_u = nn.Conv2d(in_channels=2, out_channels=hidden_channels, kernel_size=1,
                               stride=self.input_stride, padding=0, bias=True, )
        self.Wh3_u = nn.Conv2d(in_channels=2, out_channels=hidden_channels, kernel_size=1,
                               stride=self.input_stride, padding=0, bias=True, )
        self.Wh4_u = nn.Conv2d(in_channels=hidden_channels, out_channels=1, kernel_size=1,
                               stride=1, padding=0, bias=True)
        # Nonlinear term for v (up to 3rd order)
        self.Wh1_v = nn.Conv2d(in_channels=2, out_channels=hidden_channels, kernel_size=1,
                               stride=self.input_stride, padding=0, bias=True, )
        self.Wh2_v = nn.Conv2d(in_channels=2, out_channels=hidden_channels, kernel_size=1,
                               stride=self.input_stride, padding=0, bias=True, )
        self.Wh3_v = nn.Conv2d(in_channels=2, out_channels=hidden_channels, kernel_size=1,
                               stride=self.input_stride, padding=0, bias=True, )
        self.Wh4_v = nn.Conv2d(in_channels=hidden_channels, out_channels=1, kernel_size=1,
                               stride=1, padding=0, bias=True)

        # initialize filter's wweight and bias
        self.filter_list = [self.Wh1_u, self.Wh2_u, self.Wh3_u, self.Wh4_u, self.Wh1_v, self.Wh2_v, self.Wh3_v, self.Wh4_v]
        self.init_filter(self.filter_list, c=0.02)

    def init_filter(self, filter_list, c):
        '''
        :param filter_list: list of filter for initialization
        :param c: constant multiplied on Xavier initialization
        '''
        for filter in filter_list:
            # Xavier initialization and then scale
            torch.nn.init.xavier_uniform_(filter.weight)
            filter.weight.data = c*filter.weight.data
            # filter.weight.data.uniform_(-c * np.sqrt(1 / (5 * 5 * 16)), c * np.sqrt(1 / (5 * 5 * 16)))
            if filter.bias is not None:
                filter.bias.data.fill_(0.0)

    def forward(self, h):

        # periodic padding, can also be achieved using 'circular' padding
        h_pad = torch.cat((    h[:, :, :, -2:],     h,     h[:, :, :, 0:2]), dim=3)
        h_pad = torch.cat((h_pad[:, :, -2:, :], h_pad, h_pad[:, :, 0:2, :]), dim=2)
        u_pad = h_pad[:, 0:1, ...]  # 104x104
        v_pad = h_pad[:, 1:2, ...]
        u_prev = h[:, 0:1, ...]     # 100x100
        v_prev = h[:, 1:2, ...]

        u_res = self.mu_up*torch.sigmoid(self.CA)*self.W_laplace(u_pad) + self.Wh4_u( self.Wh1_u(h)*self.Wh2_u(h)*self.Wh3_u(h) )
        v_res = self.mu_up*torch.sigmoid(self.CB)*self.W_laplace(v_pad) + self.Wh4_v( self.Wh1_v(h)*self.Wh2_v(h)*self.Wh3_v(h) )
        u_next = u_prev + u_res * self.dt
        v_next = v_prev + v_res * self.dt
        ch = torch.cat((u_next, v_next), dim=1)

        return ch, ch


    def init_hidden_tensor(self, prev_state):
        return prev_state.cuda()


class RCNN(nn.Module):

    ''' Recurrent convolutional neural network layer '''

    def __init__(self, input_channels, hidden_channels, init_state_low, input_kernel_size,
                       step=1, effective_step=[1]):

        super(RCNN, self).__init__()
        
        # input channels of layer includes input_channels and hidden_channels of cells 
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.output_channels = 1  # no use, always 1
        self.input_kernel_size = input_kernel_size
        self.step = step
        self.effective_step = effective_step
        self._all_layers = []
        # self.init_state = torch.nn.Parameter(torch.tensor(ini_state, dtype=torch.float32).cuda(), requires_grad=False)
        self.init_state_low = init_state_low
        self.init_state = []

        # Upconv as initial state generator
        self.UpconvBlock = upscaler().cuda()

        name = 'crnn_cell'
        cell = RCNNCell(
            input_channels = self.input_channels,
            hidden_channels = self.hidden_channels,
            input_kernel_size = self.input_kernel_size)

        setattr(self, name, cell)
        self._all_layers.append(cell)


    def forward(self):

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


class Conv2dDerivative(nn.Module):
    def __init__(self, DerFilter, deno, kernel_size=5, name=''):
        '''
        :param DerFilter: constructed derivative filter, e.g. Laplace filter
        :param deno: resolution of the filter, used to divide the output, e.g. c*dt, c*dx or c*dx^2
        :param kernel_size:
        :param name: optional name for the operator
        '''
        super(Conv2dDerivative, self).__init__()
        self.deno = deno  # constant in the finite difference
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
        return derivative / self.deno


class Conv1dDerivative(nn.Module):
    def __init__(self, DerFilter, deno, kernel_size=3, name=''):
        super(Conv1dDerivative, self).__init__()

        self.deno = deno  # $\delta$*constant in the finite difference
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
        return derivative / self.deno


class loss_generator(nn.Module):
    ''' Loss generator for physics loss '''

    def __init__(self, dt = (1.0/2), dx = (1.0/100)):

        '''
        Construct the derivatives, X = Width, Y = Height      

        '''

        self.dt = dt
        self.dx = dx
       
        super(loss_generator, self).__init__()

        # spatial derivative operator
        self.laplace = Conv2dDerivative(
            DerFilter = lap_2d_op,
            deno = (dx**2),
            kernel_size = 5,
            name = 'laplace_operator').cuda()

        # temporal derivative operator
        self.dt = Conv1dDerivative(
            DerFilter = [[[-1, 1, 0]]],
            deno = (dt*1),
            kernel_size = 3,
            name = 'partial_t').cuda()

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

        u = output[0:-2, 0:1, 2:-2, 2:-2]  # [step, c, height(Y), width(X)]
        v = output[0:-2, 1:2, 2:-2, 2:-2]  # [step, c, height(Y), width(X)]

        # make sure the dimensions consistent
        assert laplace_u.shape == u_t.shape
        assert u_t.shape == v_t.shape
        assert laplace_u.shape == u.shape
        assert laplace_v.shape == v.shape

        # gray scott eqn
        Du = 2e-5
        Dv = Du/4
        f = 1/25
        k = 3/50

        f_u = (Du*laplace_u - u*(v**2) + f*(1-u) - u_t)/1
        f_v = (Dv*laplace_v + u*(v**2) - (f+k)*v - v_t)/1

        return f_u, f_v

def get_ic_loss(model):
    Upconv = model.UpconvBlock
    init_state_low = model.init_state_low
    init_state_bicubic = F.interpolate(init_state_low, (100, 100), mode='bicubic')
    mse_loss = nn.MSELoss()
    init_state_pred = Upconv(init_state_low)
    loss_ic = mse_loss(init_state_pred, init_state_bicubic)
    return loss_ic

def loss_gen(output, loss_func):
    '''calculate the phycis loss'''
    
    # Padding x axis due to periodic boundary condition
    # shape: [27, 2, 131, 131]
    output = torch.cat((output[:, :, :, -2:], output, output[:, :, :, 0:3]), dim=3)
    output = torch.cat((output[:, :, -2:, :], output, output[:, :, 0:3, :]), dim=2)

    # get physics loss
    mse_loss = nn.MSELoss()
    f_u, f_v = loss_func.get_phy_Loss(output)
    loss = mse_loss(f_u, torch.zeros_like(f_u).cuda()) + \
           mse_loss(f_v, torch.zeros_like(f_v).cuda())
    return loss

def pretrain_upscaler(Upconv, init_state_low, epoch=4000):
    '''
    :param Upconv: upscalar model
    :param init_state_low: low resolution measurement
    :return:
    '''
    init_state_bicubic = F.interpolate(init_state_low, (100, 100), mode='bicubic')
    optimizer = optim.Adam(Upconv.parameters(), lr = 0.02)
    scheduler = StepLR(optimizer, step_size=100, gamma=0.99)
    mse_loss = nn.MSELoss()
    for epoch in range(epoch):
        optimizer.zero_grad()
        init_state_pred = Upconv(init_state_low)
        loss = mse_loss(init_state_pred, init_state_bicubic)
        loss.backward(retain_graph=True)
        print('[%d] loss: %.9f' % ((epoch+1), loss.item()))
        optimizer.step()
        scheduler.step()

def train(model, truth, n_iters, time_batch_size, learning_rate, dt, dx, cont=True):
    # define some parameters
    train_loss_list = []
    # truth = truth[:, :, :-1, :-1] 
    # model
    if cont:
        model, optimizer, scheduler = load_model(model)
    else:
        optimizer = optim.Adam(model.parameters(), lr = learning_rate)
        scheduler = StepLR(optimizer, step_size=200, gamma=0.985)
    #for param_group in optimizer.param_groups:
    #    param_group['lr']=0.0012
    loss_func = loss_generator(dt, dx)
    for epoch in range(n_iters):
        # input: [time,. batch, channel, height, width]
        optimizer.zero_grad()
        num_time_batch = 1
        batch_loss, phy_loss, ic_loss, data_loss, val_loss = [0]*5
        # One single batch
        output, second_last_state = model()  # output is a list
        output = torch.cat(tuple(output), dim=0)
        mse_loss = nn.MSELoss()
        # 21x2x50x50
        pred, gt = output[0:-1:20, :, ::4, ::4], truth[::20, :, ::4, ::4].cuda()
        idx = int(pred.shape[0]*0.9)
        pred_tra, pred_val = pred[:idx], pred[idx:]     # prediction
        gt_tra,   gt_val   =   gt[:idx],   gt[idx:]     # ground truth
        loss_data  = mse_loss(pred_tra, gt_tra)         # data loss
        loss_valid = mse_loss(pred_val, gt_val)
        loss_ic   = get_ic_loss(model)
        # get physics loss (for validation, not used for training)
        loss_phy = loss_gen(output, loss_func)
        loss = 40*loss_data + 0.25*loss_ic
        loss.backward(retain_graph=True)
        batch_loss += loss.item()
        phy_loss, ic_loss, data_loss, val_loss  = loss_phy.item(), loss_ic.item(), loss_data.item(), loss_valid.item()
        optimizer.step()
        scheduler.step()
        # print loss in each epoch
        print('[%d/%d %d%%] loss: %.7f, ic_loss: %.7f, data_loss: %.7f, val_loss: %.7f, loss phy_loss: %.8f' % ((epoch+1), n_iters, ((epoch+1)/n_iters*100.0),
              batch_loss, ic_loss, data_loss, val_loss, phy_loss))
        train_loss_list.append(batch_loss)
        # Save checkpoint
        if (epoch+1)%100 == 0:
            for param_group in optimizer.param_groups:
                print(param_group['lr'])
            print('save model!!!')
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, './model/checkpoint.pt')
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
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.98)
    return model, optimizer, scheduler


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def postProcess_2x3(output, truth, low_res, xmin, xmax, ymin, ymax, num, fig_save_path):
    ''' num: Number of time step
    '''
    x = np.linspace(-0.5, 0.5, 101)
    y = np.linspace(-0.5, 0.5, 101)
    x_star, y_star = np.meshgrid(x, y)
    u_low_res, v_low_res = low_res[num, 0, ...], low_res[num, 1, ...]
    u_low_res, v_low_res = np.kron(u_low_res.detach().cpu().numpy(), np.ones((4, 4))), \
                           np.kron(v_low_res.detach().cpu().numpy(), np.ones((4, 4)))
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
    cf = ax[0, 0].scatter(x_star, y_star, c=u_pred, alpha=1.0, edgecolors='none', cmap='hot', marker='s', s=4, vmin=0, vmax=1)
    ax[0, 0].axis('square')
    ax[0, 0].set_xlim([xmin, xmax])
    ax[0, 0].set_ylim([ymin, ymax])
    ax[0, 0].set_xticks([])
    ax[0, 0].set_yticks([])
    ax[0, 0].set_title('u (PeCRNN)')
    fig.colorbar(cf, ax=ax[0, 0], fraction=0.046, pad=0.04)
    #
    cf = ax[0, 1].scatter(x_star, y_star, c=u_star, alpha=1.0, edgecolors='none', cmap='hot', marker='s', s=4, vmin=0, vmax=1)
    ax[0, 1].axis('square')
    ax[0, 1].set_xlim([xmin, xmax])
    ax[0, 1].set_ylim([ymin, ymax])
    ax[0, 1].set_xticks([])
    ax[0, 1].set_yticks([])
    ax[0, 1].set_title('u (Ref.)')
    fig.colorbar(cf, ax=ax[0, 1], fraction=0.046, pad=0.04)
    #
    cf = ax[0, 2].scatter(x_star, y_star, c=u_low_res, alpha=1.0, edgecolors='none', cmap='hot', marker='s', s=4, vmin=0, vmax=1)
    ax[0, 2].axis('square')
    ax[0, 2].set_xlim([xmin, xmax])
    ax[0, 2].set_ylim([ymin, ymax])
    ax[0, 2].set_xticks([])
    ax[0, 2].set_yticks([])
    ax[0, 2].set_title('u (Meas.)')
    fig.colorbar(cf, ax=ax[0, 2], fraction=0.046, pad=0.04)
    #
    cf = ax[1, 0].scatter(x_star, y_star, c=v_pred, alpha=1.0, edgecolors='none', cmap='hot', marker='s', s=4, vmin=0, vmax=1)
    ax[1, 0].axis('square')
    ax[1, 0].set_xlim([xmin, xmax])
    ax[1, 0].set_ylim([ymin, ymax])
    ax[1, 0].set_xticks([])
    ax[1, 0].set_yticks([])
    ax[1, 0].set_title('v (PeCRNN)')
    fig.colorbar(cf, ax=ax[1, 0], fraction=0.046, pad=0.04)
    #
    cf = ax[1, 1].scatter(x_star, y_star, c=v_star, alpha=1.0, edgecolors='none', cmap='hot', marker='s', s=4, vmin=0, vmax=1)
    ax[1, 1].axis('square')
    ax[1, 1].set_xlim([xmin, xmax])
    ax[1, 1].set_ylim([ymin, ymax])
    ax[1, 1].set_xticks([])
    ax[1, 1].set_yticks([])
    ax[1, 1].set_title('v (Ref.)')
    fig.colorbar(cf, ax=ax[1, 1], fraction=0.046, pad=0.04)
    #
    cf = ax[1, 2].scatter(x_star, y_star, c=v_low_res, alpha=1.0, edgecolors='none', cmap='hot', marker='s', s=4, vmin=0, vmax=1)
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

def postProcess_2x2(output, truth, xmin, xmax, ymin, ymax, num, fig_save_path):
    ''' num: Number of time step
    '''
    x = np.linspace(-0.5, 0.5, 101)
    y = np.linspace(-0.5, 0.5, 101)
    x_star, y_star = np.meshgrid(x, y)
    u_star, v_star = truth[num, 0, ...], truth[num, 1, ...]
    u_pred, v_pred = output[num, 0, :, :].detach().cpu().numpy(), \
                     output[num, 1, :, :].detach().cpu().numpy()
    #
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(7, 7))
    fig.subplots_adjust(hspace=0.25, wspace=0.25)
    #
    cf = ax[0, 0].scatter(x_star, y_star, c=u_pred, alpha=1.0, edgecolors='none', cmap='hot', marker='s', s=4, vmin=0, vmax=1)
    ax[0, 0].axis('square')
    ax[0, 0].set_xlim([xmin, xmax])
    ax[0, 0].set_ylim([ymin, ymax])
    ax[0, 0].set_xticks([])
    ax[0, 0].set_yticks([])
    ax[0, 0].set_title('u (PeCRNN)')
    fig.colorbar(cf, ax=ax[0, 0], fraction=0.046, pad=0.04)
    #
    cf = ax[0, 1].scatter(x_star, y_star, c=np.abs(u_star-u_pred), alpha=1.0, edgecolors='none', cmap='hot', marker='s', s=4, vmin=0, vmax=1*0.2)
    ax[0, 1].axis('square')
    ax[0, 1].set_xlim([xmin, xmax])
    ax[0, 1].set_ylim([ymin, ymax])
    ax[0, 1].set_xticks([])
    ax[0, 1].set_yticks([])
    ax[0, 1].set_title('u (Error)')
    fig.colorbar(cf, ax=ax[0, 1], fraction=0.046, pad=0.04)
    #
    cf = ax[1, 0].scatter(x_star, y_star, c=v_pred, alpha=1.0, edgecolors='none', cmap='hot', marker='s', s=4, vmin=0, vmax=1)
    ax[1, 0].axis('square')
    ax[1, 0].set_xlim([xmin, xmax])
    ax[1, 0].set_ylim([ymin, ymax])
    ax[1, 0].set_xticks([])
    ax[1, 0].set_yticks([])
    ax[1, 0].set_title('v (PeCRNN)')
    fig.colorbar(cf, ax=ax[1, 0], fraction=0.046, pad=0.04)
    #
    cf = ax[1, 1].scatter(x_star, y_star, c=np.abs(v_star-v_pred), alpha=1.0, edgecolors='none', cmap='hot', marker='s', s=4, vmin=0, vmax=1*0.2)
    ax[1, 1].axis('square')
    ax[1, 1].set_xlim([xmin, xmax])
    ax[1, 1].set_ylim([ymin, ymax])
    ax[1, 1].set_xticks([])
    ax[1, 1].set_yticks([])
    ax[1, 1].set_title('v (Error)')
    fig.colorbar(cf, ax=ax[1, 1], fraction=0.046, pad=0.04)
    #
    plt.savefig(fig_save_path + 'error_'+str(num).zfill(3)+'.png')
    plt.close('all')

def summary_parameters(model):
    for i in model.parameters():
        print(i.shape)

def add_noise(truth, pec=0.05):  # BxCx101x101
    from torch.distributions import normal
    assert truth.shape[1]==2
    torch.manual_seed(66)
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


if __name__ == '__main__':

    ################# prepare the input dataset ####################
    time_steps = 400   # 200->400->800 multi-stage training works best, then 2500 for inference.
    dt = 0.5
    dx = 1.0/100
    dy = 1.0/100

    ################### define the Initial conditions ####################
    import scipy.io as sio
    data = sio.loadmat('./data/2DRD_2x3001x100x100_[dt=05].mat')['uv']
    UV = np.transpose(data, (1, 0, 2, 3)).astype(np.float32)             # 2001x2x100x100
    truth_clean = torch.from_numpy(UV[:3001])  # [201, 2, 100, 100]
    # Add noise 10%
    UV = add_noise(torch.tensor(UV), pec=0.1)
    # Retrieve initial condition
    IC = UV[0:1, :, :, :]                                                # 1x2x100x100

    # get the ground truth (noisy) for training
    truth = UV[:time_steps+1]                                                    # [201, 2, 100, 100]

    ################# build the model #####################
    # define the mdel hyperparameters
    time_batch_size = time_steps
    steps = time_batch_size + 1
    effective_step = list(range(0, steps))
    n_iters = 5000   # 10000 for 200 steps, 5000 for 4000 steps, 5000 for 800 steps
    learning_rate = 1e-3
    save_path = './model/'

    # Low-res initial condition
    U0_low = IC[0, 0, ::4, ::4]
    V0_low = IC[0, 1, ::4, ::4]
    h0 = torch.cat((U0_low[None, None, ...], V0_low[None, None, ...]), dim=1)
    init_state_low = h0.cuda()

    model = RCNN(
        input_channels = 2, 
        hidden_channels = 8,
        init_state_low = init_state_low,
        input_kernel_size = 5,
        step = steps, 
        effective_step = effective_step).cuda()

    # train the model
    start = time.time()
    cont = True   # if continue training (or use pretrained model), set cont=True
    if not cont:
        pretrain_upscaler(model.UpconvBlock, init_state_low)
    train_loss_list = train(model, truth, n_iters, time_batch_size, learning_rate, dt, dx, cont=cont)
    end = time.time()

    print('The training time is: ', (end-start))


    # Do the forward inference
    output, _ = model()
    output = torch.cat(tuple(output), dim=0)

    # Padding x,y axis due to periodic boundary condition
    output = torch.cat((output, output[:, :, :, 0:1]), dim=3)
    output = torch.cat((output, output[:, :, 0:1, :]), dim=2)
    truth_clean  = torch.cat((truth_clean, truth_clean[:, :, :, 0:1]), dim=3)
    truth_clean  = torch.cat((truth_clean, truth_clean[:, :, 0:1, :]), dim=2)

    # init_state_bicubic = F.interpolate(init_state_low, (100, 100), mode='bicubic')
    low_res = truth[:, :, ::4, ::4] # .cpu().numpy()

    fig_save_path = './figures/'

    # post-process
    for i in range(0, 2001, 10):
        print(i)
        postProcess_2x3(output, truth_clean, low_res, xmin=-0.5, xmax=0.5, ymin=-0.5, ymax=0.5,
                        num=i, fig_save_path=fig_save_path)
        # postProcess_2x2(output, truth_clean, xmin=-0.5, xmax=0.5, ymin=-0.5, ymax=0.5,
        #                 num=i, fig_save_path='./error/')

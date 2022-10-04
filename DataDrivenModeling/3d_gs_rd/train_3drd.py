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
from sympy import pprint, Symbol, exp, sqrt, Matrix
from sympy import init_printing

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
torch.set_default_dtype(torch.float32)

torch.manual_seed(66)
np.random.seed(66)

laplace_3d = np.zeros((1, 1, 5, 5, 5))
elements = [
    (-15/2, (0, 0, 0)),
    (4 / 3, (1, 0, 0)),
    (4 / 3, (0, 1, 0)),
    (4 / 3, (0, 0, 1)),
    (4 / 3, (-1, 0, 0)),
    (4 / 3, (0, -1, 0)),
    (4 / 3, (0, 0, -1)),
    (-1 / 12, (-2, 0, 0)),
    (-1 / 12, (0, -2, 0)),
    (-1 / 12, (0, 0, -2)),
    (-1 / 12, (2, 0, 0)),
    (-1 / 12, (0, 2, 0)),
    (-1 / 12, (0, 0, 2)),
]
for weight, (x, y, z) in elements:
    laplace_3d[0, 0, x+2, y+2, z+2] = weight

class upscaler(nn.Module):
    ''' Upscaler to convert low-res to high-res initial state '''

    def __init__(self):
        super(upscaler, self).__init__()
        self.layers = []
        self.layers.append(
            nn.ConvTranspose3d(2, 8, kernel_size=5, padding=5 // 2, stride=2, output_padding=1, bias=True))
        self.layers.append(torch.nn.Sigmoid())
        self.layers.append(
            nn.ConvTranspose3d(8, 8, kernel_size=5, padding=5 // 2, stride=1, output_padding=0, bias=True))
        self.layers.append(nn.Conv3d(8, 2, 1, 1, padding=0, bias=True))
        self.convnet = torch.nn.Sequential(*self.layers)

    def forward(self, h):
        return self.convnet(h)

class RCNNCell(nn.Module):
    ''' Recurrent convolutional neural network Cell '''

    def __init__(self, input_channels, hidden_channels, input_kernel_size=5):

        super(RCNNCell, self).__init__()

        # the initial parameters
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.input_kernel_size = input_kernel_size
        self.input_stride = 1

        self.dx = 100/48
        self.dt = 0.5
        self.mu_up = 0.274  # upper bound for the diffusion coefficient
        # Design the laplace_u term
        np.random.seed(1234)   # [-1, 1]
        self.CA = torch.nn.Parameter(torch.tensor((np.random.rand()-0.5)*2, dtype=torch.float32), requires_grad=True)
        self.CB = torch.nn.Parameter(torch.tensor((np.random.rand()-0.5)*2, dtype=torch.float32), requires_grad=True)

        # padding_mode='replicate' not working for the test
        self.W_laplace = nn.Conv3d(1, 1, self.input_kernel_size, self.input_stride, padding=0, bias=False)
        self.W_laplace.weight.data = 1/self.dx**2*torch.tensor(laplace_3d, dtype=torch.float32)
        self.W_laplace.weight.requires_grad = False

        # Nonlinear term for u (up to 3rd order)
        self.Wh1_u = nn.Conv3d(in_channels=2, out_channels=hidden_channels, kernel_size=1,
                               stride=self.input_stride, padding=0, bias=True, )
        self.Wh2_u = nn.Conv3d(in_channels=2, out_channels=hidden_channels, kernel_size=1,
                               stride=self.input_stride, padding=0, bias=True, )
        self.Wh3_u = nn.Conv3d(in_channels=2, out_channels=hidden_channels, kernel_size=1,
                               stride=self.input_stride, padding=0, bias=True, )
        self.Wh4_u = nn.Conv3d(in_channels=hidden_channels, out_channels=1, kernel_size=1,
                               stride=1, padding=0, bias=True)
        # Nonlinear term for v (up to 3rd order)
        self.Wh1_v = nn.Conv3d(in_channels=2, out_channels=hidden_channels, kernel_size=1,
                               stride=self.input_stride, padding=0, bias=True, )
        self.Wh2_v = nn.Conv3d(in_channels=2, out_channels=hidden_channels, kernel_size=1,
                               stride=self.input_stride, padding=0, bias=True, )
        self.Wh3_v = nn.Conv3d(in_channels=2, out_channels=hidden_channels, kernel_size=1,
                               stride=self.input_stride, padding=0, bias=True, )
        self.Wh4_v = nn.Conv3d(in_channels=hidden_channels, out_channels=1, kernel_size=1,
                               stride=1, padding=0, bias=True)

        # initialize filter's wweight and bias
        self.filter_list = [self.Wh1_u, self.Wh2_u, self.Wh3_u, self.Wh4_u, self.Wh1_v, self.Wh2_v, self.Wh3_v,
                            self.Wh4_v]
        self.init_filter(self.filter_list, c=0.01)


    def init_filter(self, filter_list, c):
        '''
				:param filter_list: list of filter for initialization
				:param c: constant multiplied on Xavier initialization
				'''
        for filter in filter_list:
            # Xavier initialization and then scale
            torch.nn.init.xavier_uniform_(filter.weight)
            filter.weight.data = c * filter.weight.data
            # filter.weight.data.uniform_(-c * np.sqrt(1 / (5 * 5 * 16)), c * np.sqrt(1 / (5 * 5 * 16)))
            if filter.bias is not None:
                filter.bias.data.fill_(0.0)


    def forward(self, h):

        h_pad = torch.cat((    h[:, :, :, :, -2:],     h,     h[:, :, :, :, 0:2]), dim=4)
        h_pad = torch.cat((h_pad[:, :, :, -2:, :], h_pad, h_pad[:, :, :, 0:2, :]), dim=3)
        h_pad = torch.cat((h_pad[:, :, -2:, :, :], h_pad, h_pad[:, :, 0:2, :, :]), dim=2)
        u_pad = h_pad[:, 0:1, ...]  # (N+4)x(N+4)
        v_pad = h_pad[:, 1:2, ...]
        u_prev = h[:, 0:1, ...]     # NxN
        v_prev = h[:, 1:2, ...]

        u_res = self.mu_up*torch.sigmoid(self.CA)*self.W_laplace(u_pad) + self.Wh4_u( self.Wh1_u(h)*self.Wh2_u(h)*self.Wh3_u(h) )
        v_res = self.mu_up*torch.sigmoid(self.CB)*self.W_laplace(v_pad) + self.Wh4_v( self.Wh1_v(h)*self.Wh2_v(h)*self.Wh3_v(h) )
        u_next = u_prev + u_res * self.dt
        v_next = v_prev + v_res * self.dt
        ch = torch.cat((u_next, v_next), dim=1)

        return ch, ch


    def init_hidden_tensor(self, prev_state):
        ''' 
            Initial hidden state with h from previous batch
            shape: [batch, channel, height, width]
        '''
        # self.init_state = torch.nn.Parameter(torch.tensor(prev_state, dtype=torch.float32).cuda(), requires_grad=True)
        return prev_state.cuda()


class RCNN(nn.Module):

    ''' Recurrent convolutional neural network layer '''

    def __init__(self, input_channels, hidden_channels, init_state_low, input_kernel_size,
                       output_channels=1, step=1, effective_step=None):


        super(RCNN, self).__init__()
        
        # input channels of layer includes input_channels and hidden_channels of cells 
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.output_channels = 1    # no use, always 1
        self.input_kernel_size = input_kernel_size
        self.step = step
        self.effective_step = effective_step
        self._all_layers = []
        self.init_state_low = init_state_low
        self.init_state = []

        # Upconv as initial state generator
        self.UpconvBlock = upscaler().cuda()

        name = 'crnn_cell'
        cell = RCNNCell(
                        input_channels = self.input_channels,
                        hidden_channels = self.hidden_channels,
                        input_kernel_size = self.input_kernel_size,
                        )

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
                # last output is a dummy for central FD
                second_last_state = internal_state.clone()

            # after many layers output the result save at time step t
            if step in self.effective_step:
                outputs.append(o)

        return outputs, second_last_state


class Conv3dDerivative(nn.Module):
    def __init__(self, DerFilter, deno, kernel_size=5, name=''):
        '''
        :param DerFilter: constructed derivative filter, e.g. Laplace filter
        :param deno: resolution of the filter, used to divide the output, e.g. c*dt, c*dx or c*dx^2
        :param kernel_size:
        :param name: optional name for the operator
        '''
        super(Conv3dDerivative, self).__init__()
        self.deno = deno  # constant in the finite difference
        self.name = name
        self.input_channels = 1
        self.output_channels = 1
        self.kernel_size = kernel_size

        self.filter = nn.Conv3d(self.input_channels, self.output_channels, self.kernel_size,
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

    def __init__(self, dt = (0.5), dx = (100/48)):

        self.dt = dt
        self.dx = dx
       
        super(loss_generator, self).__init__()

        # spatial derivative operator
        self.laplace = Conv3dDerivative(DerFilter = laplace_3d,
                                        deno = (dx**2),
                                        kernel_size = 5,
                                        name = 'laplace_operator').cuda()

        # temporal derivative operator
        self.dt = Conv1dDerivative(DerFilter = [[[-1, 1, 0]]],
                                   deno = (dt*1),
                                   kernel_size = 3,
                                   name = 'partial_t').cuda()


    def get_phy_Loss(self, output):  # Staggered version
        # spatial derivatives
        laplace_u = self.laplace(output[0:-2, 0:1, :, :, :])
        laplace_v = self.laplace(output[0:-2, 1:2, :, :, :])

        # temporal derivatives - u
        u = output[:, 0:1, 2:-2, 2:-2, 2:-2]
        lent = u.shape[0]
        lenx = u.shape[3]
        leny = u.shape[2]
        lenz = u.shape[4]
        u_conv1d = u.permute(2, 3, 4, 1, 0)  # [height(Y), width(X), depth, c, step]
        u_conv1d = u_conv1d.reshape(lenx*leny*lenz, 1, lent)
        u_t = self.dt(u_conv1d)  # lent-2 due to no-padding
        u_t = u_t.reshape(leny, lenx, lenz, 1, lent-2)
        u_t = u_t.permute(4, 3, 0, 1, 2)  # [step-2, c, height(Y), width(X)]

        # temporal derivatives - v
        v = output[:, 1:2, 2:-2, 2:-2, 2:-2]
        v_conv1d = v.permute(2, 3, 4, 1, 0)  # [height(Y), width(X), c, step]
        v_conv1d = v_conv1d.reshape(lenx*leny*lenz, 1, lent)
        v_t = self.dt(v_conv1d)  # lent-2 due to no-padding
        v_t = v_t.reshape(leny, lenx, lenz, 1, lent-2)
        v_t = v_t.permute(4, 3, 0, 1, 2)  # [step-2, c, height(Y), width(X)]

        u = output[0:-2, 0:1, 2:-2, 2:-2, 2:-2]  # [step, c, height(Y), width(X), depth]
        v = output[0:-2, 1:2, 2:-2, 2:-2, 2:-2]  # [step, c, height(Y), width(X)]

        # GS eqn
        Du = 0.2
        Dv = 0.1
        f = 0.025
        k = 0.055
        # compute residual
        f_u = (Du*laplace_u - u*v**2 + f*(1-u) - u_t)
        f_v = (Dv*laplace_v + u*v**2 - (f+k)*v - v_t)
        return f_u, f_v

def get_ic_loss(model):
    Upconv = model.UpconvBlock
    init_state_low = model.init_state_low
    init_state_bicubic = F.interpolate(init_state_low, (48, 48, 48), mode='trilinear')
    mse_loss = nn.MSELoss()
    init_state_pred = Upconv(init_state_low)
    loss_ic = mse_loss(init_state_pred, init_state_bicubic)
    return loss_ic

def loss_func(output, loss_generator):
    '''calculate the phycis loss'''
    # Padding x axis due to periodic boundary condition
    output = torch.cat((output[:, :, :, :, -2:], output, output[:, :, :, :, 0:3]), dim=4)
    output = torch.cat((output[:, :, :, -2:, :], output, output[:, :, :, 0:3, :]), dim=3)
    output = torch.cat((output[:, :, -2:, :, :], output, output[:, :, 0:3, :, :]), dim=2)
    # get physics loss
    mse_loss = nn.MSELoss()
    f_u, f_v = loss_generator.get_phy_Loss(output)
    loss = mse_loss(f_u, torch.zeros_like(f_u).cuda()) \
         + mse_loss(f_v, torch.zeros_like(f_v).cuda())
    return loss


def pretrain_upscaler(Upconv, init_state_low, epoch=10000):
    '''
    :param Upconv: upscalar model
    :param init_state_low: low resolution measurement
    :return:
    '''
    init_state_trilinear = F.interpolate(init_state_low, (48, 48, 48), mode='trilinear')
    optimizer = optim.Adam(Upconv.parameters(), lr = 0.02)
    scheduler = StepLR(optimizer, step_size=100, gamma=0.98)
    mse_loss = nn.MSELoss()
    for epoch in range(epoch):
        optimizer.zero_grad()
        init_state_pred = Upconv(init_state_low)
        loss = mse_loss(init_state_pred, init_state_trilinear)
        loss.backward(retain_graph=True)
        print('[%d] loss: %.9f' % ((epoch+1), loss.item()))
        optimizer.step()
        scheduler.step()

def load_model(model):
    # Load model and optimizer state
    checkpoint = torch.load('./model/checkpoint.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=250, gamma=0.975)
    return model, optimizer, scheduler

def train(model, truth, n_iters, learning_rate, dt, dx, cont=True): # cont=False
    # define some parameters
    train_loss_list = []
    # model
    if cont:
        model, optimizer, scheduler = load_model(model)
    else:
        optimizer = optim.Adam(model.parameters(), lr = learning_rate)
        scheduler = StepLR(optimizer, step_size=250, gamma=0.975)
    # for param_group in optimizer.param_groups:
    #     #     param_group['lr']=0.001
    loss_gen = loss_generator(dt, dx)
    for epoch in range(n_iters):
        optimizer.zero_grad()
        num_time_batch = 1
        batch_loss, phy_loss, ic_loss, data_loss = [0]*4
        for time_batch_id in range(num_time_batch):
            # update the first input for each time batch
            if time_batch_id == 0:
                pass
            else:
                pass
            # output is a list
            output, second_last_state = model()
            output = torch.cat(tuple(output), dim=0)
            mse_loss = nn.MSELoss()
            # 21x2x48x48x48
            loss_data = mse_loss(output[:-1:15, :, ::2, ::2, ::2], truth[::15, :, ::2, ::2, ::2].cuda())
            loss_ic   = get_ic_loss(model)
            # get physics loss (for validation, not used for training)
            loss_phy = loss_func(output, loss_gen)
            loss = 10*loss_data + 5.0*loss_ic
            loss.backward(retain_graph=True)
            batch_loss += loss.item()
            phy_loss, ic_loss, data_loss = loss_phy.item(), loss_ic.item(), loss_data.item()
            # phy_loss, ic_loss, data_loss = 0, loss_ic.item(), loss_data.item()
        optimizer.step()
        scheduler.step()
        # print loss in each epoch
        print('[%d/%d %d%%] loss: %.11f, ic_loss: %.11f, data_loss: %.11f, phy_loss: %.11f' % ((epoch+1), n_iters, ((epoch+1)/n_iters*100.0),
            batch_loss, ic_loss, data_loss, phy_loss))
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
        ## If Overflow (NaN) happens, restore optimizer to the last checkpoint
        if torch.isnan(loss_phy).any():
            model, optimizer, scheduler = load_model(model)
            for param_group in optimizer.param_groups:
                lr = param_group['lr']*0.9
                param_group['lr'] = lr
            print('Restore to the last checkpoint!!')
            print('LR reset to be ', lr)

    return train_loss_list

def save_model(model, model_name, save_path):
    ''' save the model '''
    torch.save(model.state_dict(), save_path + model_name + '.pt')

def get_expression(model):
    u, v = Symbol('u'), Symbol('v')
    U = Matrix([u, v])
    W1_u, B1_u = model.crnn_cell.Wh1_u.weight.data.cpu().numpy(), model.crnn_cell.Wh1_u.bias.data.cpu().numpy()
    W1_u, B1_u = Matrix(W1_u[:, :, 0, 0, 0]), Matrix(B1_u)
    W2_u, B2_u = model.crnn_cell.Wh2_u.weight.data.cpu().numpy(), model.crnn_cell.Wh2_u.bias.data.cpu().numpy()
    W2_u, B2_u = Matrix(W2_u[:, :, 0, 0, 0]), Matrix(B2_u)
    W3_u, B3_u = model.crnn_cell.Wh3_u.weight.data.cpu().numpy(), model.crnn_cell.Wh3_u.bias.data.cpu().numpy()
    W3_u, B3_u = Matrix(W3_u[:, :, 0, 0, 0]), Matrix(B3_u)
    W4_u, B4_u = model.crnn_cell.Wh4_u.weight.data.cpu().numpy(), model.crnn_cell.Wh4_u.bias.data.cpu().numpy()
    W4_u, B4_u = Matrix(W4_u[:, :, 0, 0, 0]), Matrix(B4_u)
    u_mat_4x1 = ((W1_u*U+B1_u).multiply_elementwise(W2_u*U+B2_u)).multiply_elementwise(W3_u*U+B3_u)
    u_term = W4_u*u_mat_4x1 + B4_u
    u_term.simplify()
    print('u_term: ', u_term)
    W1_v, B1_v = model.crnn_cell.Wh1_v.weight.data.cpu().numpy(), model.crnn_cell.Wh1_v.bias.data.cpu().numpy()
    W1_v, B1_v = Matrix(W1_v[:, :, 0, 0, 0]), Matrix(B1_v)
    W2_v, B2_v = model.crnn_cell.Wh2_v.weight.data.cpu().numpy(), model.crnn_cell.Wh2_v.bias.data.cpu().numpy()
    W2_v, B2_v = Matrix(W2_v[:, :, 0, 0, 0]), Matrix(B2_v)
    W3_v, B3_v = model.crnn_cell.Wh3_v.weight.data.cpu().numpy(), model.crnn_cell.Wh3_v.bias.data.cpu().numpy()
    W3_v, B3_v = Matrix(W3_v[:, :, 0, 0, 0]), Matrix(B3_v)
    W4_v, B4_v = model.crnn_cell.Wh4_v.weight.data.cpu().numpy(), model.crnn_cell.Wh4_v.bias.data.cpu().numpy()
    W4_v, B4_v = Matrix(W4_v[:, :, 0, 0, 0]), Matrix(B4_v)
    v_mat_4x1 = ((W1_v*U+B1_v).multiply_elementwise(W2_v*U+B2_v)).multiply_elementwise(W3_v*U+B3_v)
    v_term = W4_v*v_mat_4x1 + B4_v
    v_term.simplify()
    print('v_term: ', v_term)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def summary_parameters(model):
    for i in model.parameters():
        print(i.shape)

def add_noise(truth, pec=0.05):  # TxCx50X50X50
    from torch.distributions import normal
    assert truth.shape[1] == 2
    uv = [truth[:, 0:1, :, :, :], truth[:, 1:2, :, :, :]]
    uv_noi = []
    torch.manual_seed(66)
    for truth in uv:
        n_distr = normal.Normal(0.0, 1.0)
        R = n_distr.sample(truth.shape)
        std_R = torch.std(R)          # std of samples
        std_T = torch.std(truth)
        noise = R*std_T/std_R*pec
        uv_noi.append(truth + noise)
    return torch.cat(uv_noi, dim=1)


if __name__ == '__main__':

    ################# prepare the input dataset ####################
    time_steps = 300    # 150->300 for training, 1000 for inference
    dt = 0.5
    dx = 100/48

    ################### define the Initial conditions ####################
    import scipy.io as sio
    data = sio.loadmat('./data/3DRD_2x3001x48x48x48_[dt=0.5].mat')['uv']
    UV = np.transpose(data, (1, 0, 2, 3, 4)).astype(np.float32)
    truth_clean = torch.from_numpy(UV)
    # Add noise 10%
    UV = add_noise(torch.tensor(UV), pec=0.1)
    IC = UV[1000:1001, :, :, :, :]                                            # 1x2x100x100

    # get the ground truth (noisy) for training
    truth = UV[1000:1001+time_steps]

    ################# build the model #####################
    # define the mdel hyperparameters
    time_batch_size = time_steps
    steps = time_batch_size + 1
    # effective_step = [i-1 for i in range(10, steps, 10)]
    effective_step = list(range(0, steps))
    n_iters = 12000             # 20000
    learning_rate = 2e-3
    save_path = './model/'

    # Low-res initial condition
    U0_low = IC[0, 0, ::2, ::2, ::2]
    V0_low = IC[0, 1, ::2, ::2, ::2]
    h0 = torch.cat((U0_low[None, None, ...], V0_low[None, None, ...]), dim=1)
    init_state_low = h0.cuda()

    model = RCNN(
                input_channels = 2,
                hidden_channels = 2,   # hidden channel 2 or 4 because 3D model is too computationally expensive
                output_channels = 2,
                init_state_low = init_state_low,
                input_kernel_size = 5,
                step = steps,
                effective_step = effective_step).cuda()

    # train the model
    start = time.time()
    cont = True   # to train from scratch, set cont=False
    if not cont:
        pretrain_upscaler(model.UpconvBlock, init_state_low)
    train_loss_list = train(model, truth, n_iters, learning_rate, dt, dx, cont=cont)
    end = time.time()

    print('The training time is: ', (end-start))


    # Do the forward inference
    with torch.no_grad():
        output, _ = model()
    output = torch.cat(tuple(output), dim=0)
    
    # post-process
    import scipy
    UV = output.detach().cpu().numpy()
    UV = np.transpose(UV, (1, 0, 2, 3, 4))
    scipy.io.savemat('uv_2x31x48x48x48_[PeRCNN].mat', {'uv': UV[:, :-1:10]})

    # Save the data and use Plot3D to plot the isosurface







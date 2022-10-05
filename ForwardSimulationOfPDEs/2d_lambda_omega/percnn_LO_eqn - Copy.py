import os
import time
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
torch.set_default_dtype(torch.float64)

torch.manual_seed(66)
np.random.seed(66)

# Discrete Laplacian operator (2D)
lapl_op = [[[[    0,   0, -1/12,   0,     0],
             [    0,   0,   4/3,   0,     0],
             [-1/12, 4/3,    -5, 4/3, -1/12],
             [    0,   0,   4/3,   0,     0],
             [    0,   0, -1/12,   0,     0]]]]

class RCNNCell(nn.Module):
    ''' Recurrent Convolutional NN Cell '''

    def __init__(self, input_kernel_size=1, input_stride=1, input_padding=0):
        

        super(RCNNCell, self).__init__()

        # Initial parameters
        self.input_kernel_size = input_kernel_size
        self.input_stride = input_stride
        self.input_padding = input_padding

        # Discretization parameter
        self.dx = 0.2           # 20.0/100
        self.dt = 0.0125        # 10/800

        # Diffusion coefficient (assumed known)
        self.DA = torch.nn.Parameter(torch.tensor(0.2, dtype=torch.float64), requires_grad=True)
        self.DB = torch.nn.Parameter(torch.tensor(0.2, dtype=torch.float64), requires_grad=True)

        # Conv2d operator as Laplace operator
        self.W_laplace = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=5,
                                   stride=self.input_stride, padding=self.input_padding, bias=False)
        self.W_laplace.weight.data = torch.tensor(lapl_op, dtype=torch.float64)/self.dx**2
        self.W_laplace.weight.requires_grad=False

        # Parallel Conv2d Layers (three layers)
        # Nonlinear term for u (up to 3rd order)
        self.Wh1_u = nn.Conv2d(in_channels=2, out_channels=4, kernel_size=self.input_kernel_size,
                               stride=self.input_stride, padding=0, bias=True, )
        self.Wh2_u = nn.Conv2d(in_channels=2, out_channels=4, kernel_size=self.input_kernel_size,
                               stride=self.input_stride, padding=0, bias=True, )
        self.Wh3_u = nn.Conv2d(in_channels=2, out_channels=4, kernel_size=self.input_kernel_size,
                               stride=self.input_stride, padding=0, bias=True, )
        # Aggregate conv layer 1x1
        self.Wh4_u = nn.Conv2d(in_channels=4, out_channels=1, kernel_size=1, stride=1, padding=0, bias=True)

        # Nonlinear term for v (up to 3rd order)
        self.Wh1_v = nn.Conv2d(in_channels=2, out_channels=4, kernel_size=self.input_kernel_size,
                               stride=self.input_stride, padding=0, bias=True, )
        self.Wh2_v = nn.Conv2d(in_channels=2, out_channels=4, kernel_size=self.input_kernel_size,
                               stride=self.input_stride, padding=0, bias=True, )
        self.Wh3_v = nn.Conv2d(in_channels=2, out_channels=4, kernel_size=self.input_kernel_size,
                               stride=self.input_stride, padding=0, bias=True, )
        self.Wh4_v = nn.Conv2d(in_channels=4, out_channels=1, kernel_size=1, stride=1, padding=0, bias=True)

        # initialize filter's weight and bias
        self.filter_list = [self.Wh1_u, self.Wh2_u, self.Wh3_u, self.Wh4_u, self.Wh1_v, self.Wh2_v, self.Wh3_v, self.Wh4_v]
        self.init_filter(self.filter_list, c=0.5)

        # # Initialize weight to avoid overflow
        # c = 1
        # self.Wh1_u.weight.data.uniform_(-c*np.sqrt(1 / (3 * 3 * 320)), c*np.sqrt(1 / (3 * 3 * 320)))
        # self.Wh2_u.weight.data.uniform_(-c*np.sqrt(1 / (3 * 3 * 320)), c*np.sqrt(1 / (3 * 3 * 320)))
        # self.Wh3_u.weight.data.uniform_(-c*np.sqrt(1 / (3 * 3 * 320)), c*np.sqrt(1 / (3 * 3 * 320)))
        # self.Wh4_u.weight.data.uniform_(-c*np.sqrt(1 / (3 * 3 * 3)), c*np.sqrt(1 / (3 * 3 * 3)))
        # self.Wh1_v.weight.data.uniform_(-c*np.sqrt(1 / (3 * 3 * 320)), c*np.sqrt(1 / (3 * 3 * 320)))
        # self.Wh2_v.weight.data.uniform_(-c*np.sqrt(1 / (3 * 3 * 320)), c*np.sqrt(1 / (3 * 3 * 320)))
        # self.Wh3_v.weight.data.uniform_(-c*np.sqrt(1 / (3 * 3 * 320)), c*np.sqrt(1 / (3 * 3 * 320)))
        # self.Wh4_v.weight.data.uniform_(-c*np.sqrt(1 / (3 * 3 * 3)), c*np.sqrt(1 / (3 * 3 * 3)))

    def init_filter(self, filter_list, c):
        '''
        :param filter_list: list of filter for initialization
        :param c: constant multiplied on Xavier initialization
        '''
        for filter in filter_list:
            filter.weight.data.uniform_(-c * np.sqrt(1 / np.prod(filter.weight.shape[:-1])),
                                         c * np.sqrt(1 / np.prod(filter.weight.shape[:-1])))
            if filter.bias is not None:
                filter.bias.data.fill_(0.0)


    def forward(self, h):

        h_pad = torch.cat((    h[:, :, :, -2:],     h,     h[:, :, :, 0:2]), dim=3)
        h_pad = torch.cat((h_pad[:, :, -2:, :], h_pad, h_pad[:, :, 0:2, :]), dim=2)
        u_pad = h_pad[:, 0:1, ...]  # 104x104
        v_pad = h_pad[:, 1:2, ...]
        u_prev = h[:, 0:1, ...]     # 100x100
        v_prev = h[:, 1:2, ...]

        u_res = self.DA*self.W_laplace(u_pad) + self.Wh4_u( self.Wh1_u(h)*self.Wh2_u(h)*self.Wh3_u(h) )
        v_res = self.DB*self.W_laplace(v_pad) + self.Wh4_v( self.Wh1_v(h)*self.Wh2_v(h)*self.Wh3_v(h) )
        u_next = u_prev + self.dt*u_res
        v_next = v_prev + self.dt*v_res
        ch = torch.cat((u_next, v_next), dim=1)
        return ch, ch


    def init_hidden_tensor(self, prev_state):
        ''' 
            Initial hidden state with h from previous batch
            shape: [batch, channel, height, width]
        '''
        # self.init_state = torch.nn.Parameter(torch.tensor(prev_state, dtype=torch.float64).cuda(), requires_grad=True)
        return (Variable(prev_state).cuda())


class RCNN(nn.Module):

    ''' Recurrent convolutional neural network layer '''

    def __init__(self, input_kernel_size, ini_state, input_stride, input_padding, step=1, effective_step=[1]):
        
        '''        
        Args:
        -----------
        input_stride: int
            Convolution stride, only for input
            b/c we need to keep the hidden state have same dimension

        ini_state: tensor
            tensor to initialize initial state of the recurrent network

        input_padding: input
            Convolution padding, only for input
            b/c we need to keep the hidden state have same dimension
        
        step: int
            number of time steps

        effective_step: list
        '''

        super(RCNN, self).__init__()

        self.input_kernel_size = input_kernel_size
        self.input_stride = input_stride
        self.input_padding = input_padding
        self.step = step
        self.effective_step = effective_step
        self._all_layers = []
        self.init_state = torch.tensor(ini_state, dtype=torch.float64).cuda()

        name = 'rcnn_cell'
        cell = RCNNCell(input_kernel_size = self.input_kernel_size,
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
        
        ini_state: tensor, shape: [batch, channel, height, width]
            initial state
        
        Returns:
        --------
        outputs: list
            output results list, vertical output (h)
        
        second_last_state: list, length = # layers
            the second last state  
        
        '''

        # self.ini_state = ini_state
        internal_state = []
        outputs = [self.init_state]
        second_last_state = []

        for step in range(self.step):
            name = 'rcnn_cell'
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
    def __init__(self, DerFilter, resol, kernel_size=3, name=''):
        super(Conv2dDerivative, self).__init__()

        self.resol = resol  # constant in the finite difference
        self.name = name
        self.input_channels = 1
        self.output_channels = 1
        self.kernel_size = kernel_size
        assert kernel_size == len(DerFilter[0][0])

        self.filter = nn.Conv2d(self.input_channels, self.output_channels, self.kernel_size, 
                                1, padding=0, bias=False)

        # Fixed gradient operator
        self.filter.weight = nn.Parameter(torch.DoubleTensor(DerFilter), requires_grad=False)

    def forward(self, input):
        derivative = self.filter(input)
        return derivative / self.resol


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
        self.filter.weight = nn.Parameter(torch.DoubleTensor(DerFilter), requires_grad=False)

    def forward(self, input):
        derivative = self.filter(input)
        return derivative / self.resol


class loss_generator(nn.Module):
    ''' Loss generator for physics loss '''

    def __init__(self, dt = (0.0125), dx = 0.2):

        self.dt = dt
        self.dx = dx
       
        super(loss_generator, self).__init__()

        # spatial derivative operator
        self.laplace = Conv2dDerivative(DerFilter = lapl_op,
                                        resol = (dx**2),
                                        kernel_size = 5,
                                        name = 'laplace_operator').cuda()

        # temporal derivative operator
        self.dt = Conv1dDerivative(DerFilter = [[[-1, 1, 0]]],
                                   resol = (dt*1),
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

        # spatial derivatives, output [202x1x105x105]
        laplace_u = self.laplace(output[0:-2, 0:1, :, :])  # 400x1x101x101
        laplace_v = self.laplace(output[0:-2, 1:2, :, :])  # 400x1x101x101

        # temporal derivatives - u
        u = output[:, 0:1, 2:-2, 2:-2]
        lent = u.shape[0]
        lenx = u.shape[3]
        leny = u.shape[2]
        u_conv1d = u.permute(2, 3, 1, 0)  # [height(Y), width(X), c, step]
        u_conv1d = u_conv1d.reshape(lenx*leny,1,lent)
        u_t = self.dt(u_conv1d)  # lent-2 due to no-padding
        u_t = u_t.reshape(leny, lenx, 1, lent-2)
        u_t = u_t.permute(3, 2, 0, 1)  # [step-2, c, height(Y), width(X)]

        # temporal derivatives - v
        v = output[:, 1:2, 2:-2, 2:-2]
        v_conv1d = v.permute(2, 3, 1, 0)  # [height(Y), width(X), c, step]
        v_conv1d = v_conv1d.reshape(lenx*leny,1,lent)
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

        # lambda-omega eqn
        f_u = 0.1*laplace_u + (1-u**2-v**2)*u + (u**2+v**2)*v - u_t
        f_v = 0.1*laplace_v - (u**2+v**2)*u + (1-u**2-v**2)*v - v_t
        return f_u, f_v


def loss_gen(output, loss_func):
    '''calculate the phycis loss'''

    # Padding x axis due to periodic boundary condition
    # shape after: [202, 2, 105, 105]
    output = torch.cat((output[:, :, :, -2:], output, output[:, :, :, 0:3]), dim=3)
    output = torch.cat((output[:, :, -2:, :], output, output[:, :, 0:3, :]), dim=2)

    # get physics loss
    mse_loss = nn.MSELoss()
    f_u, f_v = loss_func.get_phy_Loss(output)
    loss = mse_loss(f_u, torch.zeros_like(f_u).cuda()) + mse_loss(
        f_v, torch.zeros_like(f_v).cuda())
    return loss


def train(model, init_state, n_iters, total_step, time_batch_size, learning_rate,
          dt, dx, save_path):
    # define some parameters
    train_loss_list = []
    state_detached = None
    best_loss = 1e4
    # model
    optimizer = optim.Adam(model.parameters(), lr = learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.98)
    loss_func = loss_generator(dt, dx)
    for epoch in range(n_iters):
        optimizer.zero_grad()
        # get the number of batches
        num_time_batch = int(total_step/time_batch_size)
        batch_loss = 0
        for time_batch_id in range(num_time_batch):
            # update the first input for each time batch
            if time_batch_id == 0:
                model.init_state = torch.tensor(init_state, dtype=torch.float64).cuda()
            else:
                model.init_state = torch.tensor(state_detached, dtype=torch.float64).cuda()
            # output is a list
            output, second_last_state = model()
            output = torch.cat(tuple(output), dim=0)
            loss_phy = loss_gen(output, loss_func)
            loss = loss_phy
            loss.backward(retain_graph=True)
            batch_loss += loss.item()
            # update the state and output for next batch
            state_detached = second_last_state.detach()
            print('[%d/%d %d%%] batch num %d, batch loss: %.15f, ' % ((epoch + 1), n_iters, ((epoch + 1) / n_iters * 100.0), time_batch_id, loss.item()))
        optimizer.step()
        scheduler.step()
        # print loss in each epoch
        print('__'*30)
        print('[%d/%d %d%%] Epoch loss: %.15f, ' % ((epoch+1), n_iters, ((epoch+1)/n_iters*100.0), batch_loss))
        for param_group in optimizer.param_groups:
            print('LR: ', param_group['lr'])
        print('__'*30)
        train_loss_list.append(batch_loss)
        # save model
        if (epoch+1)%100 == 0:
            print('save model!!!')
            save_model(model, 'rcnn_pde', save_path)
    return train_loss_list


def save_model(model, model_name, save_path):
    ''' save the model '''
    torch.save(model.state_dict(), save_path + model_name + '.pt')


def load_model(model, model_name, save_path):
    ''' load the model '''
    model.load_state_dict(torch.load(save_path + model_name + '.pt'))


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def postProcess(output, true, axis_lim, u_lim, v_lim, num, fig_save_path):
    ''' num: Number of time step
    '''
    xmin, xmax, ymin, ymax = axis_lim
    u_min, u_max = u_lim
    v_min, v_max = v_lim
    x = np.linspace(-10, 10, 101)
    y = np.linspace(-10, 10, 101)
    x_star, y_star = np.meshgrid(x, y)
    u_star = true[100+num, 0,  ...]
    u_pred = output[num, 0, :, :].detach().cpu().numpy()
    #
    v_star = true[100+num, 1, ...]
    v_pred = output[num, 1, :, :].detach().cpu().numpy()
    #
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(7, 7))
    fig.subplots_adjust(hspace=0.3, wspace=0.3)
    #
    cf = ax[0, 0].scatter(x_star, y_star, c=u_pred, alpha=0.99, edgecolors='none', cmap='hot', marker='s', s=2.5, vmin=u_min, vmax=u_max)
    ax[0, 0].axis('square')
    ax[0, 0].set_xlim([xmin, xmax])
    ax[0, 0].set_ylim([ymin, ymax])
    # cf.cmap.set_under('whitesmoke')
    # cf.cmap.set_over('black')
    ax[0, 0].set_xticks([])
    ax[0, 0].set_yticks([])
    ax[0, 0].set_title('u-RCNN')
    fig.colorbar(cf, ax=ax[0, 0], fraction=0.046, pad=0.04)
    #
    cf = ax[0, 1].scatter(x_star, y_star, c=u_star, alpha=0.99, edgecolors='none', cmap='hot', marker='s', s=2.5, vmin=u_min, vmax=u_max)
    ax[0, 1].axis('square')
    ax[0, 1].set_xlim([xmin, xmax])
    ax[0, 1].set_ylim([ymin, ymax])
    # cf.cmap.set_under('whitesmoke')
    # cf.cmap.set_over('black')
    ax[0, 1].set_xticks([])
    ax[0, 1].set_yticks([])
    ax[0, 1].set_title('u-Ref.')
    fig.colorbar(cf, ax=ax[0, 1], fraction=0.046, pad=0.04)
    #
    cf = ax[1, 0].scatter(x_star, y_star, c=v_pred, alpha=0.99, edgecolors='none', cmap='hot', marker='s', s=2.5, vmin=v_min, vmax=v_max)
    ax[1, 0].axis('square')
    ax[1, 0].set_xlim([xmin, xmax])
    ax[1, 0].set_ylim([ymin, ymax])
    # cf.cmap.set_under('whitesmoke')
    # cf.cmap.set_over('black')
    ax[1, 0].set_xticks([])
    ax[1, 0].set_yticks([])
    ax[1, 0].set_title('v-RCNN')
    fig.colorbar(cf, ax=ax[1, 0], fraction=0.046, pad=0.04)
    #
    cf = ax[1, 1].scatter(x_star, y_star, c=v_star, alpha=0.99, edgecolors='none', cmap='hot', marker='s', s=2.5, vmin=v_min, vmax=v_max)
    ax[1, 1].axis('square')
    ax[1, 1].set_xlim([xmin, xmax])
    ax[1, 1].set_ylim([ymin, ymax])
    # cf.cmap.set_under('whitesmoke')
    # cf.cmap.set_over('black')
    ax[1, 1].set_title('v-Ref.')
    fig.colorbar(cf, ax=ax[1, 1], fraction=0.046, pad=0.04)
    #
    # plt.draw()
    plt.savefig(fig_save_path + 'uv_comparison_'+str(num).zfill(3)+'.png')
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
    dt = 10.0/800
    dx = 20.0/100
    dy = 20.0/100

    ################### define the Initial conditions ####################
    UV = sio.loadmat('./data/uv_2x1602x100x100_Euler_[dt=0.0125,HighOrderLap].mat')['uv']
    U0 = UV[0:1, 100:101, :, :]
    V0 = UV[1:2, 100:101, :, :]
    h0 = torch.cat((torch.tensor(U0), torch.tensor(V0)), dim=1)
    init_state = h0.cuda()

    ################# build the model #####################
    # define the model hyper-parameters
    total_step = 800
    time_batch_size = 400
    steps = time_batch_size + 1
    effective_step = list(range(0, steps))
    learning_rate = 1e-2
    n_iters = 1000
    save_path = './model/'

    model = RCNN(
        input_kernel_size = 1,
        ini_state = init_state,
        input_stride = 1,
        input_padding = 0,
        step = steps, 
        effective_step = effective_step).cuda()

    # Option 1: load the model from checkpoint
    # load_model(model, 'rcnn_pde', save_path)

    # get the ground truth for plotting
    # [2, 1602, 100, 100]
    truth = UV
    # [1602, 2, 100, 100]
    truth = np.transpose(truth, (1, 0, 2, 3))
    truth = torch.tensor(truth, dtype=torch.float64)

    # Option 2: train the model from scratch
    start = time.time()
    train_loss_list = train(model, init_state, n_iters, total_step, time_batch_size, learning_rate, dt, dx, save_path)
    end = time.time()

    save_model(model, 'rcnn_pde', save_path)

    print('The training time is: ', (end-start))


    ## Do the inference
    hist = []
    state_detached = None
    num_time_batch = int(total_step/time_batch_size)
    for time_batch_id in range(num_time_batch):
        # update the first input for each time batch
        if time_batch_id == 0:
            model.init_state = torch.tensor(init_state, dtype=torch.float64).cuda()
        else:
            model.init_state = torch.tensor(state_detached, dtype=torch.float64).cuda()
        # output is a list
        output, second_last_state = model()
        if time_batch_id == (num_time_batch-1):
            output = output[:-1]
        else:
            output = output[:-2]
        output = torch.cat(tuple(output), dim=0)
        print(output.shape)
        hist.append(output)
        state_detached = second_last_state.detach()
    output = torch.cat(hist, dim=0)
    # Padding x axis due to periodic boundary condition
    output = torch.cat((output, output[:, :, :, 0:1]), dim=3)
    output = torch.cat((output, output[:, :, 0:1, :]), dim=2)
    truth = torch.cat((truth, truth[:, :, :, 0:1]), dim=3)
    truth = torch.cat((truth, truth[:, :, 0:1, :]), dim=2)

    fig_save_path = './figures/'
    
    # post-process
    for i in range(0, 801, 10):
        postProcess(output, truth, axis_lim = [-10, 10, -10, 10,],
                    u_lim = [-1.0, 1.0], v_lim = [-1.0, 1.0],             # u_lim = [-1.2, 1.1], v_lim = [-0.2, 0.5],
                    num=i, fig_save_path=fig_save_path)

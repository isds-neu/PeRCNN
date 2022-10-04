import torch
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from derivatives import Loss_generator, to_numpy_float64


class STRidgeTrainer:
    def __init__(self, R0, Ut, normalize = 2, split_ratio = 0.8):

        self.R0 = R0
        self.Ut = Ut
        self.normalize = normalize
        self.split_ratio = split_ratio

        # split the training and testing data
        np.random.seed(0)
        n, d = self.R0.shape

        R = np.zeros((n, d), dtype=np.float32)
        if normalize != 0:
            Mreg = np.zeros((d, 1))
            for i in range(d):
                Mreg[i] = 1.0 / (np.linalg.norm(R0[:, i], normalize))
                R[:, i] = Mreg[i] * R0[:, i]
            normalize_inner = 0
        else:
            R = R0
            Mreg = np.ones((d, 1)) * d   # why multiply by d
            normalize_inner = 2

        self.Mreg = Mreg
        self.R = R        # R0 - raw, R - normalized
        self.normalize_inner = normalize_inner

        train_idx = []
        test_idx = []
        for i in range(n):
            if np.random.rand() < split_ratio:
                train_idx.append(i)
            else:
                test_idx.append(i)

        self.TrainR = R[train_idx, :]
        self.TestR  = R[test_idx, :]
        self.TrainY = Ut[train_idx, :]
        self.TestY  = Ut[test_idx, :]

    def train(self, maxit = 200, STR_iters = 10, lam = 0.0001, d_tol = 10, l0_penalty = None, kappa = 1):
        """
        This function trains a predictor using STRidge.

        It runs over different values of tolerance and trains predictors on a training set, then evaluates them
        using a loss function on a holdout set.

        Please note published article has typo.  Loss function used here for model selection evaluates fidelity using 2-norm,
        not squared 2-norm.
        """

        # Set up the initial tolerance and l0 penalty
        tol = d_tol

        # Get the standard least squares estimator
        w_best = np.linalg.lstsq(self.TrainR, self.TrainY)[0]
        err_f = np.mean((self.TestY - self.TestR.dot(w_best))**2)
        err_w = np.count_nonzero(w_best)

        if l0_penalty == None:
            l0_penalty = kappa*err_f

        err_best = err_f + l0_penalty*err_w

        tol_best = 0

        # Now increase tolerance until test performance decreases
        for iter in range(maxit):

            # Get a set of coefficients and error
            w = self.STRidge(self.TrainR, self.TrainY, lam, STR_iters, tol, self.Mreg, normalize = self.normalize_inner)

            err_f = np.mean((self.TestY - self.TestR.dot(w))**2)
            err_w = np.count_nonzero(w)
            err = err_f + l0_penalty*err_w

            print('__'*20)
            print('Number of iter: ', iter)
            print('Tolerence: %.7f '%(tol))
            print('Regression error: %.10f, Penalty: %.2f '% (err_f, err_w))
            print('Weight w (normalized):')
            print(w.T)

            # Has the accuracy improved?
            if err <= err_best:
                err_best = err
                w_best = w
                tol_best = tol
                tol = tol + d_tol

            else:
                tol = max([0,tol - 2*d_tol])
                d_tol = 2*d_tol / (maxit - iter)
                tol = tol + d_tol

        print('Train STRidge completed!')
        print("Optimal tolerance:", tol_best)

        return np.multiply(self.Mreg, w_best)

    def STRidge(self, X0, y, lam, maxit, tol, Mreg, normalize = 0):
        """
        Sequential Threshold Ridge Regression algorithm for finding (hopefully) sparse
        approximation to X^{-1}y.  The idea is that this may do better with correlated observables.

        This assumes y is only one column
        """

        n, d = X0.shape
        X = np.zeros((n,d), dtype=np.float64)

        # if not normalized in the outer loop, do it here
        if normalize != 0:
            Mreg = np.zeros((d, 1))
            for i in range(d):
                Mreg[i] = 1.0/(np.linalg.norm(X0[:,i], normalize))
                # Mreg[i] = 1.0 / np.max(np.abs(X0[:, i]))
                X[:,i] = Mreg[i]*X0[:,i]
        else:
            X = X0

        # Get the standard ridge esitmate
        if lam != 0:
            w = np.linalg.lstsq(X.T.dot(X) + lam*np.eye(d), X.T.dot(y))[0]
        else:
            w = np.linalg.lstsq(X,y)[0]

        num_relevant = d
        biginds = np.where( abs(w) > tol)[0]

        # Threshold and continue
        for j in range(maxit):

            # Figure out which items to cut out
            smallinds = np.where( abs(w) < tol)[0]
            new_biginds = [i for i in range(d) if i not in smallinds]

            # If nothing changes then stop
            if num_relevant == len(new_biginds): break
            else: num_relevant = len(new_biginds)

            # Also make sure we didn't just lose all the coefficients
            if len(new_biginds) == 0:
                if j == 0:
                    print('Warning: initial tolerance %.4f too large, all coefficients under threshold!'%tol)
                    # return all zero coefficient immediately
                    return w*0
                else:
                    # break and return the w in the last iteration
                    break
            biginds = new_biginds

            # Otherwise get a new guess
            w[smallinds] = 0
            if lam != 0:
                w[biginds] = np.linalg.lstsq(X[:, biginds].T.dot(X[:, biginds]) + lam*np.eye(len(biginds)),X[:, biginds].T.dot(y))[0]
            else:
                w[biginds] = np.linalg.lstsq(X[:, biginds],y)[0]

        # Now that we have the sparsity pattern, use standard least squares to get w
        if biginds != []:
            w[biginds] = np.linalg.lstsq(X[:, biginds],y)[0]

        if normalize != 0:
            return np.multiply(Mreg, w)
        else:
            return w


def gen_library():
    listA = ['ones', 'u', 'v', 'u**2', 'u*v', 'v**2', 'u**3', 'u**2*v', 'u*v**2', 'v**3']
    listB = ['ones', 'u_x', 'u_y', 'v_x', 'v_y', 'lap_u', 'lap_v']   #
    library = []
    for A in listA:
        for B in listB:
            library.append(A + '*' + B)
    return library


def gen_data_matrix(terms_dict, lib, idx):
    """
    Note: deprecated!
    :param terms_dict: dict of possible terms, e.g., {'u':u, 'v':v, 'u_x': u_x, ...}
    :param library: LIST type, library of the possible combinations of term, e.g., ['u*v', 'u_x*v', 'lap_u', ...]
    :param idx: downsampling index for the training data set
    :return: coef * lhs = rhs
    """
    # Construct the lhs and rhs
    idx = idx
    for key in terms_dict.keys():
        expression = key + '=terms_dict[\'' + str(key) + '\'][idx, :]'
        print('Execute expression:', expression)
        exec(expression)

    lhs_columns = [eval(exp) for exp in lib]
    lhs = np.concatenate(lhs_columns, axis=1)
    rhs = eval('u_t')
    return lhs, rhs



if __name__ == '__main__':

    # read data of u, v
    UV = sio.loadmat('../data/uv_2x201x100x100_[PeRCNN,41x51x51,30%noise,3layers].mat')['uv']
    UV = np.swapaxes(UV, 0, 1)
    pred_HR = torch.from_numpy(UV[50:150]).float().cuda()  # [T, 2, h, w]

    loss_generator = Loss_generator()
    mse_u, mse_v = loss_generator.get_residual_mse(pred_HR)

    terms = loss_generator.get_library(pred_HR)

    # Prepare each single terms
    terms_dict = to_numpy_float64(terms)

    # Prepare all possible combinations among terms
    lib = gen_library()

    # Prepare ground truth
    coef = np.zeros((len(lib), 1))
    u_term = {'ones*lap_v': 0.1, 'v*ones': 1.0, 'u**3*ones': -1, 'u*v**2*ones': -1,
              'u**2*v*ones': -1.0, 'v**3*ones': -1.0}

    for i, name in enumerate(lib):
        if name in u_term.keys():
            coef[i] = u_term[name]

    print(lib)

    # randomly subsample 10% of the measurements clip
    n = terms_dict['v'].shape[0]
    idx = np.random.choice(n, int(n * 0.2), replace=False)

    for key in terms_dict.keys():
        expression = key + '=terms_dict[\'' + str(key) + '\'][idx, :]'
        print('Execute expression:', expression)
        exec(expression)

    # Check the residual of ground truth (if you are interested)
    # beta, mu_u, mu_v = 1.0, 0.1, 0.1
    # f_u = u_t - (mu_u * laplace_u + (1 - u ** 2 - v ** 2) * u + beta * (u ** 2 + v ** 2) * v)
    # f_v = v_t - (mu_v * laplace_v + (1 - u ** 2 - v ** 2) * v - beta * (u ** 2 + v ** 2) * u)

    lhs_columns = [eval(exp) for exp in lib]
    lhs = np.concatenate(lhs_columns, axis=1)
    rhs = eval('v_t')

    residual = rhs - lhs@coef

    # invoke STRidge algorithm to get the sparse coefficient vector
    trainer = STRidgeTrainer(R0=lhs, Ut=rhs, normalize=2, split_ratio=0.8)

    w_best = trainer.train(maxit = 20, STR_iters = 20, lam = 0.0001, d_tol = 5, kappa = 0.1)  # kappa = 0.5 for 10% noise, 2 for below

    # get the valuation metrics, L2 error and discovery accuracy
    err_l2 = np.linalg.norm(w_best - coef, 2)/np.linalg.norm(coef, 2)
    dis_pr = np.count_nonzero(w_best * coef, 0) / np.count_nonzero(w_best, 0)
    dis_rc = np.count_nonzero(w_best*coef, 0)/np.count_nonzero(coef, 0)

    # note this is not the final result
    print('Relative L2 error: %.3f, discovery recall: %.3f, precision: %.3f'%(err_l2, dis_rc, dis_pr))

    # visualize the result
    fig, ax = plt.subplots(figsize = (18, 6))
    fig.subplots_adjust(bottom=0.2)
    ax.plot(lib, w_best, '--', linewidth=1.5, marker="*", label='Identified')
    ax.plot(lib, coef, '--', linewidth=1.5, marker="^", label='Truth')
    ax.set_xlabel('Coefficient')
    ax.set_ylabel('Value')
    plt.xticks(rotation=45)
    plt.legend()
    plt.show()

    identified_dict = {}
    for coef, term in zip(list(w_best[:, 0]), lib):
        if coef!=0:
            identified_dict[term] = coef

    print(identified_dict)

    # 0% noise  - {'ones*lap_v': 0.09455076507004463, 'v*ones': 0.987352592547758, 'u**3*ones': -0.9987499978401593, 'u**2*v*ones': -0.9945223179762985, 'u*v**2*ones': -0.9985767984520534, 'v**3*ones': -0.9928637536467936}
    # 5% noise  - {'ones*lap_v': 0.09533786376103379, 'v*ones': 0.9678307729447329, 'u**3*ones': -0.9977638809090453, 'u**2*v*ones': -0.973095915565658, 'u*v**2*ones': -1.0033666654329623, 'v**3*ones': -0.9702761170114511}
    # 10% noise - {'ones*lap_v': 0.09910130758758114, 'v*ones': 0.9499333664113959, 'u**3*ones': -1.005521937332795, 'u**2*v*ones': -0.9479578472808914, 'u*v**2*ones': -1.016075343502691, 'v**3*ones': -0.9463513812954865', u*ones': 0.006583249742472701, }
    # 20% noise - {'ones*lap_v': 0.10013417212610601, 'v*ones': 0.9254492768821463, 'u**3*ones': -0.9473383518483611, 'u**2*v*ones': -0.9186026576606379, 'u*v**2*ones': -0.960085361324526, 'v**3*ones': -0.9157825834961745, 'u*ones': -0.0437554887605747}
    # 30% noise - {'ones*lap_v': 0.10200562180569925, 'v*ones': 0.8114827539411924, 'u**3*ones': -0.7885664073006875, 'u**2*v*ones': -0.7848834527036386, 'u*v**2*ones': -0.8137552664403047, 'v**3*ones': -0.7858290033263355, 'u*ones': -0.1862358951990892}

    # Pareto Front analysis for sparse regression (L0 penalty weighting coefficient)
    list_gamma = [0.01*1.2**i for i in range(45)]
    list_LS_loss, list_L0_loss, list_w = [], [], []
    for i, gamma in enumerate(list_gamma):
        print('{i}th iteration'.format(i=i))
        w_best = trainer.train(maxit = 20, STR_iters = 20, lam = 0.0001, d_tol = 5, kappa = gamma)
        list_w.append(w_best)
        list_LS_loss.append(np.mean((rhs - lhs.dot(w_best)) ** 2))
        list_L0_loss.append(np.count_nonzero(w_best))

    with open('pareto_analysis_v.npy', 'wb') as f:
        np.save(f, [list_gamma, list_LS_loss, list_L0_loss])
    with open('./pareto_analysis_v.npy', 'rb') as f:
        list_l0_penalty, list_LS_loss, list_L0_loss = np.load(f)


    from matplotlib import rc
    rc('text', usetex=True)
    rc('legend', fontsize=14)


    fig, ax1 = plt.subplots(figsize=(7.0, 4.5))
    fig.subplots_adjust(bottom=0.2)
    ax1.plot(list_gamma, list_LS_loss, marker='o', markersize=4, label=r'$\mathrm{Regression\ error}$', color='dodgerblue')
    # ax1.spines['left'].set_color('dodgerblue')
    # ax1.yaxis.label.set_color('dodgerblue')
    ax1.set_xlabel(r'$\gamma$', color='black', fontsize=17)
    ax1.set_xscale('log')
    ax1.tick_params(axis='y', labelsize=15, direction='in', colors='dodgerblue')
    ax1.tick_params(axis='x', labelsize=15, direction='in', colors='black')
    # ax1.tick_params(labelsize=15, direction='in', which='both', colors='black')
    ax1.set_ylabel(r'$||\mathbf{\hat{\Phi}\Xi-\hat{Z}}||_2$', color='dodgerblue', fontsize=15)
    ax1.plot([], [], linewidth=1.5, color='orangered', marker='o', markersize=4, label=r'$\ell_0\mathrm{\ penalty}$', )
    legend = plt.legend(frameon=True, ncol=1, fontsize=15, loc='upper center')
    # legend.get_frame().set_alpha(0.2)
    ax2 = ax1.twinx()
    ax2.plot(list_gamma, list_L0_loss, linewidth=1.5, color='orangered', marker='o', markersize=4, label=r'$\mathrm{L0\ penalty}$')
    ax2.spines['right'].set_color('red')
    # ax2.set_yticks([1, 3, 5, 7, 9])
    ax2.tick_params(labelsize=15, direction='in', which='both', colors='orangered')
    ax2.set_ylabel(r'$\mathbf{||\Xi||}_0$', color='orangered', fontsize=15)
    plt.savefig('./select_gamma_v.png', dpi=300)
    plt.show()

    fig, ax1 = plt.subplots(figsize=(6.5, 4.5))
    fig.subplots_adjust(bottom=0.2, left=0.2)
    ax1.plot(list_L0_loss, list_LS_loss, marker='o', label=r'$\mathrm{Fitting\ error}$', color='black')
    ax1.set_xlabel(r'$\mathbf{||\Xi||}_0$', color='black', fontsize=15)
    ax1.set_ylabel(r'$tol$', color='black', fontsize=17)
    # ax1.set_xscale('log')
    # ax1.set_yscale('log')
    ax1.tick_params(labelsize=15, direction='in', which='both', colors='black')
    ax1.set_ylabel(r'$||\mathbf{\hat{\Phi}\Xi-\hat{Z}}||_2$', color='black', fontsize=15)
    plt.savefig('./pareto_v.png', dpi=300)
    plt.show()

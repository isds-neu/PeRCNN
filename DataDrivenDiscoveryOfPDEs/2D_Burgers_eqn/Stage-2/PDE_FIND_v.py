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

    def train(self, maxit = 200, STR_iters = 10, lam = 0.0001, d_tol = 10, l0_penalty = None, kappa = 1.0, must_have = 6):
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
            w = self.STRidge(self.TrainR, self.TrainY, lam, STR_iters, tol, self.Mreg, normalize = self.normalize_inner, must_have=must_have)

            err_f = np.mean((self.TestY - self.TestR.dot(w))**2)
            err_w = np.count_nonzero(w)
            err_must_have = 50.0 if abs(w[must_have])<1e-10 else 0.0
            err = err_f + l0_penalty*err_w  # + err_must_have

            print('__'*20)
            print('Number of iter: ', iter)
            print('Tolerence: %.7f '%(tol))
            print('Regression error: %.10f, Penalty: %.2f , Must-have penalty %.2f'% (err_f, err_w, err_must_have))
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

    def STRidge(self, X0, y, lam, maxit, tol, Mreg, normalize = 0, must_have=6):
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

            if must_have not in new_biginds:
                new_biginds.append(must_have)
                new_biginds.sort()

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
    # listA = ['ones', 'u', 'v', 'u**2', 'u*v', 'v**2']   #
    listA = ['ones', 'u', 'v', 'u**2', 'u*v', 'v**2', 'u**3', 'u**2*v', 'u*v**2', 'v**3']
    listB = ['ones', 'u_x', 'u_y', 'v_x', 'v_y', 'lap_u', 'lap_v']
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
    UV = sio.loadmat('../data/uv_2x201x100x100_[PeRCNN,41x51x51,20%noise,3layers].mat')['uv']
    UV = np.swapaxes(UV, 0, 1)
    pred_HR = torch.from_numpy(UV[50:150]).float().cuda()  # [T, 2, h, w]

    loss_generator = Loss_generator()
    mse_u, mse_v = loss_generator.get_residual_mse(pred_HR)

    terms = loss_generator.get_phy_residual(pred_HR)

    # Prepare each single terms
    terms_dict = to_numpy_float64(terms)

    # Prepare all possible combinations among terms
    lib = gen_library()

    # Prepare ground truth
    coef = np.zeros((len(lib), 1))
    for i, name in enumerate(lib):
        if name == 'ones*lap_v':
            coef[i] = 0.005
        if name == 'v*v_y' or name == 'u*v_x':
            coef[i] = -1

    print(lib)

    # randomly subsample 10% of the measurements clip
    n = terms_dict['v'].shape[0]
    idx = np.random.choice(n, int(n * 0.2), replace=False)

    for key in terms_dict.keys():
        expression = key + '=terms_dict[\'' + str(key) + '\'][idx, :]'
        print('Execute expression:', expression)
        exec(expression)

    # Check the residual of ground truth (if you are interested)
    # f_u = (u_t - nu * lap_u + u * u_x + v * u_y)
    # f_v = (v_t - nu * lap_v + u * v_x + v * v_y)
    lhs_columns = [eval(exp) for exp in lib]
    lhs = np.concatenate(lhs_columns, axis=1)
    rhs = eval('v_t')

    residual = rhs - lhs@coef

    # invoke STRidge algorithm to get the sparse coefficient vector
    trainer = STRidgeTrainer(R0=lhs, Ut=rhs, normalize=2, split_ratio=0.8)

    w_best = trainer.train(maxit = 150, STR_iters = 40, lam = 0.01, d_tol = 20, kappa = 1, must_have=6)  #

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
        if coef != 0:
            identified_dict[term] = coef

    print(identified_dict)

    # 0% noise  - {'ones*lap_v': 0.0050228611575184046, 'u*v_x': -0.983758330289342, 'v*v_y': -0.9712699608632717}
    # 5% noise  - {'ones*lap_v': 0.005254877974727355, 'u*v_x': -0.9877588783011164, 'v*v_y': -0.9884791915637039}
    # 10% noise - {'ones*lap_v': 0.005310344513948105, 'u*v_x': -0.9775847730371318, 'v*v_y': -0.9763934960670052}
    # 20% noise - {'ones*v_y': 0.17330399139307787, 'ones*lap_v': 0.005283945407260581, 'u*v_x': -0.975708097908202, 'v*v_y': -0.8933259722110424, 'v**2*v_y': -0.020155873870671692}
    # 30% noise - {'ones*v_y': 0.19273814529637842, 'ones*lap_v': 0.0056624116686659585, 'u*v_x': -0.9699321681661814, 'v*v_y': -0.8577201679703845, 'v**2*v_y': -0.0074639884391997}

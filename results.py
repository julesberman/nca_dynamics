import scipy
import numpy as np


class Results:
    def __init__(self, name, train_errors, test_errors, filters, params, betas):
        self.name = name
        self.train_errors = train_errors
        self.test_errors = test_errors
        self.params = params
        self.filters = filters
        self.betas = betas

    def get_mean_and_stds(self, x, axis=0):
        return np.mean(x, axis=axis), np.std(x, axis=axis)

    def get_train_errs(self):
        return self.get_mean_and_stds(self.train_errors, axis=1)

    def get_test_errs(self):
        return self.get_mean_and_stds(self.test_errors, axis=1)

    def get_opt_test_err(self):
        err, stds = self.get_test_errs()
        opt_i = self.get_opt_beta_i()
        return err[opt_i], stds[opt_i]

    def get_opt_beta_i(self):
        opt_i = np.argmin(np.mean(self.train_errors, axis=1))
        return opt_i

    def get_opt_beta(self, opt_set='train'):
        if opt_set == 'train':
            errs = self.train_errors
        if opt_set == 'test':
            errs = self.test_errors
        opt_i = np.argmin(np.mean(errs, axis=1))
        return self.betas[opt_i]

    def get_filter_for_beta(self, beta):
        i = (self.betas == beta)[0]
        opt_f = self.filters[i][0]
        return (np.mean(opt_f, axis=0), np.std(opt_f, axis=0)), self.betas[i][0]

    def get_opt_filter(self, opt_set='train'):
        if opt_set == 'train':
            errs = self.train_errors
        if opt_set == 'test':
            errs = self.test_errors
        opt_i = np.argmin(np.mean(errs, axis=1))
        opt_f = self.filters[opt_i]
        return (np.mean(opt_f, axis=0), np.std(opt_f, axis=0)), self.betas[opt_i]

    def get_params(self):
        num_params = len(self.params[0][0])
        remaped_params = []
        for n in range(num_params):
            p = np.zeros_like(self.params)
            for i in range(self.params.shape[0]):
                for j in range(self.params.shape[1]):
                    p[i, j] = self.params[i, j][n]
            remaped_params.append(p)

        return tuple(remaped_params)

    def get_avg_spectrum(self, beta=None):
        if beta is None:
            beta_i = self.get_opt_beta_i()
        else:
            beta_i = (self.betas == beta)[0]
        all_As = self.get_params()[3]
        beta_As = all_As[beta_i]
        Ws, VLs, VRs = [], [], []
        for A in beta_As:
            w, vl, vr = scipy.linalg.eig(A, left=True, right=True)

            # arrange for plotting
            sort = np.argsort(w)
            w = w[sort].real
            vl = vl[:, sort].real
            vr = vr[:, sort].real
            w = np.flip(w, axis=0)
            vl = np.flip(vl.T, axis=0)
            vr = np.flip(vr.T, axis=0)
            # orient
            for i in range(len(w)):
                vl[i] *= vl[i, -1]
                vr[i] *= vr[i, -1]

            Ws.append(w)
            VLs.append(vl)
            VRs.append(vr)

        return self.get_mean_and_stds(Ws, axis=0), self.get_mean_and_stds(VLs, axis=0), self.get_mean_and_stds(VRs, axis=0)

    def get_avg_A(self, beta=None):
        if beta is None:
            beta_i = self.get_opt_beta_i()
        else:
            beta_i = (self.betas == beta)[0]
        all_As = self.get_params()[3]
        beta_As = all_As[beta_i]
        avg_A = np.zeros_like(beta_As[0])
        for A in beta_As:
            avg_A += A / len(beta_As)

        return avg_A

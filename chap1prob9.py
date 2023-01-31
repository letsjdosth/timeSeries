from random import normalvariate, seed, uniform
from math import log, exp, pi, inf
from functools import partial

import numpy as np
import matplotlib.pyplot as plt

from pyBayes import MCMC_Core
from pyBayes.rv_gen_gamma import Sampler_univariate_InvGamma

seed(20230131)

class TAR_Simulator:
    def __init__(self) -> None:
        self.true_phi1 = 0.8
        self.true_phi2 = -0.3
        self.v1 = 1
        self.v2 = 0.5
        self.theta_threshold = 0.5
        self.d_lag = 1
    
    def sample_path_generator(self, T_length: int, initial:None|list[float]=None):
        path = []
        if initial is None:
            path.append(0)
        else:
            path += initial

        for _ in range(T_length-len(path)):
            if path[-self.d_lag] > -self.theta_threshold:
                #M1
                new = path[-1] * self.true_phi1 + normalvariate(0, self.v1**0.5)
            else:
                #M2
                new = path[-1] * self.true_phi2 + normalvariate(0, self.v2**0.5)
            path.append(new)
        return path

# ======================

def unif_proposal_log_pdf(from_smpl, to_smpl, window, hyper_a):
    from_smpl = from_smpl[0]
    to_smpl = to_smpl[0]
    applied_window = [max(-hyper_a, from_smpl-window/2), min(hyper_a, from_smpl+window/2)]
    if to_smpl<applied_window[0] or to_smpl>applied_window[1]:
        # return -inf
        raise ValueError("to_smpl has an unacceptable value")
    else:
        applied_window_len = applied_window[1] - applied_window[0]
        # return 1/applied_window_len
        return -log(applied_window_len)

def unif_proposal_sampler(from_smpl, window, hyper_a):
    from_smpl = from_smpl[0]
    applied_window = [max(-hyper_a, from_smpl-window/2), min(hyper_a, from_smpl+window/2)]
    return [uniform(applied_window[0], applied_window[1])]

# ======================

class Chap1Prob9(MCMC_Core.MCMC_Gibbs):
    def __init__(self, initial, y_path):
        # 0     1     2   3   4
        # phi1, phi2, v1, v2, theta
        self.MC_sample = [initial]
        self.inv_gamma_sampler_inst = Sampler_univariate_InvGamma()
        self.y_path = y_path

        #hyperparams
        self.hyper_c = 5
        self.hyper_a = 2
        self.hyper_a0 = 0.1
        self.hyper_b0 = 0.1

    def sampler(self, **kwargs):
        last = self.MC_sample[-1]
        new = self.deep_copier(last)
        #update new
        new = self.full_conditional_sampler_phi1(new)
        new = self.full_conditional_sampler_phi2(new)
        new = self.full_conditional_sampler_v1(new)
        new = self.full_conditional_sampler_v2(new)
        new = self.full_conditional_sampler_theta(new)
        self.MC_sample.append(new)
    
    def _get_y_with_threshold(self, theta_thres, model):
        #get [(y_t, y_{t-1})]
        selected_yt_yt1 = []
        if model==1: #theta > -y_{t-1}
            for i in range(1,len(self.y_path)):
                if self.y_path[i-1] > -theta_thres:
                    selected_yt_yt1.append((self.y_path[i],self.y_path[i-1]))
        elif model==2: #theta <= -y_{t-1}
            for i in range(1,len(self.y_path)):
                if self.y_path[i-1] <= -theta_thres:
                    selected_yt_yt1.append((self.y_path[i],self.y_path[i-1]))
        else:
            raise ValueError("model should be 1 or 2")
        return selected_yt_yt1

    def full_conditional_sampler_phi1(self, last_param):
        # 0     1     2   3   4
        # phi1, phi2, v1, v2, theta
        new_sample = [x for x in last_param]
        #update new
        selected_yt_yt1 = self._get_y_with_threshold(last_param[4], 1)
        yt1_square_sum = sum([y[1]**2 for y in selected_yt_yt1])
        yt_yt1_prod_sum = sum([y[0]*y[1] for y in selected_yt_yt1])
        variance = 1/(yt1_square_sum/last_param[2] + 1/self.hyper_c)
        mean = variance * yt_yt1_prod_sum / last_param[2]
        new_phi1 = normalvariate(mean, variance**0.5)
        
        new_sample[0] = new_phi1
        return new_sample
    
    def full_conditional_sampler_phi2(self, last_param):
        # 0     1     2   3   4
        # phi1, phi2, v1, v2, theta
        new_sample = [x for x in last_param]
        #update new
        selected_yt_yt1 = self._get_y_with_threshold(last_param[4], 2)
        yt1_square_sum = sum([y[1]**2 for y in selected_yt_yt1])
        yt_yt1_prod_sum = sum([y[0]*y[1] for y in selected_yt_yt1])
        variance = 1/(yt1_square_sum/last_param[3] + 1/self.hyper_c)
        mean = variance * yt_yt1_prod_sum / last_param[3]
        new_phi2 = normalvariate(mean, variance**0.5)
        
        new_sample[1] = new_phi2
        return new_sample
    
    def full_conditional_sampler_v1(self, last_param):
        # 0     1     2   3   4
        # phi1, phi2, v1, v2, theta
        new_sample = [x for x in last_param]
        #update new
        selected_yt_yt1 = self._get_y_with_threshold(last_param[4], 1)
        N1 = len(selected_yt_yt1)
        sum_square_error = sum([(y[0]-y[1]*last_param[0])**2 for y in selected_yt_yt1])
        alpha = self.hyper_a0 + N1/2
        beta = self.hyper_b0 + sum_square_error/2
        new_v1 = self.inv_gamma_sampler_inst.sampler(alpha, beta)

        new_sample[2] = new_v1
        return new_sample
    
    def full_conditional_sampler_v2(self, last_param):
        # 0     1     2   3   4
        # phi1, phi2, v1, v2, theta
        new_sample = [x for x in last_param]
        #update new
        selected_yt_yt1 = self._get_y_with_threshold(last_param[4], 2)
        N2 = len(selected_yt_yt1)
        sum_square_error = sum([(y[0]-y[1]*last_param[1])**2 for y in selected_yt_yt1])
        alpha = self.hyper_a0 + N2/2
        beta = self.hyper_b0 + sum_square_error/2
        new_v2 = self.inv_gamma_sampler_inst.sampler(alpha, beta)

        new_sample[3] = new_v2
        return new_sample
    
    def full_conditional_sampler_theta(self, last_param):
        # 0     1     2   3   4
        # phi1, phi2, v1, v2, theta
        new_sample = [x for x in last_param]
        #update new

        def log_sum(selected_yt_yt1, phi, v):
            sum_val = 0
            for y in selected_yt_yt1:
                sum_val += ((-0.5)*log(2*pi*v) - (0.5/v)*(y[0]-y[1]*phi)**2)
            return sum_val

        def log_target_pdf(theta, last_param):
            theta = theta[0]
            selected_yt_yt1_for_model_1 = self._get_y_with_threshold(theta, 1) #can I use 'self'?
            selected_yt_yt1_for_model_2 = self._get_y_with_threshold(theta, 2)
            log_pdf = 0
            log_pdf += log_sum(selected_yt_yt1_for_model_1, last_param[0], last_param[2])
            log_pdf += log_sum(selected_yt_yt1_for_model_2, last_param[1], last_param[3])
            if theta > self.hyper_a or theta < -self.hyper_a:
                return -inf
            else:
                return log_pdf
        mh_log_target_pdf = partial(log_target_pdf, last_param=last_param)
        mh_log_proposal_pdf = partial(unif_proposal_log_pdf, hyper_a = self.hyper_a, window=0.04)
        mh_proposal_sampler = partial(unif_proposal_sampler, hyper_a = self.hyper_a, window=0.04)

        MH_inst = MCMC_Core.MCMC_MH(mh_log_target_pdf, mh_log_proposal_pdf, mh_proposal_sampler, [last_param[4]])
        MH_inst.generate_samples(2, verbose=False)
        new_theta = MH_inst.MC_sample[-1][0]

        new_sample[4] = new_theta
        return new_sample


inst_simulator = TAR_Simulator()
sim_path = inst_simulator.sample_path_generator(10000)
# print(len(sim_path))
# plt.plot(sim_path)
# plt.show()

initial = [0,0,1,1,0]
gibbs_inst = Chap1Prob9(initial, sim_path)
gibbs_inst.generate_samples(3600) # set the window-size in theta's MH smaller and run Gibbs longer.

diag_inst = MCMC_Core.MCMC_Diag()
diag_inst.set_mc_sample_from_MCMC_instance(gibbs_inst)
diag_inst.set_variable_names(["phi1", "phi2", "v1", "v2", "theta"])
diag_inst.show_traceplot((2,3))
diag_inst.burnin(600)
diag_inst.print_summaries(4)
diag_inst.show_traceplot((2,3))
diag_inst.show_acf(30, (2,3))
diag_inst.show_hist((2,3))


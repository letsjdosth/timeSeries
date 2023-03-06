from random import seed
import csv

import numpy as np
import matplotlib.pyplot as plt

from ts_util.least_squares import OLS_by_QR, sym_defpos_matrix_inversion_cholesky

from pyBayes.rv_gen_gamma import Sampler_univariate_InvGamma
from pyBayes.MCMC_Core import MCMC_Diag


data_yt = []
data_zt = []
with open('dataset/yt_223_midterm.csv', newline='\n') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        data_yt.append(float(row[0]))
with open('dataset/zt_223_midterm.csv', newline='\n') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        data_zt.append(float(row[0]))

data_T = len(data_yt) #400


if __name__=="__main__":
    seed(20230304)
    numpy_sampler = np.random.default_rng(20230304)
    inv_gamma_sampler = Sampler_univariate_InvGamma(20230304)
    
    # plt.plot(np.arange(400), data_yt)
    # plt.plot(np.arange(400), data_zt)
    # plt.title("y & z path")
    # plt.show()
    

    p_order= 1
    y_ar_p = np.array(data_yt[p_order:])
    Ft_ar_p = [data_yt[p_order-1::-1] + [data_zt[1]]]
    for i in range(p_order, len(data_yt)-1):
        Ft_ar_p.append(data_yt[i:i-p_order:-1] + [data_zt[i+1]])
    Ft_ar_p = np.array(Ft_ar_p)

    FFt_inv, _ = sym_defpos_matrix_inversion_cholesky(Ft_ar_p.T@Ft_ar_p)
    
    MC_samples = []
    phi_samples = []
    for _ in range(3000):
        #sample v
        v_marginal_inv_gamma_alpha = (len(data_yt) - p_order*2 -1)/2
        beta_ols, sum_of_squares_ols = OLS_by_QR(Ft_ar_p, y_ar_p)
        v_marginal_inv_gamma_beta = 0.5*sum_of_squares_ols
        now_v = inv_gamma_sampler.sampler(v_marginal_inv_gamma_alpha, v_marginal_inv_gamma_beta)
        #sample phi
        phi_condpost_mean = beta_ols
        phi_condpost_var = FFt_inv * now_v
        now_beta = numpy_sampler.multivariate_normal(phi_condpost_mean, phi_condpost_var)

        new = list(now_beta) + [now_v]
        MC_samples.append(new)


    diag_inst_m2 = MCMC_Diag()
    diag_inst_m2.set_mc_samples_from_list(MC_samples)
    diag_inst_m2.set_variable_names(["phi","gamma","v"])
    diag_inst_m2.show_traceplot((1,3))
    diag_inst_m2.show_hist((1,3))
    diag_inst_m2.print_summaries(4)


    

# 5-step ahead prediction
    m2_predicted = []
    additional_zt = [0.106, 1.879, 1.56, 2.07, 0.66, None]
    for sample in diag_inst_m2.MC_sample:
        last_yz = [[data_yt[-1], additional_zt[0]]]
        for i in range(5):
            new_y = np.dot(np.array(sample[0:2]), np.array(last_yz[-1])) + numpy_sampler.normal(0, np.sqrt(sample[2]))
            last_yz.append([new_y, additional_zt[i+1]])
        
        m2_predicted.append([x[0] for x in last_yz][1:])

    m2_pred_inst = MCMC_Diag()
    m2_pred_inst.set_mc_samples_from_list(m2_predicted)
    m2_pred_inst.set_variable_names(["m2_"+str(i+1)+"step_predicted" for i in range(5)])
    m2_pred_inst.show_hist((1,5))
    m2_pred_inst.print_summaries(4)

    
    plt.plot(np.arange(400), data_yt)
    plt.plot(np.arange(405), data_zt+additional_zt[0:5])
    m2_pred_quant = m2_pred_inst.get_sample_quantile([0.025, 0.975])
    m2_pred_lower = [x[0] for x in m2_pred_quant]
    m2_pred_upper = [x[1] for x in m2_pred_quant]
    plt.plot([400, 401, 402, 403, 404, 405], [data_yt[-1]] + m2_pred_lower, color="grey")
    plt.plot([400, 401, 402, 403, 404, 405], [data_yt[-1]] + m2_pred_upper, color="grey")
    plt.plot([400, 401, 402, 403, 404, 405], [data_yt[-1]] + m2_pred_inst.get_sample_mean())
    plt.show()

from random import seed
import csv

import numpy as np
import matplotlib.pyplot as plt

from ts_util.arma_spectral_density import ARMA
from ts_util.least_squares import OLS_by_QR, sym_defpos_matrix_inversion_cholesky
from ts_util.time_series_utils import ar_polynomial_roots

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
    
    plt.plot(np.arange(400), data_yt)
    plt.plot(np.arange(400), data_zt)
    plt.title("y & z path")
    plt.show()
    

    # AR(4) fit
    p_order= 4
    y_ar_p = np.array(data_yt[p_order:])
    Ft_ar_p = [data_yt[p_order-1::-1]]
    for i in range(p_order, len(data_yt)-1):
        Ft_ar_p.append(data_yt[i:i-p_order:-1])
    Ft_ar_p = np.array(Ft_ar_p)

    FFt_inv, _ = sym_defpos_matrix_inversion_cholesky(Ft_ar_p.T@Ft_ar_p)
    
    MC_samples = []
    phi_samples = []
    bic_at_samples = []
    for _ in range(3000):
        #sample v
        v_marginal_inv_gamma_alpha = (len(data_yt) - p_order*2)/2
        beta_ols, sum_of_squares_ols = OLS_by_QR(Ft_ar_p, y_ar_p)
        v_marginal_inv_gamma_beta = 0.5*sum_of_squares_ols
        now_v = inv_gamma_sampler.sampler(v_marginal_inv_gamma_alpha, v_marginal_inv_gamma_beta)
        #sample phi
        phi_condpost_mean = beta_ols
        phi_condpost_var = FFt_inv * now_v
        now_phi = numpy_sampler.multivariate_normal(phi_condpost_mean, phi_condpost_var)

        new = list(now_phi) + [now_v]
        MC_samples.append(new)
        phi_samples.append(list(now_phi))


    # phi, w
    diag_inst_ar_4 = MCMC_Diag()
    diag_inst_ar_4.set_mc_samples_from_list(MC_samples)
    diag_inst_ar_4.set_variable_names(["\\tilde\{phi\}"+str(i) for i in range(1,p_order+1)]+["\\tilde\{v\}"])
    diag_inst_ar_4.show_traceplot((2,3))
    diag_inst_ar_4.show_hist((2,3))
    diag_inst_ar_4.print_summaries(4)


    # largest moduli of reciprocal roots
    ar4_poly_polar_rec_roots_at_samples = ar_polynomial_roots(phi_samples, reciprocal=True)
    ar4_rec_root_1st_largest_modulus = []
    ar4_rec_root_2nd_largest_modulus = []

    for rec_roots in ar4_poly_polar_rec_roots_at_samples:
        ar4_rec_root_1st_largest_modulus.append(rec_roots[-1][0])
        ar4_rec_root_2nd_largest_modulus.append(rec_roots[-2][0])
    
    print("1st")
    print("mean:", np.mean(ar4_rec_root_1st_largest_modulus), 
            ", var:", np.var(ar4_rec_root_1st_largest_modulus), 
            ", 95%CI:", np.quantile(ar4_rec_root_1st_largest_modulus,[0.025, 0.975]))
    print("2nd")
    print("mean:", np.mean(ar4_rec_root_2nd_largest_modulus), 
            ", var:", np.var(ar4_rec_root_2nd_largest_modulus), 
            ", 95%CI:", np.quantile(ar4_rec_root_2nd_largest_modulus,[0.025, 0.975]))
    
    fig_rec_root, ax_rec_root = plt.subplots(1, 2, figsize=(10, 5))
    fig_rec_root.tight_layout()
    ax_rec_root[0].hist(ar4_rec_root_1st_largest_modulus, bins=60)
    ax_rec_root[0].set_title("moduli of reciprocal root having the 1st largest moduli")

    ax_rec_root[1].hist(ar4_rec_root_2nd_largest_modulus, bins=60)
    ax_rec_root[1].set_title("moduli of reciprocal root having the 2st largest moduli")
    plt.show()


    # spectral density

    y_fft = np.fft.rfft(data_yt)
    periodogram_y = [f*f.conjugate()/(data_T*2*np.pi) for f in y_fft]
    periodogram_y[0] = np.mean(data_yt)

    fig_spec, ax_spec = plt.subplots(1, 2, figsize=(10, 5))
    fig_spec.tight_layout()

    post_spec_list = []
    for sample in diag_inst_ar_4.MC_sample:
        arma_4_0_inst = ARMA(sample[4], sample[0:4], None)
        spec_4_0, grid = arma_4_0_inst.spectral_density(data_T)
        ax_spec[0].plot(grid, spec_4_0, color="grey")
        post_spec_list.append(spec_4_0)
    post_spec_mean = np.mean(np.array(post_spec_list), axis=0)
    
    ax_spec[0].plot(grid, post_spec_mean, color="green")
    ax_spec[0].set_ylim(0,40)
    ax_spec[0].set_title("mean spec.density from AR(4) posterior samples")
    
    ax_spec[1].plot(grid, periodogram_y, color="#ff7f0e")
    ax_spec[1].plot(grid, post_spec_mean, color="green")
    ax_spec[1].set_ylim(0,40)
    ax_spec[1].set_title("posterior mean spec.density from AR(4) vs periodogram")
    plt.show()


# 5-step ahead prediction
    ar4_predicted = []
    for sample in diag_inst_ar_4.MC_sample:
        last_y = data_yt[-4:]
        for i in range(5):
            new_y = np.dot(np.array(sample[0:4]), np.array(last_y[-4:])) + numpy_sampler.normal(0, np.sqrt(sample[4]))
            last_y.append(new_y)
        ar4_predicted.append(last_y[-5:])

    ar4_pred_inst = MCMC_Diag()
    ar4_pred_inst.set_mc_samples_from_list(ar4_predicted)
    ar4_pred_inst.set_variable_names(["ar4_"+str(i+1)+"step_predicted" for i in range(5)])
    ar4_pred_inst.show_hist((1,5))
    ar4_pred_inst.print_summaries(4)

    
    plt.plot(np.arange(400), data_yt)
    ar4_pred_quant = ar4_pred_inst.get_sample_quantile([0.025, 0.975])
    ar4_pred_lower = [x[0] for x in ar4_pred_quant]
    ar4_pred_upper = [x[1] for x in ar4_pred_quant]
    plt.plot([400, 401, 402, 403, 404, 405], [data_yt[-1]] + ar4_pred_lower, color="grey")
    plt.plot([400, 401, 402, 403, 404, 405], [data_yt[-1]] + ar4_pred_upper, color="grey")
    plt.plot([400, 401, 402, 403, 404, 405], [data_yt[-1]] + ar4_pred_inst.get_sample_mean(), color="green")
    plt.show()

    ar4_param_mean = diag_inst_ar_4.get_sample_mean()
    print("naive estimates")
    phi_case1 = 0.5*(ar4_param_mean[0]+np.sqrt(ar4_param_mean[0]**2 + 4*ar4_param_mean[1]))
    theta_case1 = -ar4_param_mean[1]/phi_case1
    print("phi", phi_case1)
    print("theta", theta_case1)
    print("or")
    phi_case2 = 0.5*(ar4_param_mean[0]-np.sqrt(ar4_param_mean[0]**2 + 4*ar4_param_mean[1]))
    theta_case2 = -ar4_param_mean[1]/phi_case2
    print("phi", phi_case2)
    print("theta", theta_case2)
    
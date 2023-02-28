import math
from random import seed, normalvariate


import numpy as np
import matplotlib.pyplot as plt

from ts_util.arma_spectral_density import ARMA
from ts_util.least_squares import OLS_by_QR, sym_defpos_matrix_inversion_cholesky

from pyBayes.rv_gen_gamma import Sampler_univariate_InvGamma
from pyBayes.MCMC_Core import MCMC_Diag



if __name__=="__main__":
    #part (a, b, c)
    seed(20230226)
    arma_inst = ARMA(1, [0.9], None, 20230226)

    x_path_T500, _  = arma_inst.generate_random_path(500)
    y_path_T500 = [x + normalvariate(0, 1) for x in x_path_T500]
    plt.plot(np.arange(500), x_path_T500)
    plt.plot(np.arange(500), y_path_T500)
    plt.title("x & y path: one realization")
    plt.show()
    
    x_fft = np.fft.rfft(x_path_T500)
    periodogram_x = [f*f.conjugate()/(500*2*np.pi) for f in x_fft]
    periodogram_x[0] = np.mean(x_path_T500)
    y_fft = np.fft.rfft(y_path_T500)
    periodogram_y = [f*f.conjugate()/(500*2*np.pi) for f in y_fft]
    periodogram_y[0] = np.mean(y_path_T500)

    x_true_spec_density, grid = arma_inst.spectral_density(500, domain_0pi=True)
    y_true_spec_density = [f + (1/(2*math.pi)) for f in x_true_spec_density]
    
    fig1, ax1 = plt.subplots(1, 3, figsize=(15, 5))
    fig1.tight_layout()
    ax1[0].plot(grid, x_true_spec_density)
    ax1[0].plot(grid, y_true_spec_density)
    ax1[0].set_title("x & y: true spec.densities")

    ax1[1].plot(grid, periodogram_x, color="grey")
    ax1[1].plot(grid, x_true_spec_density, color="#1f77b4")
    ax1[1].set_title("x: true spec.density & periodogram")

    ax1[2].plot(grid, periodogram_y, color="grey")
    ax1[2].plot(grid, y_true_spec_density, color="#ff7f0e")
    ax1[2].set_title("y: true spec.density & periodogram")
    plt.show()


    #part d
    
    tuples_diag_bic = []
    numpy_sampler = np.random.default_rng(20230130)
    inv_gamma_sampler = Sampler_univariate_InvGamma(20230130)

    # BIC settings
    maximum_ar_order_p_star = 5
    n_at_p_star = 500 - maximum_ar_order_p_star

    for p_order in [1,2,3,4,5]:
        y_ar_p = np.array(y_path_T500[p_order:])
        Ft_ar_p = [y_path_T500[p_order-1::-1]]
        for i in range(p_order, len(y_path_T500)-1):
            Ft_ar_p.append(y_path_T500[i:i-p_order:-1])
        Ft_ar_p = np.array(Ft_ar_p)

        FFt_inv, _ = sym_defpos_matrix_inversion_cholesky(Ft_ar_p.T@Ft_ar_p)
        
        MC_samples = []
        bic_at_samples = []
        for _ in range(3000):
            #sample v
            v_marginal_inv_gamma_alpha = (len(y_path_T500) - p_order*2)/2
            beta_ols, sum_of_squares_ols = OLS_by_QR(Ft_ar_p, y_ar_p)
            v_marginal_inv_gamma_beta = 0.5*sum_of_squares_ols
            now_v = inv_gamma_sampler.sampler(v_marginal_inv_gamma_alpha, v_marginal_inv_gamma_beta)
            #sample phi
            phi_condpost_mean = beta_ols
            phi_condpost_var = FFt_inv * now_v
            now_phi = numpy_sampler.multivariate_normal(phi_condpost_mean, phi_condpost_var)

            new = list(now_phi) + [now_v]
            MC_samples.append(new)
            bic = n_at_p_star * math.log(sum_of_squares_ols/(len(y_path_T500)-2*p_order)) + p_order*math.log(n_at_p_star)
            bic_at_samples.append(bic)

        diag_inst_ar_p = MCMC_Diag()
        diag_inst_ar_p.set_mc_samples_from_list(MC_samples)
        diag_inst_ar_p.set_variable_names(["phi"+str(i) for i in range(1,p_order+1)]+["v"])
        bic_ar_p = np.mean(bic_at_samples)
        print("BIC at p =", p_order, " : ", bic_ar_p)
        tuples_diag_bic.append((diag_inst_ar_p, bic_ar_p))

    diag_inst_ar_2, bic_ar_2 = tuples_diag_bic[1]
    diag_inst_ar_2.show_traceplot((1,3))
    diag_inst_ar_2.show_hist((1,3))
    diag_inst_ar_2.print_summaries(4)

    
    fig2, ax2 = plt.subplots(1, 2, figsize=(10, 5))
    fig2.tight_layout()
    

    post_spec_list = []
    for sample in diag_inst_ar_2.MC_sample:
        arma_2_0_inst = ARMA(sample[2], sample[0:2], None)
        spec_2_0, _ = arma_2_0_inst.spectral_density(500)
        ax2[0].plot(grid, spec_2_0, color="grey")
        post_spec_list.append(spec_2_0)
    post_spec_mean = np.mean(np.array(post_spec_list), axis=0)
    
    ax2[0].plot(grid, post_spec_mean, color="green")
    ax2[0].set_title("mean spec.density from AR(2) posterior samples")
    ax2[1].plot(grid, y_true_spec_density, color="#ff7f0e")
    ax2[1].plot(grid, post_spec_mean, color="green")
    ax2[1].set_title("true(y) vs posterior mean spec.density from AR(2)")
    plt.show()
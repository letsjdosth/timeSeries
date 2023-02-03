import csv
import cmath
import math

import matplotlib.pyplot as plt
import numpy as np

import pyBayes.time_series_utils as ts
from pyBayes.rv_gen_gamma import Sampler_univariate_InvGamma
from pyBayes.MCMC_Core import MCMC_Diag

# loading data

infln = []
with open('dataset/usdata.csv', newline='\n') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)
    for row in reader:
        infln.append(float(row[0]))

diff_infln = ts.difference_oper(infln) # T=195
T = 195
# plt.plot(diff_infln)
# plt.show()


# AR(8) fit
y = np.array(diff_infln[8:]) # 187*1
Ft = [diff_infln[7::-1]]
for i in range(8, len(diff_infln)-1):
    Ft.append(diff_infln[i:i-8:-1])
Ft = np.array(Ft) #187*8
print(y.shape, Ft.shape)

FFt_inv = np.linalg.inv(Ft.T@Ft)

numpy_sampler = np.random.default_rng(20230130)
inv_gamma_sampler = Sampler_univariate_InvGamma(20230130)
MC_samples = []
for _ in range(10000):
    #sample v
    v_marginal_inv_gamma_alpha = (T-16)/2
    sum_of_squares = (y- Ft @ FFt_inv @ Ft.T @ y)
    v_marginal_inv_gamma_beta = 0.5*((sum_of_squares.T)@sum_of_squares)
    now_v = inv_gamma_sampler.sampler(v_marginal_inv_gamma_alpha, v_marginal_inv_gamma_beta)
    
    phi_condpost_mean = FFt_inv@Ft.T@y
    phi_condpost_var = FFt_inv * now_v
    now_phi = numpy_sampler.multivariate_normal(phi_condpost_mean, phi_condpost_var)

    new = list(now_phi) + [now_v]
    MC_samples.append(new)

diag_inst = MCMC_Diag()
diag_inst.set_mc_samples_from_list(MC_samples)
diag_inst.set_variable_names(["phi"+str(i) for i in range(1,9)]+["v"])
diag_inst.print_summaries(5)
diag_inst.show_hist((3,3))


# ar_poly_polar_roots_at_samples = []
# for sample in diag_inst.MC_sample:
#     phi_sample = ([1] + [-x for x in sample])[0:9]
#     # print(phi_sample)
#     ar_poly = np.polynomial.polynomial.Polynomial(phi_sample)
#     ar_poly_roots = ar_poly.roots()
#     ar_poly_polar_roots_at_samples.append([cmath.polar(x) for x in ar_poly_roots])

phi_samples = [sample[0:8] for sample in diag_inst.MC_sample]
ar_poly_polar_rec_roots_at_samples = ts.ar_polynomial_roots(phi_samples, reciprocal=True)


# longest period
def sort_key2(c):
    return abs(c[1])

ar_char_amp_of_longest_period = []
ar_char_longest_period = []

for rec_roots in ar_poly_polar_rec_roots_at_samples:
    rec_roots.sort(key=sort_key2, reverse=False)
    for rec_root in rec_roots:
        if rec_root[1] == 0: #real root
            pass
        else:
            ar_char_longest_period.append(2*math.pi/abs(rec_root[1]))
            ar_char_amp_of_longest_period.append(rec_root[0])
            break
print("mean:",np.mean(ar_char_longest_period), 
        ", var:",np.var(ar_char_longest_period), 
        ", 95%CI:",np.quantile(ar_char_longest_period,[0.05, 0.95]))
plt.hist(ar_char_longest_period, bins=1200)
plt.xlim(0, 40)
plt.title("the longest period")
plt.show()

print("mean:",np.mean(ar_char_amp_of_longest_period), 
        ", var:",np.var(ar_char_amp_of_longest_period), 
        ", 95%CI:",np.quantile(ar_char_amp_of_longest_period,[0.05, 0.95]))
plt.hist(ar_char_amp_of_longest_period, bins=60)
plt.xlim(0, 1.75)
plt.title("the amplitude of the longest period case")
plt.show()



# largest moduli of reciprocal roots (Is the series stable? // how much dominant is the oscilatory effect?)
def sort_key1(c):
    return c[0]

ar_char_largest_moduli = []

ar_char_largest_moduli_real = [] #for comparing amplitudes between oscilatory vs exp-decaying componants
ar_char_largest_moduli_imaginary = []

for rec_roots in ar_poly_polar_rec_roots_at_samples:
    rec_roots.sort(key=sort_key1, reverse=True)
    largest_rec_root = rec_roots[0]
    ar_char_largest_moduli.append(largest_rec_root[0])
    if largest_rec_root[1] == 0: #real
        ar_char_largest_moduli_real.append(largest_rec_root[0])
    else:
        ar_char_largest_moduli_imaginary.append(largest_rec_root[0])

print("mean:", np.mean(ar_char_largest_moduli), 
        ", var:", np.var(ar_char_largest_moduli), 
        ", 95%CI:", np.quantile(ar_char_largest_moduli,[0.05, 0.95]))
plt.hist(ar_char_largest_moduli, bins=60)
plt.xlim(0, 1.5)
plt.title("the largest modulus (among all real and imaginary roots)")
plt.show()


# compare?
print(np.count_nonzero(ar_char_largest_moduli_real), 
        np.count_nonzero(ar_char_largest_moduli_imaginary))
from random import uniform, seed
import math

import numpy as np
import matplotlib.pyplot as plt


from pyBayes.MCMC_Core import MCMC_Gibbs, MCMC_MH, MCMC_Diag
from pyBayes.DLM_Core import DLM_model_container, DLM_simulator, DLM_full_model
from ts_util.least_squares import sym_defpos_matrix_inversion_cholesky
from ts_util.time_series_utils import difference_oper, autocorr

# ===
delta1 = 0.95
true_model1_inst = DLM_model_container(102)
true_model1_inst.set_F_const_design_mat(np.array([[1],[0]]))
true_model1_inst.set_G_const_transition_mat(np.array([
    [1,1],
    [0,1]
]))
true_model1_inst.set_V_const_obs_eq_covariance(np.array([[100]]))
true_model1_inst.set_W_const_state_error_cov(100*np.array([
    [1-(delta1**2), (1-delta1)**2],
    [(1-delta1)**2, (1-delta1)**3]
]))
simulator1_inst = DLM_simulator(true_model1_inst, 20230318)
simulator1_inst.simulate_data(np.array([100, 0]), 
                              100*np.array([[1-delta1**2, (1-delta1)**2],[(1-delta1)**2, (1-delta1)**3]]))
data1_theta, data1_y = simulator1_inst.get_theta_y()

# ===
delta2 = 0.8
true_model2_inst = DLM_model_container(102)
true_model2_inst.set_F_const_design_mat(np.array([[1],[0]]))
true_model2_inst.set_G_const_transition_mat(np.array([
    [1,1],
    [0,1]
]))
true_model2_inst.set_V_const_obs_eq_covariance(np.array([[100]]))
true_model2_inst.set_W_const_state_error_cov(100*np.array([
    [1-(delta2**2), (1-delta2)**2],
    [(1-delta2)**2, (1-delta2)**3]
]))
simulator2_inst = DLM_simulator(true_model2_inst, 20230318)
simulator2_inst.simulate_data(np.array([100, 0]), 
                              100*np.array([[1-delta2**2, (1-delta2)**2],[(1-delta2)**2, (1-delta2)**3]]))

data2_theta, data2_y = simulator2_inst.get_theta_y()

fig_sim2, ax_sim2 = plt.subplots(3,3,figsize=(12, 4))
fig_sim2.tight_layout()
ax_sim2[0,0].plot(data2_y)
ax_sim2[0,0].set_title("y")
ax_sim2[0,1].plot(difference_oper(data2_y))
ax_sim2[0,1].set_title("diff1(y)")
ax_sim2[0,2].plot(difference_oper(difference_oper(data2_y)))
ax_sim2[0,2].set_title("diff2(y)")
ax_sim2[1,0].bar(range(51), autocorr(data2_y,50))
ax_sim2[1,0].set_title("y,acf")
ax_sim2[1,1].bar(range(51), autocorr(difference_oper(data2_y),50))
ax_sim2[1,1].set_title("diff1(y),acf")
ax_sim2[1,2].bar(range(51), autocorr(difference_oper(difference_oper(data2_y)),50))
ax_sim2[1,2].set_title("diff2(y),acf")
ax_sim2[2,0].plot([theta[0] for theta in data2_theta])
ax_sim2[2,0].set_title("theta1")
ax_sim2[2,1].plot([theta[1] for theta in data2_theta])
ax_sim2[2,1].set_title("theta2")
plt.show()


class FFBS_final(MCMC_Gibbs):
    def __init__(self, y_observation, dlm_model_container: DLM_model_container, initial_delta:float, seed_val):
        self.data_y = y_observation
        self.data_T = len(y_observation)
        seed(seed_val)
        self.DLM_model = dlm_model_container

        #param
        # 0                           1
        # [[theta(0:T)](zero!! to T), delta]
        self.MC_sample = [[[np.array([[0],[0]]) for _ in range(self.data_T+1)], initial_delta]]

        self.np_random_inst = np.random.default_rng(seed_val)
    
    def sampler(self, **kwargs):
        last = self.MC_sample[-1]
        new = self.deep_copier(last)
        #update new
        new = self.full_conditional_sampler_theta(new)
        new = self.full_conditional_sampler_delta(new)
        self.MC_sample.append(new)
    
    def full_conditional_sampler_theta(self, last_param):
        #param
        # 0                           1
        # [[theta(0:T)](zero!! to T), delta]
        new_sample = [x for x in last_param]

        last_delta = last_param[1]
        W_delta = 100 * np.array([[1-last_delta**2, (1-last_delta)**2],
                                    [(1-last_delta)**2, (1-last_delta)**3]])
        ffbs_model_container_inst = DLM_model_container(self.data_T)
        ffbs_model_container_inst.set_F_const_design_mat(np.array([[1],[0]]))
        ffbs_model_container_inst.set_G_const_transition_mat(np.array([[1,1],[0,1]]))
        ffbs_model_container_inst.set_V_const_obs_eq_covariance([100])
        ffbs_model_container_inst.set_W_const_state_error_cov(W_delta)
        ffbs_filtering_inst = DLM_full_model(self.data_y, ffbs_model_container_inst, last_param[0][0], np.array([[1,0],[0,1]]))
        ffbs_filtering_inst.run()
        a, R = ffbs_filtering_inst.get_prior_a_R()
        m, C = ffbs_filtering_inst.get_posterior_m_C()
        self.C = C

        theta_T = self.np_random_inst.multivariate_normal(np.reshape(m[-1],(2,)), C[-1])
        new_sample[0][self.data_T] = np.reshape(theta_T,(2,1))


        for t in range(self.data_T-1, 0, -1):
            Bt = C[t-1]@ np.transpose(self.DLM_model.G_sys_eq_transition[t]) @ np.linalg.inv(R[t])
            It = np.reshape(m[t-1],(2,)) + Bt @ (np.reshape(new_sample[0][t+1],(2,)) - np.reshape(a[t],(2,)))
            Lt = C[t-1] - Bt @ R[t] @ np.transpose(Bt)
            theta_t = self.np_random_inst.multivariate_normal(It, Lt)
            new_sample[0][t] = np.reshape(theta_t, (2,1))

        #at 0, using prior,
        C0 = np.array([[1,0],[0,1]])
        m0 = np.array([[100],[0]])
        B0 = C0 @ np.transpose(self.DLM_model.G_sys_eq_transition[0]) @ np.linalg.inv(R[0])
        I0 = m0 + B0 @ np.transpose(new_sample[0][1] - np.transpose(a[0]))
        L0 = C0 - B0 @ R[0] @ np.transpose(B0)
        theta_0 = self.np_random_inst.multivariate_normal(np.transpose(I0)[0], L0)
        new_sample[0][0] = np.reshape(theta_0, (2,1))
        return new_sample

    def full_conditional_sampler_delta(self, last_param):
        #param
        # 0                           1
        # [[theta(0:T)](zero!! to T), delta]
        new_sample = [x for x in last_param]
        #update new
        def unif01_proposal_log_pdf(from_smpl, to_smpl, window=0.1):
            from_smpl = from_smpl[0]
            to_smpl = to_smpl[0]
            applied_window = [max(0, from_smpl-window/2), min(1, from_smpl+window/2)]
            if to_smpl<applied_window[0] or to_smpl>applied_window[1]:
                # return -inf
                raise ValueError("to_smpl has an unacceptable value")
            else:
                applied_window_len = applied_window[1] - applied_window[0]
                # return 1/applied_window_len
                return -np.log(applied_window_len)

        def unif01_proposal_sampler(from_smpl, window=0.1):
            from_smpl = from_smpl[0]
            applied_window = [max(0, from_smpl-window/2), min(1, from_smpl+window/2)]
            return [uniform(applied_window[0], applied_window[1])]
        
        def log_target_pdf(now_delta):
            now_delta = now_delta[0]
            W_delta = 100 * np.array([[1-now_delta**2, (1-now_delta)**2],
                                      [(1-now_delta)**2, (1-now_delta)**3]])
            # inv_W_delta, log_det_W_delta= sym_defpos_matrix_inversion_cholesky(W_delta)

            post = 0
            for i in range(1, self.data_T+1):
                thetas = last_param[0]
                Gi = self.DLM_model.G_sys_eq_transition[i-1]
                # inv_W_delta, log_det_W_delta= sym_defpos_matrix_inversion_cholesky(Gi @ self.C[i-1] @ np.transpose(Gi) + W_delta)
                inv_W_delta, log_det_W_delta= sym_defpos_matrix_inversion_cholesky(W_delta)
                innov_theta = thetas[i] - (Gi @ thetas[i-1])
                post -= (np.transpose(innov_theta) @ inv_W_delta @ innov_theta/2 + log_det_W_delta/2)
            return post

        new_delta = last_param[1]
        while new_delta == last_param[1]:
            mh_inst = MCMC_MH(log_target_pdf, unif01_proposal_log_pdf, unif01_proposal_sampler, [last_param[1]])
            mh_inst.generate_samples(2, verbose=False)
            new_delta = (mh_inst.MC_sample[-1])[0]
        new_sample[1] = new_delta
        return new_sample

model2e_container_inst = DLM_model_container(102)
model2e_container_inst.set_F_const_design_mat(np.array([[1],[0]]))
model2e_container_inst.set_G_const_transition_mat(np.array([[1,1],[0,1]]))
model2e_container_inst.set_V_const_obs_eq_covariance([100])
gibbs_inst = FFBS_final(data2_y, model2e_container_inst, 0.5, 20230319)
gibbs_inst.generate_samples(3000)

theta_samples = [x[0] for x in gibbs_inst.MC_sample[300:]]
for theta in theta_samples:
    plt.plot([x[1] for x in theta], color="green")
    plt.plot([x[0] for x in theta], color="red")
plt.plot(range(1,102+1), data2_y)
plt.show()

delta_samples = [[x[1]] for x in gibbs_inst.MC_sample[300:]]
delta_diag_inst = MCMC_Diag()
delta_diag_inst.set_mc_samples_from_list(delta_samples)
delta_diag_inst.set_variable_names(["delta"])
delta_diag_inst.print_summaries(4)
delta_diag_inst.show_traceplot((1,1))
delta_diag_inst.show_hist((1,1))
delta_diag_inst.show_acf(30, (1,1))



#(+)
# choosing delta by optimization
optimal_delta = 0
optimal_mse = math.inf
for delta in np.linspace(0.1, 1, num=91, endpoint=True):
    model2e_dopt_container = DLM_model_container(102) 
    model2e_dopt_container.set_F_const_design_mat(np.array([[1],[0]]))
    model2e_dopt_container.set_G_const_transition_mat(np.array([[1,1],[0,1]]))
    model2e_dopt_container.set_V_const_obs_eq_covariance(np.array([[100]]))
    model2e_dopt_container.set_W_const_state_error_cov(100*np.array([
        [1-(delta**2), (1-delta)**2],
        [(1-delta)**2, (1-delta)**3]
    ]))
    model2e_dopt_fit_inst = DLM_full_model(
        data2_y, model2e_dopt_container,
        np.array([100,0]),
        np.array([[1,0],[0,1]])
    )
    model2e_dopt_fit_inst.run()
    mse_onestep_f = np.mean([e**2 for e in model2e_dopt_fit_inst.e_one_step_forecast_err])
    if mse_onestep_f < optimal_mse:
        optimal_delta = delta
        optimal_mse = mse_onestep_f
print(optimal_delta, optimal_mse)  # 0.77 189.8744434187878


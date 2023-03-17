import numpy as np
import scipy.stats as scss
import matplotlib.pyplot as plt

from pyBayes.DLM_Core import DLM_univariate_y_without_V_W_in_D0, DLM_model_container

data_uk_gas_consume = [ 4.9, 4.5, 3.1, 3.1, 4.6, 4.8, 6.7, 8.6,
    8.3, 7.2, 9.2, 5.5, 4.7, 4.4, 3.4, 2.8, 4.0, 5.1, 6.5, 9.2,
    7.7, 7.7, 8.9, 5.7, 5.0, 4.5, 3.3, 2.8, 4.0, 5.6, 6.6, 10.3,
    8.5, 7.9, 8.9, 5.4, 4.4, 4.0, 3.0, 3.1, 4.4, 5.5, 6.5, 10.1,
    7.7, 9.0, 9.0, 6.5, 5.1, 4.3, 2.7, 2.8, 4.6, 5.5, 6.9, 9.5,
    8.8, 8.7, 10.1, 6.1, 5.0, 4.5, 3.1, 2.9, 4.8
]

data_T = len(data_uk_gas_consume) #65
data_yt = np.array([[y-np.mean(data_uk_gas_consume)] for y in data_uk_gas_consume])

# model 1

model1_model_inst = DLM_model_container(data_T+20)
model1_model_inst.set_F_const_design_mat(np.array([
    [1],
    [0],
    [1],
    [0]
]))
model1_model_inst.set_G_const_transition_mat(np.array([
    [np.sqrt(3)/2, 0.5, 0, 0],
    [-0.5, np.sqrt(3)/2, 0, 0],
    [0,0,0,1],
    [0,0,-1,0]
]))


# choosing delta
for delta in np.linspace(0.85, 1, num=32, endpoint=True):
    dlm_fit_inst = DLM_univariate_y_without_V_W_in_D0(data_yt, model1_model_inst,
                                                    initial_m0_given_D0 = np.array([0,0,0,0]),
                                                    initial_C0st_given_D0 = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]),
                                                    n0_given_D0 = 0.01,
                                                    S0_given_D0 = 1,
                                                    discount_factor_for_Wst = delta
    )
    dlm_fit_inst.run()
    mse_onestep_f = np.sum([e**2 for e in dlm_fit_inst.e_one_step_forecast_err])/data_T
    print("delta:", delta, " mse:", mse_onestep_f)


#at delta=0.985

model1_fit_inst = DLM_univariate_y_without_V_W_in_D0(data_yt, model1_model_inst,
                                                    initial_m0_given_D0 = np.array([0,0,0,0]),
                                                    initial_C0st_given_D0 = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]),
                                                    n0_given_D0 = 0.01,
                                                    S0_given_D0 = 1,
                                                    discount_factor_for_Wst = 0.985
)
model1_fit_inst.run()
model1_fit_inst.get_one_step_forecast_f_Q()

posterior_mt, posterior_ct = model1_fit_inst.get_posterior_m_C()
forecast_f, forecast_Q = model1_fit_inst.get_one_step_forecast_f_Q()

z95 = 1.959964
t95 = [scss.t.ppf([0.025, 0.975], df=d)[0] for d in model1_fit_inst.n_precision_shape] # len: 400

# one step forecast
plt.scatter(range(1, data_T+1), [y[0] for y in data_yt], s=10) #blue dot: obs
plt.plot(range(1, data_T+1), [f[0] for f in forecast_f], color="green") #green: one-step forecast E(Y_t|D_{t-1})
cred_interval_upper = [f[0] + t*np.sqrt(q[0][0]) for f, q, t in zip(forecast_f, forecast_Q, t95)]
cred_interval_lower = [f[0] - t*np.sqrt(q[0][0]) for f, q, t in zip(forecast_f, forecast_Q, t95)]
plt.plot(range(1, data_T+1), cred_interval_upper, color="grey") #one-step forecast (Y_t|D_{t-1}) 95% credible interval
plt.plot(range(1, data_T+1), cred_interval_lower, color="grey") #one-step forecast (Y_t|D_{t-1}) 95% credible interval
plt.show()


# filtering - theta1
plt.scatter(range(1, data_T+1), [y[0] for y in data_yt], s=10) #blue dot: obs
plt.plot(range(1, data_T+1), [m[0] for m in posterior_mt], color="orange") #orange: posterior E(theta_t|D_t)
cred_interval_upper = [m[0] + t*np.sqrt(c[0][0]) for m, c, t in zip(posterior_mt, posterior_ct, t95)]
cred_interval_lower = [m[0] - t*np.sqrt(c[0][0]) for m, c, t in zip(posterior_mt, posterior_ct, t95)]
plt.plot(range(1, data_T+1), cred_interval_upper, color="grey") #posterior (\theta_t|D_{t}) 95% credible interval
plt.plot(range(1, data_T+1), cred_interval_lower, color="grey") #posterior (\theta_t|D_{t}) 95% credible interval
plt.show()

# filtering - theta3
plt.scatter(range(1, data_T+1), [y[0] for y in data_yt], s=10) #blue dot: obs diff
plt.plot(range(1, data_T+1), [m[2] for m in posterior_mt], color="orange") #orange: posterior E(theta_t|D_t)
cred_interval_upper = [m[2] + t*np.sqrt(c[2][2]) for m, c, t in zip(posterior_mt, posterior_ct, t95)]
cred_interval_lower = [m[2] - t*np.sqrt(c[2][2]) for m, c, t in zip(posterior_mt, posterior_ct, t95)]
plt.plot(range(1, data_T+1), cred_interval_upper, color="grey") #posterior (\theta_t|D_{t}) 95% credible interval
plt.plot(range(1, data_T+1), cred_interval_lower, color="grey") #posterior (\theta_t|D_{t}) 95% credible interval
plt.show()

# smoothing
model1_fit_inst.run_retrospective_analysis()
retro_a_at_T, retro_R_at_T = model1_fit_inst.get_retrospective_a_R()


plt.scatter(range(1, data_T+1), [y[0] for y in data_yt], s=10) #blue dot: obs
plt.plot(range(1, data_T+1), [a[0] for a in retro_a_at_T], color="red")
cred_interval_upper = [a[0] + z95*np.sqrt(r[0][0]) for a, r in zip(retro_a_at_T, retro_R_at_T)]
cred_interval_lower = [a[0] - z95*np.sqrt(r[0][0]) for a, r in zip(retro_a_at_T, retro_R_at_T)]
plt.plot(range(1, data_T+1), cred_interval_upper, color="grey") #posterior (\theta_t|D_{T}) 95% credible interval
plt.plot(range(1, data_T+1), cred_interval_lower, color="grey") #posterior (\theta_t|D_{T}) 95% credible interval
plt.show()


plt.scatter(range(1, data_T+1), [y[0] for y in data_yt], s=10) #blue dot: obs diff
plt.plot(range(1, data_T+1), [a[2] for a in retro_a_at_T], color="red")
cred_interval_upper = [a[2] + z95*np.sqrt(r[2][2]) for a, r in zip(retro_a_at_T, retro_R_at_T)]
cred_interval_lower = [a[2] - z95*np.sqrt(r[2][2]) for a, r in zip(retro_a_at_T, retro_R_at_T)]
plt.plot(range(1, data_T+1), cred_interval_upper, color="grey") #posterior (\theta_t|D_{T}) 95% credible interval
plt.plot(range(1, data_T+1), cred_interval_lower, color="grey") #posterior (\theta_t|D_{T}) 95% credible interval
plt.show()

#comparison

plt.scatter(range(1, data_T+1), [y[0] for y in data_yt], s=10) #blue dot: obs
plt.plot(range(1, data_T+1), [m[0] for m in posterior_mt], color="orange") #orange: posterior E(theta_t|D_t), filtering
plt.plot(range(1, data_T+1), [a[0] for a in retro_a_at_T], color="red") #red: smoothing, E(theta_t|D_T)
plt.show()

plt.scatter(range(1, data_T+1), [y[0] for y in data_yt], s=10) #blue dot: obs diff
plt.plot(range(1, data_T+1), [m[2] for m in posterior_mt], color="orange") #orange: posterior E(theta_t|D_t), filtering
plt.plot(range(1, data_T+1), [a[2] for a in retro_a_at_T], color="red") #red: smoothing, E(theta_t|D_T)
plt.show()


# 20-step ahead forecast
forecast_step = 20
model1_fit_inst.run_forecast_analysis(data_T, data_T+forecast_step)
fo_mean, fo_q = model1_fit_inst.get_forecast_f_Q()


plt.scatter(range(1, data_T+1), [y[0] for y in data_yt], s=10) #blue dot: obs
plt.plot(range(1, data_T+1), [f[0] for f in forecast_f], color="green") #green: one-step forecast E(Y_t|D_{t-1})
cred_interval_upper = [f[0] + t*np.sqrt(q[0][0]) for f, q, t in zip(forecast_f, forecast_Q, t95)]
cred_interval_lower = [f[0] - t*np.sqrt(q[0][0]) for f, q, t in zip(forecast_f, forecast_Q, t95)]
plt.plot(range(1, data_T+1), cred_interval_upper, color="grey") #one-step forecast (Y_t|D_{t-1}) 95% credible interval
plt.plot(range(1, data_T+1), cred_interval_lower, color="grey") #one-step forecast (Y_t|D_{t-1}) 95% credible interval

plt.plot(range(data_T+1, data_T+forecast_step+1), [f[0] for f in fo_mean], color="green") #green: 5-step ahead prediction
plt.plot(range(data_T+1, data_T+forecast_step+1), [f[0]+z95*np.sqrt(q[0][0]) for f, q in zip(fo_mean, fo_q)], color="grey")
plt.plot(range(data_T+1, data_T+forecast_step+1), [f[0]-z95*np.sqrt(q[0][0]) for f, q in zip(fo_mean, fo_q)], color="grey")
plt.show()


pred_cred_interval_upper = [f[0]+z95*np.sqrt(q[0][0]) for f, q in zip(fo_mean, fo_q)]
pred_cred_interval_lower = [f[0]-z95*np.sqrt(q[0][0]) for f, q in zip(fo_mean, fo_q)]

print(fo_mean, fo_q)
for i in range(forecast_step):
    print(str(i+1)+" step ahead, ", "mean:", round(fo_mean[i][0],3), " variance:", round(fo_q[i][0][0],3), 
        " 95% CI:(", round(pred_cred_interval_lower[i],3),",",round(pred_cred_interval_upper[i],3),")")
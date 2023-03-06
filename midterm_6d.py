import csv

import numpy as np
import scipy.stats as scss
import matplotlib.pyplot as plt

from pyBayes.DLM_Core import DLM_univariate_y_without_V_W_in_D0, DLM_D0_container


data_yt = []
data_zt = []
with open('dataset/yt_223_midterm.csv', newline='\n') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        data_yt.append(np.array([float(row[0])]))
with open('dataset/zt_223_midterm.csv', newline='\n') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        data_zt.append(np.array([float(row[0])]))

data_T = len(data_yt) #400

if __name__=="__main__":
    dlm_d0_inst = DLM_D0_container(data_T+5)
    dlm_d0_inst.set_F_const_design_mat(np.array([[1],[0]]))
    dlm_d0_inst.set_G_const_transition_mat(np.array([[1,1],
                                                    [0,1]]))
    dlm_d0_inst.set_u_no_covariate()

    # choosing delta
    # for delta in np.linspace(0.7, 1, num=31, endpoint=True):
    #     dlm_fit_inst = DLM_univariate_y_without_V_W_in_D0(
    #         data_yt,
    #         dlm_d0_inst,
    #         np.array([[0],[0]]),
    #         np.array([[1,0],
    #                 [0,1]]),
    #         0.01,
    #         1,
    #         discount_factor_for_Wst=delta
    #     )
    #     dlm_fit_inst.run()
        
    #     posterior_mt, posterior_ct = dlm_fit_inst.get_posterior_m_C()
    #     forecast_f, forecast_Q = dlm_fit_inst.get_one_step_forecast_f_Q()
        
    #     mse_onestep_f = np.sum([(f-y)**2 + float(q[0][0]) for y, f, q in zip(data_yt, forecast_f, forecast_Q)])/data_T
    #     print("delta:", delta, " mse:", mse_onestep_f)

    dlm_fit_inst = DLM_univariate_y_without_V_W_in_D0(
        data_yt,
        dlm_d0_inst,
        np.array([[0],[0]]),
        np.array([[1,0],
                [0,1]]),
        1,
        1,
        discount_factor_for_Wst=0.92
    )
    dlm_fit_inst.run()
    
    posterior_mt, posterior_ct = dlm_fit_inst.get_posterior_m_C()
    forecast_f, forecast_Q = dlm_fit_inst.get_one_step_forecast_f_Q()
    
    # print(dlm_fit_inst.n_precision_shape)
    z95 = 1.959964
    t95 = [scss.t.ppf([0.025, 0.975], df=d)[0] for d in dlm_fit_inst.n_precision_shape] # len: 400
    # print(t95)

    
    # print(dlm_fit_inst.S_precision_rate)
    # print(dlm_fit_inst.R_prior_scale)

    # one step forecast
    # plt.plot(range(data_T), [m[0] for m in posterior_mt], color="orange") #orange: posterior E(theta_t|D_t)
    # plt.plot(range(data_T), [m[1] for m in posterior_mt], color="orange") #orange: posterior E(theta_t|D_t)
    plt.plot(range(data_T), [f[0] for f in forecast_f], color="green") #green: one-step forecast E(Y_t|D_{t-1})
    plt.scatter(range(data_T), [y[0] for y in data_yt], s=10) #blue dot: obs
    cred_interval_upper = [f[0] + t*np.sqrt(q[0][0]) for f, q, t in zip(forecast_f, forecast_Q, t95)]
    cred_interval_lower = [f[0] - t*np.sqrt(q[0][0]) for f, q, t in zip(forecast_f, forecast_Q, t95)]
    plt.plot(range(data_T), cred_interval_upper, color="grey") #one-step forecast (Y_t|D_{t-1}) 95% credible interval
    plt.plot(range(data_T), cred_interval_lower, color="grey") #one-step forecast (Y_t|D_{t-1}) 95% credible interval
    plt.show()


    # filtering - theta1
    plt.scatter(range(data_T), [y[0] for y in data_yt], s=10) #blue dot: obs
    plt.plot(range(data_T), [m[0] for m in posterior_mt], color="orange") #orange: posterior E(theta_t|D_t)
    cred_interval_upper = [m[0] + t*np.sqrt(c[0][0]) for m, c, t in zip(posterior_mt, posterior_ct, t95)]
    cred_interval_lower = [m[0] - t*np.sqrt(c[0][0]) for m, c, t in zip(posterior_mt, posterior_ct, t95)]
    plt.plot(range(data_T), cred_interval_upper, color="grey") #posterior (\theta_t|D_{t}) 95% credible interval
    plt.plot(range(data_T), cred_interval_lower, color="grey") #posterior (\theta_t|D_{t}) 95% credible interval
    plt.show()

    # filtering - theta2
    data_yt_diff = np.array(data_yt[1:])-np.array(data_yt[:-1])
    # plt.scatter(range(data_T), [y[0] for y in data_yt], s=10) #blue dot: obs
    plt.scatter(range(1,data_T), [y[0] for y in data_yt_diff], s=10) #blue dot: obs diff
    plt.plot(range(data_T), [m[1] for m in posterior_mt], color="orange") #orange: posterior E(theta_t|D_t)
    cred_interval_upper = [m[1] + t*np.sqrt(c[1][1]) for m, c, t in zip(posterior_mt, posterior_ct, t95)]
    cred_interval_lower = [m[1] - t*np.sqrt(c[1][1]) for m, c, t in zip(posterior_mt, posterior_ct, t95)]
    plt.plot(range(data_T), cred_interval_upper, color="grey") #posterior (\theta_t|D_{t}) 95% credible interval
    plt.plot(range(data_T), cred_interval_lower, color="grey") #posterior (\theta_t|D_{t}) 95% credible interval
    plt.show()

    # smoothing
    dlm_fit_inst.run_retrospective_analysis()
    retro_a_at_T, retro_R_at_T = dlm_fit_inst.get_retrospective_a_R()
    plt.scatter(range(data_T), [y[0] for y in data_yt], s=10) #blue dot: obs
    plt.plot(range(data_T), [a[0] for a in retro_a_at_T], color="red")
    cred_interval_upper = [a[0] + z95*np.sqrt(r[0][0]) for a, r in zip(retro_a_at_T, retro_R_at_T)]
    cred_interval_lower = [a[0] - z95*np.sqrt(r[0][0]) for a, r in zip(retro_a_at_T, retro_R_at_T)]
    plt.plot(range(data_T), cred_interval_upper, color="grey") #posterior (\theta_t|D_{T}) 95% credible interval
    plt.plot(range(data_T), cred_interval_lower, color="grey") #posterior (\theta_t|D_{T}) 95% credible interval
    plt.show()


    plt.scatter(range(1,data_T), [y[0] for y in data_yt_diff], s=10) #blue dot: obs diff
    plt.plot(range(data_T), [a[1] for a in retro_a_at_T], color="red")
    cred_interval_upper = [a[1] + z95*np.sqrt(r[1][1]) for a, r in zip(retro_a_at_T, retro_R_at_T)]
    cred_interval_lower = [a[1] - z95*np.sqrt(r[1][1]) for a, r in zip(retro_a_at_T, retro_R_at_T)]
    plt.plot(range(data_T), cred_interval_upper, color="grey") #posterior (\theta_t|D_{T}) 95% credible interval
    plt.plot(range(data_T), cred_interval_lower, color="grey") #posterior (\theta_t|D_{T}) 95% credible interval
    plt.show()

    #comparison

    plt.scatter(range(data_T), [y[0] for y in data_yt], s=10) #blue dot: obs
    plt.plot(range(data_T), [m[0] for m in posterior_mt], color="orange") #orange: posterior E(theta_t|D_t), filtering
    plt.plot(range(data_T), [a[0] for a in retro_a_at_T], color="red") #red: smoothing, E(theta_t|D_T)
    plt.show()

    plt.scatter(range(1,data_T), [y[0] for y in data_yt_diff], s=10) #blue dot: obs diff
    plt.plot(range(data_T), [m[1] for m in posterior_mt], color="orange") #orange: posterior E(theta_t|D_t), filtering
    plt.plot(range(data_T), [a[1] for a in retro_a_at_T], color="red") #red: smoothing, E(theta_t|D_T)
    plt.show()


    # 5-step ahead forecast
    dlm_fit_inst.run_forecast_analysis(data_T, data_T+5)
    fo_mean, fo_q = dlm_fit_inst.get_forecast_f_Q()
    # print(fo_mean, fo_q)
    
    plt.scatter(range(data_T), [y[0] for y in data_yt], s=10) #blue dot: obs
    plt.plot(range(data_T), [m[0] for m in posterior_mt], color="orange") #orange: posterior E(theta_t|D_t)
    cred_interval_upper = [m[0] + t*np.sqrt(c[0][0]) for m, c, t in zip(posterior_mt, posterior_ct, t95)]
    cred_interval_lower = [m[0] - t*np.sqrt(c[0][0]) for m, c, t in zip(posterior_mt, posterior_ct, t95)]
    plt.plot(range(data_T), cred_interval_upper, color="grey") #posterior (\theta_t|D_{t}) 90% credible interval
    plt.plot(range(data_T), cred_interval_lower, color="grey") #posterior (\theta_t|D_{t}) 90% credible interval
    plt.plot(range(data_T), [a[0] for a in retro_a_at_T], color="red")
    cred_interval_upper = [a[0] + z95*np.sqrt(r[0][0]) for a, r in zip(retro_a_at_T, retro_R_at_T)]
    cred_interval_lower = [a[0] - z95*np.sqrt(r[0][0]) for a, r in zip(retro_a_at_T, retro_R_at_T)]
    plt.plot(range(data_T), cred_interval_upper, color="grey") #posterior (\theta_t|D_{T}) 95% credible interval
    plt.plot(range(data_T), cred_interval_lower, color="grey") #posterior (\theta_t|D_{T}) 95% credible interval
    
    plt.plot(range(data_T, data_T+5), [f[0] for f in fo_mean], color="green") #green: 5-step ahead prediction
    plt.plot(range(data_T, data_T+5), [f[0]+z95*np.sqrt(q[0][0]) for f, q in zip(fo_mean, fo_q)], color="grey")
    plt.plot(range(data_T, data_T+5), [f[0]-z95*np.sqrt(q[0][0]) for f, q in zip(fo_mean, fo_q)], color="grey")
    plt.show()

    
    pred_cred_interval_upper = [f[0]+z95*np.sqrt(q[0][0]) for f, q in zip(fo_mean, fo_q)]
    pred_cred_interval_lower = [f[0]-z95*np.sqrt(q[0][0]) for f, q in zip(fo_mean, fo_q)]
    for i in range(5):
        print(str(i+1)+" step ahead, ", "mean:", round(fo_mean[i][0][0],3), " variance:", round(fo_q[i][0][0],3), 
            " 95% CI:(", round(pred_cred_interval_lower[i][0],3),",",round(pred_cred_interval_upper[i][0],3),")")
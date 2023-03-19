import random
import math

import numpy as np
import matplotlib.pyplot as plt

from pyBayes.DLM_Core import DLM_model_container, DLM_simulator, DLM_without_W_by_discounting, DLM_visualizer, DLM_univariate_y_without_V_in_D0
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


# fig_sim1, ax_sim1 = plt.subplots(3,3,figsize=(12, 4))
# fig_sim1.tight_layout()
# ax_sim1[0,0].plot(data1_y)
# ax_sim1[0,0].set_title("y")
# ax_sim1[0,1].plot(difference_oper(data1_y))
# ax_sim1[0,1].set_title("diff1(y)")
# ax_sim1[0,2].plot(difference_oper(difference_oper(data1_y)))
# ax_sim1[0,2].set_title("diff2(y)")
# ax_sim1[1,0].bar(range(51), autocorr(data1_y,50))
# ax_sim1[1,0].set_title("y,acf")
# ax_sim1[1,1].bar(range(51), autocorr(difference_oper(data1_y),50))
# ax_sim1[1,1].set_title("diff1(y),acf")
# ax_sim1[1,2].bar(range(51), autocorr(difference_oper(difference_oper(data1_y)),50))
# ax_sim1[1,2].set_title("diff2(y),acf")
# ax_sim1[2,0].plot([theta[0] for theta in data1_theta])
# ax_sim1[2,0].set_title("theta1")
# ax_sim1[2,1].plot([theta[1] for theta in data1_theta])
# ax_sim1[2,1].set_title("theta2")
# plt.show()

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

# fig_sim2, ax_sim2 = plt.subplots(3,3,figsize=(12, 4))
# fig_sim2.tight_layout()
# ax_sim2[0,0].plot(data2_y)
# ax_sim2[0,0].set_title("y")
# ax_sim2[0,1].plot(difference_oper(data2_y))
# ax_sim2[0,1].set_title("diff1(y)")
# ax_sim2[0,2].plot(difference_oper(difference_oper(data2_y)))
# ax_sim2[0,2].set_title("diff2(y)")
# ax_sim2[1,0].bar(range(51), autocorr(data2_y,50))
# ax_sim2[1,0].set_title("y,acf")
# ax_sim2[1,1].bar(range(51), autocorr(difference_oper(data2_y),50))
# ax_sim2[1,1].set_title("diff1(y),acf")
# ax_sim2[1,2].bar(range(51), autocorr(difference_oper(difference_oper(data2_y)),50))
# ax_sim2[1,2].set_title("diff2(y),acf")
# ax_sim2[2,0].plot([theta[0] for theta in data2_theta])
# ax_sim2[2,0].set_title("theta1")
# ax_sim2[2,1].plot([theta[1] for theta in data2_theta])
# ax_sim2[2,1].set_title("theta2")
# plt.show()


# === 2c
optimal_delta = 0
optimal_mse = math.inf
for delta in np.linspace(0.5, 1, num=51, endpoint=True):
    model2c_d095_container = DLM_model_container(102) 
    #because DLM_without_W_by_discounting modifies the container, we should make a new one for each time
    model2c_d095_container.set_F_const_design_mat(np.array([[1]]))
    model2c_d095_container.set_G_const_transition_mat(np.array([[1]]))
    model2c_d095_container.set_V_const_obs_eq_covariance(np.array([[100]]))
    model2c_d095_fit_inst = DLM_without_W_by_discounting(
        data1_y, model2c_d095_container,
        np.array([100]),
        np.array([[1]]),
        delta
    )
    model2c_d095_fit_inst.run()
    mse_onestep_f = np.mean([e**2 for e in model2c_d095_fit_inst.e_one_step_forecast_err])
    if mse_onestep_f < optimal_mse:
        optimal_delta = delta
        optimal_mse = mse_onestep_f
print(optimal_delta, optimal_mse)  # 0.61 131.96412183402023

model2c_d095_container = DLM_model_container(112)
model2c_d095_container.set_F_const_design_mat(np.array([[1]]))
model2c_d095_container.set_G_const_transition_mat(np.array([[1]]))
model2c_d095_container.set_V_const_obs_eq_covariance(np.array([[100]]))
    
model2c_d095_fit_inst = DLM_without_W_by_discounting(
        data1_y, model2c_d095_container,
        np.array([100]),
        np.array([[1]]),
        0.61
    )
model2c_d095_fit_inst.run()
model2c_d095_fit_inst.run_retrospective_analysis()
model2c_d095_fit_inst.run_forecast_analysis(102, 112)

# model2c_d095_vis_inst = DLM_visualizer(model2c_d095_fit_inst, 0.95, False)
# model2c_d095_vis_inst.show_one_step_forecast()
# model2c_d095_vis_inst.show_filtering((1,1))
# model2c_d095_vis_inst.show_smoothing((1,1))
# model2c_d095_vis_inst.show_one_step_forecast(show=False, title_str="")
# model2c_d095_vis_inst.show_forecasting()
# plt.show()


# ===
optimal_delta = 0
optimal_mse = math.inf
for delta in np.linspace(0.5, 1, num=51, endpoint=True):
    model2c_d080_container = DLM_model_container(102)
    model2c_d080_container.set_F_const_design_mat(np.array([[1]]))
    model2c_d080_container.set_G_const_transition_mat(np.array([[1]]))
    model2c_d080_container.set_V_const_obs_eq_covariance(np.array([[100]]))
    model2c_d080_fit_inst = DLM_without_W_by_discounting(
        data2_y, model2c_d080_container,
        np.array([100]),
        np.array([[1]]),
        delta
    )
    model2c_d080_fit_inst.run()
    mse_onestep_f = np.mean([e**2 for e in model2c_d080_fit_inst.e_one_step_forecast_err])
    if mse_onestep_f < optimal_mse:
        optimal_delta = delta
        optimal_mse = mse_onestep_f
print(optimal_delta, optimal_mse)  # 0.5 407.3431334410629

model2c_d080_container = DLM_model_container(112)
model2c_d080_container.set_F_const_design_mat(np.array([[1]]))
model2c_d080_container.set_G_const_transition_mat(np.array([[1]]))
model2c_d080_container.set_V_const_obs_eq_covariance(np.array([[100]]))
    
model2c_d080_fit_inst = DLM_without_W_by_discounting(
        data2_y, model2c_d080_container,
        np.array([100]),
        np.array([[1]]),
        0.5
    )
model2c_d080_fit_inst.run()
model2c_d080_fit_inst.run_retrospective_analysis()
model2c_d080_fit_inst.run_forecast_analysis(102, 112)

# model2c_d080_vis_inst = DLM_visualizer(model2c_d080_fit_inst, 0.95, False)
# model2c_d080_vis_inst.show_one_step_forecast()
# model2c_d080_vis_inst.show_filtering((1,1))
# model2c_d080_vis_inst.show_smoothing((1,1))
# model2c_d080_vis_inst.show_one_step_forecast(show=False, title_str="")
# model2c_d080_vis_inst.show_forecasting()
# plt.show()


# === 2d

model2d_d095_container = DLM_model_container(102+10)
model2d_d095_container.set_F_const_design_mat(np.array([[1],[0]]))
model2d_d095_container.set_G_const_transition_mat(np.array([
    [1,1],
    [0,1]
]))
model2d_d095_container.set_Wst_const_state_error_scale_free_cov(np.array([
    [(1-0.95**2), (1-0.95)**2],
    [(1-0.95)**2, (1-0.95)**3]
]))
model2d_d095_fit_inst = DLM_univariate_y_without_V_in_D0(
    data1_y, model2d_d095_container,
    np.array([0,0]),
    np.array([[1,0],[0,1]]),
    0.01,
    1
)
model2d_d095_fit_inst.run()
model2d_d095_fit_inst.run_retrospective_analysis()
model2d_d095_fit_inst.run_forecast_analysis(102, 112)

# model2d_d095_vis_inst = DLM_visualizer(model2d_d095_fit_inst, 0.95, True)
# model2d_d095_vis_inst.show_one_step_forecast()
# model2d_d095_vis_inst.show_filtering((1,2))
# model2d_d095_vis_inst.show_smoothing((1,2))
# model2d_d095_vis_inst.show_one_step_forecast(show=False, title_str="")
# model2d_d095_vis_inst.show_forecasting()
# plt.show()

# ===

model2d_d080_container = DLM_model_container(102+10)
model2d_d080_container.set_F_const_design_mat(np.array([[1],[0]]))
model2d_d080_container.set_G_const_transition_mat(np.array([
    [1,1],
    [0,1]
]))
model2d_d080_container.set_Wst_const_state_error_scale_free_cov(np.array([
    [(1-0.8**2), (1-0.8)**2],
    [(1-0.8)**2, (1-0.8)**3]
]))
model2d_d080_fit_inst = DLM_univariate_y_without_V_in_D0(
    data2_y, model2d_d080_container,
    np.array([0,0]),
    np.array([[1,0],[0,1]]),
    0.01,
    1
)
model2d_d080_fit_inst.run()
model2d_d080_fit_inst.run_retrospective_analysis()
model2d_d080_fit_inst.run_forecast_analysis(102, 112)

# model2d_d080_vis_inst = DLM_visualizer(model2d_d080_fit_inst, cred=0.95, is_used_t_dist=True)
# model2d_d080_vis_inst.show_one_step_forecast()
# model2d_d080_vis_inst.show_filtering((1,2))
# model2d_d080_vis_inst.show_smoothing((1,2))
# model2d_d080_vis_inst.show_one_step_forecast(show=False, title_str="")
# model2d_d080_vis_inst.show_forecasting()
# plt.show()



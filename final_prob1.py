import csv
import math

import numpy as np
import matplotlib.pyplot as plt

from pyBayes.DLM_Core import DLM_univariate_y_without_V_W_in_D0_with_component_discount_factor, DLM_model_container, DLM_visualizer

# ===== loading data & plotting
data_date = []
data_gtidx = []
with open('dataset/UCSC.csv', newline='\n') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)
    for row in reader:
        data_date.append(str(row[0]))
        data_gtidx.append(int(row[1]))

data_T = len(data_gtidx) # 146

# plt.plot(data_gtidx)
# plt.xticks(range(0, data_T, 12), labels=data_date[0:data_T:12], rotation=45)
# plt.show()

# ===== annual sense based
# fig_anncycle, ax_anncycle = plt.subplots(3, 1)
# ax_anncycle[0].plot(data_gtidx)
# ax_anncycle[0].vlines([x if x%12==0 else 0 for x in range(data_T)], ymin=0, ymax=100, colors="red", linestyles='dashed')
# ax_anncycle[1].plot(data_gtidx)
# ax_anncycle[1].vlines([x if x%6==0 else 0 for x in range(data_T)], ymin=0, ymax=100, colors="red", linestyles='dashed')
# ax_anncycle[2].plot(data_gtidx)
# ax_anncycle[2].set_xticks(range(0, data_T, 12), rotation=45)
# ax_anncycle[2].set_xticklabels(data_date[0:data_T:12], rotation=45)
# ax_anncycle[2].vlines([x if x%3==0 else 0 for x in range(data_T)], ymin=0, ymax=100, colors="red", linestyles='dashed')
# plt.show()

# ===== periodogram based
data_gtidx_fft = np.fft.rfft(data_gtidx)
periodogram_grid = np.linspace(0, np.pi, num=int(data_T/2)+1, endpoint=True)
periodogram_data_gtidx = [f*f.conjugate()/(data_T*2*np.pi) for f in data_gtidx_fft]
# plt.plot(periodogram_grid, np.log(periodogram_data_gtidx))
# plt.title("log periodogram")
# plt.show()

for freq, val in zip(periodogram_grid, periodogram_data_gtidx):
    if np.log(val) > 5:
        period = 2*np.pi/freq
        print(freq, period, val)

# fig_perio_cycle, ax_perio_cycle = plt.subplots(4, 1)
# ax_perio_cycle[0].plot(data_gtidx)
# ax_perio_cycle[0].vlines([x if x%73==0 else 0 for x in range(data_T)], ymin=0, ymax=100, colors="orange", linestyles='dashed')
# ax_perio_cycle[1].plot(data_gtidx)
# ax_perio_cycle[1].vlines([x if x%48==0 else 0 for x in range(data_T)], ymin=0, ymax=100, colors="orange", linestyles='dashed')
# ax_perio_cycle[2].plot(data_gtidx)
# ax_perio_cycle[2].vlines([x if x%12==0 else 0 for x in range(data_T)], ymin=0, ymax=100, colors="orange", linestyles='dashed')
# ax_perio_cycle[3].plot(data_gtidx)
# ax_perio_cycle[3].vlines([x if x%6==0 else 0 for x in range(data_T)], ymin=0, ymax=100, colors="orange", linestyles='dashed')
# ax_perio_cycle[3].set_xticks(range(0, data_T, 12), rotation=45)
# ax_perio_cycle[3].set_xticklabels(data_date[0:data_T:12], rotation=45)
# plt.show()

# ===== 1-b,c

data_date_until_feb2020 = data_date[:110]
data_gtidx_until_feb2020 = np.array([[y] for y in data_gtidx[:110]])
# print(data_date_until_feb2020[-1])
data_T_until_feb2020 = len(data_gtidx_until_feb2020) #110

# plt.plot(data_gtidx)
# plt.xticks(range(0, data_T, 12), labels=data_date[0:data_T:12], rotation=45)
# plt.vlines([110], ymin=0, ymax=100, colors="blue", linestyles='dashed')
# plt.show()

# 2nd-order polynomial(1st degree) + p=12, year/half-year/quarter

model1a_model_inst = DLM_model_container(data_T_until_feb2020+12)
model1a_model_inst.set_F_const_design_mat(np.array([
    [1],
    [0],
    [1],
    [0],
    [1],
    [0]
]))
model1a_model_inst.set_G_const_transition_mat(np.array([
    [1, 1, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0],
    [0, 0, np.sqrt(3)/2, 0.5, 0, 0],
    [0, 0, -0.5, np.sqrt(3)/2, 0, 0],
    [0, 0, 0, 0, 0.5, np.sqrt(3)/2],
    [0, 0, 0, 0, -np.sqrt(3)/2, 0.5]
]))


# # choosing delta
# optimal_delta_combination = (0, 0)
# optimal_mse = math.inf
# for delta_trend in np.linspace(0.7, 1, num=32, endpoint=True):
#     for delta_seasonal in np.linspace(0.7, 1, num=32, endpoint=True):
#         dlm_fit_inst = DLM_univariate_y_without_V_W_in_D0_with_component_discount_factor(data_gtidx_until_feb2020, model1a_model_inst,
#                                                         initial_m0_given_D0 = np.array([0,0,0,0,0,0]),
#                                                         initial_C0st_given_D0 = np.eye(6,6), #identity
#                                                         n0_given_D0 = 0.01,
#                                                         S0_given_D0 = 1,
#                                                         discount_factor_tuple=(delta_trend, delta_seasonal, delta_seasonal),
#                                                         discount_component_blocks_partition=(2,4,6)
#         )
#         dlm_fit_inst.run()
#         mse_onestep_f = np.sum([e**2 for e in dlm_fit_inst.e_one_step_forecast_err])/data_T
#         # print("delta:(", delta_trend, ",", delta_seasonal, ") mse:", mse_onestep_f)
#         if mse_onestep_f < optimal_mse:
#             optimal_delta_combination = (delta_trend, delta_seasonal)
#             optimal_mse = mse_onestep_f
# print(optimal_delta_combination, optimal_mse) #(0.8548387096774194, 0.9806451612903226) 167.06709700556627

model1a_fit_inst = DLM_univariate_y_without_V_W_in_D0_with_component_discount_factor(data_gtidx_until_feb2020, model1a_model_inst,
                                                    initial_m0_given_D0 = np.array([0,0,0,0,0,0]),
                                                    initial_C0st_given_D0 = np.eye(6,6), #identity
                                                    n0_given_D0 = 0.01,
                                                    S0_given_D0 = 1,
                                                    discount_factor_tuple=(0.855, 0.98, 0.98),
                                                    discount_component_blocks_partition=(2,4,6)
)
model1a_fit_inst.run()
model1a_fit_inst.run_retrospective_analysis()
forecast_step = 12
model1a_fit_inst.run_forecast_analysis(data_T_until_feb2020, data_T_until_feb2020+forecast_step)

model1a_vis_inst = DLM_visualizer(model1a_fit_inst, 0.95, True)
model1a_vis_inst.set_variable_names(["theta0","theta1","a1","b1","a2","b2"])
model1a_vis_inst.show_one_step_forecast()
model1a_vis_inst.show_filtering((2,1), [0,1])
model1a_vis_inst.show_filtering((2,1), [2,4])
model1a_vis_inst.show_smoothing((2,1), [0,1])
model1a_vis_inst.show_smoothing((2,1), [2,4])
model1a_vis_inst.show_added_smoothing()

model1a_vis_inst.show_one_step_forecast(show=False, title_str="")
model1a_vis_inst.show_forecasting(show=False, print_summary=True)
plt.scatter(range(data_T_until_feb2020+1, data_T), data_gtidx[data_T_until_feb2020+1:data_T], s=10)
plt.show()

# errors
model1a_1st_forecast_err = model1a_fit_inst.e_one_step_forecast_err
# model1a_err_sq = [e**2 for e in model1a_1st_forecast_err]
model1a_fitted_err = [y-np.transpose(f)@z for y, f, z in zip(data_gtidx_until_feb2020, model1a_model_inst.F_obs_eq_design[:data_T_until_feb2020],model1a_fit_inst.m_posterior_mean)]

plt.plot(range(1,data_T_until_feb2020+1), model1a_1st_forecast_err)
plt.plot(range(1,data_T_until_feb2020+1), model1a_fitted_err)
plt.title("errors: 1_step_forecast(blue), fitted_post(orange)")
plt.show()

# v|D_T, v~inv.gamma(,)
print(model1a_fit_inst.n_precision_shape[-1])
print(model1a_fit_inst.S_precision_rate[-1])



# # ===== 1-d
# # 2nd-order polynomial(1st degree) + p=12, year/half-year/quarter

model1d_model_inst = DLM_model_container(data_T+12)
model1d_model_inst.set_F_const_design_mat(np.array([
    [1],
    [0],
    [1],
    [0],
    [1],
    [0]
]))
model1d_model_inst.set_G_const_transition_mat(np.array([
    [1, 1, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0],
    [0, 0, np.sqrt(3)/2, 0.5, 0, 0],
    [0, 0, -0.5, np.sqrt(3)/2, 0, 0],
    [0, 0, 0, 0, 0.5, np.sqrt(3)/2],
    [0, 0, 0, 0, -np.sqrt(3)/2, 0.5]
]))


# choosing delta
# optimal_delta_combination = (0, 0)
# optimal_mse = math.inf
# for delta_trend in np.linspace(0.7, 1, num=32, endpoint=True):
#     for delta_seasonal in np.linspace(0.7, 1, num=32, endpoint=True):
#         dlm_fit_inst = DLM_univariate_y_without_V_W_in_D0_with_component_discount_factor(data_gtidx, model1d_model_inst,
#                                                         initial_m0_given_D0 = np.array([0,0,0,0,0,0]),
#                                                         initial_C0st_given_D0 = np.eye(6,6), #identity
#                                                         n0_given_D0 = 0.01,
#                                                         S0_given_D0 = 1,
#                                                         discount_factor_tuple=(delta_trend, delta_seasonal, delta_seasonal),
#                                                         discount_component_blocks_partition=(2,4,6)
#         )
#         dlm_fit_inst.run()
#         mse_onestep_f = np.sum([e**2 for e in dlm_fit_inst.e_one_step_forecast_err])/data_T
#         # print("delta:(", delta_trend, ",", delta_seasonal, ") mse:", mse_onestep_f)
#         if mse_onestep_f < optimal_mse:
#             optimal_delta_combination = (delta_trend, delta_seasonal)
#             optimal_mse = mse_onestep_f
# print(optimal_delta_combination, optimal_mse) # (0.8161290322580645, 0.9612903225806452) 187.30119118548356

model1d_fit_inst = DLM_univariate_y_without_V_W_in_D0_with_component_discount_factor(data_gtidx, model1d_model_inst,
                                                    initial_m0_given_D0 = np.array([0,0,0,0,0,0]),
                                                    initial_C0st_given_D0 = np.eye(6,6), #identity
                                                    n0_given_D0 = 0.01,
                                                    S0_given_D0 = 1,
                                                    discount_factor_tuple=(0.816, 0.961, 0.961),
                                                    discount_component_blocks_partition=(2,4,6)
)
model1d_fit_inst.run()
model1d_fit_inst.run_retrospective_analysis()
forecast_step = 12
model1d_fit_inst.run_forecast_analysis(data_T, data_T+forecast_step)

model1d_vis_inst = DLM_visualizer(model1d_fit_inst, 0.95, True)
model1d_vis_inst.set_variable_names(["theta0","theta1","a1","b1","a2","b2"])
model1d_vis_inst.show_one_step_forecast()
model1d_vis_inst.show_filtering((2,1), [0,1])
model1d_vis_inst.show_filtering((2,1), [2,4])
model1d_vis_inst.show_smoothing((2,1), [0,1])
model1d_vis_inst.show_smoothing((2,1), [2,4])
model1d_vis_inst.show_added_smoothing()

model1d_vis_inst.show_one_step_forecast(show=False, title_str="")
model1d_vis_inst.show_forecasting(show=False, print_summary=True)
plt.show()


# errors
model1d_1st_forecast_err = model1d_fit_inst.e_one_step_forecast_err
# model1d_err_sq = [e**2 for e in model1d_1st_forecast_err]
model1d_fitted_err = [y-np.transpose(f)@z for y, f, z in zip(data_gtidx, model1d_model_inst.F_obs_eq_design[:data_T],model1d_fit_inst.m_posterior_mean)]

plt.plot(range(1,data_T+1), model1d_1st_forecast_err)
plt.plot(range(1,data_T+1), model1d_fitted_err)
plt.title("errors: 1_step_forecast(blue), fitted_post(orange)")
plt.show()

# v|D_T, v~inv.gamma(,)
print(model1d_fit_inst.n_precision_shape[-1])
print(model1d_fit_inst.S_precision_rate[-1])



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Ridge
from sklearn.model_selection import LeaveOneOut
from GPyOpt.methods import BayesianOptimization
from GPy.kern import Matern32
import scipy.io
import random
import tensorflow as tf
plt.rcParams['font.family'] = 'Arial'
np.seterr(divide='ignore', invalid='ignore')
# Set seed parameters to ensure reproducible results
random.seed(5)
np.random.seed(5)
tf.random.set_seed(5)

# load data
file = r'./data.xlsx'
df1 = pd.read_excel(file, index_col=0)
data = np.array(df1, 'float32')

# actual value
Y0_Actual = data[0:22, 0:4]
m2 = len(Y0_Actual)

# !!! Select forecast horizon and execute
pre_steps = 1 #  1 or 2 or 3 or 4
train_end = m2 - pre_steps

# Dataset partitioning
Y0  = data[0:train_end, 0:4].copy()
X01 = data[0:train_end, 4:8].copy()
X02 = data[0:train_end, 8:12].copy()
X03 = data[0:train_end, 12:16].copy()
Y1 = np.cumsum(Y0, axis=0)
X1 = np.cumsum(X01, axis=0)
X2 = np.cumsum(X02, axis=0)
X3 = np.cumsum(X03, axis=0)
m = len(X1)
m1 = len(Y0)

#  obtain the spatial matrix
def matrix_w(q,m):
    d = np.array([[1, 278, 169, 309],
                  [278, 1, 425, 177],
                  [169, 425, 1, 464],
                  [309, 177, 464, 1]])

    gdp = data[0:m2, 16:20].copy().T

    wd = np.zeros((4, 4))
    for i in range(4):
        for j in range(4):
            if i == j:
                wd[i, j] = 0
            else:
                wd[i, j] = 1 / d[i, j] ** q[0]

    we = np.zeros((4, 4, m))
    for k in range(m):
        for i in range(4):
            for j in range(4):
                if i == j:
                    we[i, j, k] = 0
                else:
                    we[i, j, k] = 1 / (abs(gdp[i, k] - gdp[j, k])) ** q[1]

    tt1 = np.diag([q[2], q[3], q[4], q[5]])
    tt2 = np.diag([1 - q[2], 1 - q[3], 1 - q[4], 1 - q[5]])

    w = np.zeros((4, 4, m))
    for k in range(m):
        w[:, :, k] = tt1 @ wd + tt2 @ we[:, :, k]

    return w

# The variables are collated and then used to calculate parameters
def XY_obtain(q, Y1, X1, X2, X3):
   
    m = len(Y1)
    # print(f"m = {m}, Y1 shape = {Y1.shape}, X1 shape = {X1.shape}, X2 shape = {X2.shape}, X3 shape = {X3.shape}")
   
    w0 = matrix_w(q,m)

    # Generate y vectors using loops
    y_vectors = []
    for i in range(4):
        y = np.zeros((4 * (m - 1), 1))
        y[i * (m - 1):(i + 1) * (m - 1), :] = Y1[:m - 1, i].reshape(m - 1, 1)
        y_vectors.append(y)

    # Generate x vectors for X1, X2, X3 using loops
    x_vectors = []
    for X in [X1, X2, X3]:
        for i in range(4):
            x = np.zeros((4 * (m - 1), 1))
            x[i * (m - 1):(i + 1) * (m - 1), :] = X[1:, i].reshape(m - 1, 1)
            x_vectors.append(x)

    # Calculate w * Y, w * X
    Yw = np.zeros((m, 4))
    Xw = [np.zeros((m, 4)) for _ in range(3)]
    for k in range(m):
        Yw[k, :] = (w0[:, :, k] @ Y1.T[:, k]).T
        for j, X in enumerate([X1, X2, X3]):
            Xw[j][k, :] = (w0[:, :, k] @ X.T[:, k]).T

    # Generate weighted y vectors using loops
    yw_vectors = []
    for i in range(4):
        y = np.zeros((4 * (m - 1), 1))
        y[i * (m - 1):(i + 1) * (m - 1), :] = Yw[:m - 1, i].reshape(m - 1, 1)
        yw_vectors.append(y)

    # Generate weighted x vectors for X1w, X2w, X3w using loops
    xw_vectors = []
    for X in Xw:
        for i in range(4):
            x = np.zeros((4 * (m - 1), 1))
            x[i * (m - 1):(i + 1) * (m - 1), :] = X[1:, i].reshape(m - 1, 1)
            xw_vectors.append(x)

    # Generate c vectors using loops
    c_vectors = []
    for i in range(4):
        c = np.zeros((4 * (m - 1), 1))
        c[i * (m - 1):(i + 1) * (m - 1), :] = 1
        c_vectors.append(c)

    # Combine all vectors into one matrix
    vectors = y_vectors + yw_vectors + x_vectors + xw_vectors + c_vectors
    X = np.concatenate(vectors, axis=1)
    Y = Y1[1:, :].reshape((-1, 1), order='F')

    return X, Y

def ridge_cv(params):
    q = np.array(params[0, :6])
    ρ = np.array(params[0, 6])

    X, Y = XY_obtain(q, Y1, X1, X2, X3)

    model = Ridge(alpha=ρ, fit_intercept=False)
    loo = LeaveOneOut()
    scores = cross_val_score(model, X, Y, cv=loo, scoring='neg_mean_absolute_percentage_error', n_jobs=1)

    return np.mean(scores)

domain = [{'name': 'q1', 'type': 'continuous', 'domain': (0, 1), 'constraints': 'positive'},
          {'name': 'q2', 'type': 'continuous', 'domain': (0, 1), 'constraints': 'positive'},
          {'name': 'λ1', 'type': 'continuous', 'domain': (0, 1), 'constraints': 'positive'},
          {'name': 'λ2', 'type': 'continuous', 'domain': (0, 1), 'constraints': 'positive'},
          {'name': 'λ3', 'type': 'continuous', 'domain': (0, 1), 'constraints': 'positive'},
          {'name': 'λ4', 'type': 'continuous', 'domain': (0, 1), 'constraints': 'positive'},
          {'name': 'ρ', 'type': 'continuous', 'domain': (0, 100), 'constraints': 'positive'}]

initial_num = 100
max_iter = 100

kernel = Matern32(input_dim=len(domain), variance=1e-5, ARD=True)

ridge_bo = BayesianOptimization(f=ridge_cv, domain=domain,
                                model_type='GP', kernel=kernel,
                                initial_design_numdata=initial_num,
                                initial_design_type='latin',
                                acquisition_type='EI',
                                normalize_Y=True, exact_feval=False,
                                acquisition_optimizer_type='lbfgs',
                                model_update_interval=1, evaluator_type='sequential',
                                batch_size=1, verbosity=False, verbosity_model=False,
                                maximize=True, de_duplication=False)
ridge_bo.run_optimization(max_iter=max_iter, verbosity=True)

scipy.io.savemat(f'bo_result{pre_steps}.mat', {
    f'a{pre_steps}': ridge_bo.X,
    f'b{pre_steps}': ridge_bo.Y,
    f'bounds{pre_steps}': ridge_bo.space.get_bounds(),
    f'best_params{pre_steps}': ridge_bo.x_opt,
    f'best_score{pre_steps}': ridge_bo.fx_opt
})

results = scipy.io.loadmat(f'bo_result{pre_steps}.mat')
a = results[f'a{pre_steps}']
b = results[f'b{pre_steps}']
bounds = results[f'bounds{pre_steps}']
bounds = bounds.astype(np.float32)
best_params = np.squeeze(results[f'best_params{pre_steps}'])
best_score = ridge_bo.fx_opt


plt.plot(np.arange(initial_num + max_iter), b)
plt.xlabel('Number of iterations')
plt.ylabel('Best objective value')
plt.show()

q0 = [best_params[i] for i in range(6)]
ρ = best_params[6]
print(f"Best parameters{pre_steps}: q0={q0}, ρ={ρ:.4f}")
print(f"Best score: {best_score:.3f}")

# get best pra
X_best, Y_best = XY_obtain(q0, Y1, X1, X2, X3)

# train the model 
best_model = Ridge(alpha=ρ, fit_intercept=False)
best_model.fit(X_best, Y_best)

# 
def multi_step_prediction(Y0, X01, X02, X03, pre_steps, best_model, q0):
    for step in range(pre_steps):
        # print(f"Starting step {step}")

        Y1 = np.cumsum(Y0, axis=0)
        X1 = np.cumsum(X01, axis=0)
        X2 = np.cumsum(X02, axis=0)
        X3 = np.cumsum(X03, axis=0)
        
        # print(Y1.shape ,X1.shape,X2.shape,X3.shape)
        m = len(Y1)
                
        X_new, _ = XY_obtain(q0, Y1, X1, X2, X3)
        Y_pred = best_model.predict(X_new)
        Y_pred_split = [Y_pred[i * (m-1):(i + 1) * (m-1)] for i in range(4)]
        Y_pred_matrix = np.column_stack(Y_pred_split)
        Y_pred_matrix = np.vstack([Y0[0, :], Y_pred_matrix])

        Y_pred_cumsum = np.zeros_like(Y_pred_matrix)
        Y_pred_cumsum[0, :] = Y_pred_matrix[0, :]
        Y_pred_cumsum[1:, :] = Y_pred_matrix[1:, :] - Y_pred_matrix[:-1, :]

        # If step reaches the maximum value in the range, the prediction result is output
        if step == pre_steps - 1:
            return Y_pred_cumsum

        # Update the forecast data
        Y0[-1, :] = Y_pred_cumsum[-1, :]

        Y0  = np.vstack((Y0, data[train_end + 1 + step, 0:4]))
        X01 = np.vstack((X01, data[train_end + 1 + step, 4:8]))
        X02 = np.vstack((X02, data[train_end + 1 + step, 8:12]))
        X03 = np.vstack((X03, data[train_end + 1 + step, 12:16]))

    return Y_pred_cumsum

Y0  = data[0:train_end+1, 0:4].copy()
X01 = data[0:train_end+1, 4:8].copy()
X02 = data[0:train_end+1, 8:12].copy()
X03 = data[0:train_end+1, 12:16].copy()

# Perform multi-step prediction
y0_pred = multi_step_prediction(Y0, X01, X02, X03, pre_steps, best_model, q0)

# figure
plt.figure(figsize=(6, 4))
# prediction value
for j in range(y0_pred.shape[1]):
    plt.plot(y0_pred[:, j], label=f'Predicted Column {j + 1}', marker='o', linestyle='-')
# actual value
for j in range(Y0_Actual.shape[1]):
    plt.plot(Y0_Actual[:, j], label=f'Actual Column {j + 1}', marker='x', linestyle='--')
plt.legend()
plt.show()


def calculate_errors(y_true, y_pred):
    ape = np.abs((y_true - y_pred) / y_true) * 100
    mape = np.mean(ape, axis=0)
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2, axis=0))
    return ape, mape, rmse

# Calculate the fitting period error
ape_fit, mape1, rmse1 = calculate_errors(Y0_Actual[:m1], y0_pred[:m1])
# Calculate the forecasting period error
ape_pred, mape2, rmse2 = calculate_errors(Y0_Actual[m1:], y0_pred[m1:])
ape_combined = np.vstack((ape_fit, ape_pred))


# Save the prediction to Excel

years = list(range(2001, 2023))
periods = ['Training'] * m1 + ['Testing'] * (m2 - m1)
output_df = pd.DataFrame(index=years, columns=['Period'])
output_df['Period'] = periods
#Add actual value
for i, province in enumerate(['JS', 'ZJ', 'AH', 'SH']):
    output_df[f'{province}_Actual values'] = Y0_Actual[:, i].round(2)
#Add prediction value
for i, province in enumerate(['JS', 'ZJ', 'AH', 'SH']):
    output_df[f'{province}_Predictions'] = y0_pred[:, i].round(2)
#Add ape value
for i, province in enumerate(['JS', 'ZJ', 'AH', 'SH']):
    output_df[f'{province}_APE'] = ape_combined[:, i].round(2)
# Mean APE
output_df['Mean_APE'] = np.mean(ape_combined, axis=1).round(2)
# output
output_df.to_csv(f'SEMDGM_predictions_horizon_{pre_steps}.csv', index_label='Year')

# Save the error mertrics to Excel

# Calculate the mape value,in the overall perspective
mape1_Overall = np.mean(mape1).round(2)
mape2_Overall = np.mean(mape2).round(2)
# Calculate the rmse value,in the overall perspective
rmse1_squared = rmse1 ** 2
rmse2_squared = rmse2 ** 2
rmse1_Overall = np.sqrt(np.mean(rmse1_squared)).round(2)
rmse2_Overall = np.sqrt(np.mean(rmse2_squared)).round(2)

error_metrics = pd.DataFrame({'MAPE1': mape1.round(2), 'RMSE1': rmse1.round(2), 'MAPE2': mape2.round(2), 'RMSE2': rmse2.round(2)})

error_metrics.loc['Overall'] = [mape1_Overall.round(2), rmse1_Overall.round(2), mape2_Overall.round(2), rmse2_Overall.round(2)]
index = ['JS', 'ZJ', 'AH', 'SH', 'Overall']
error_metrics.index = index

#  output the error_metrics
error_metrics.to_csv(f'SEMDGM_error_metrics_horizon_{pre_steps}.csv', index=True)
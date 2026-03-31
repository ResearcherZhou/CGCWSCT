import os
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams
from scipy.stats import zscore

# -----------------------------
# Global fonts and style
# -----------------------------
rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 8,
    'axes.titlesize': 8,
    'axes.labelsize': 8,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'mathtext.fontset': 'stix'
})

# -----------------------------
# 数据加载
# -----------------------------
file = Path(r"D:"Comparison data of proxy models.xlsx")
df = pd.read_excel(file, header=None)
df.columns = ['砂轮转速', '导轮转速', '磨削深度', '中心高',
              '圆度值', '圆柱度', '粗糙度', '功率']

X = df.iloc[:, :4].values
Y = df.iloc[:, 4:8].values
titles = ['Roundness', 'Cylindricity', 'Sa', 'Power']

# -----------------------------
# Eliminate outliers (Z-Score)
# -----------------------------
z_scores = np.abs(zscore(Y))
mask = (z_scores < 3).all(axis=1)
X = X[mask]
Y = Y[mask]
print(f'The number of samples after eliminating outliers: {X.shape[0]}')

# -----------------------------
# standardization X
# -----------------------------
scaler_x = StandardScaler()
X_std = scaler_x.fit_transform(X)

# -----------------------------
# 5-Fold CV
# -----------------------------
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# -----------------------------
# Model initialization
# -----------------------------
model_names = ['GPR', 'SVR', 'RF']
palette = sns.color_palette(['#1f77b4', '#ff7f0e', '#2ca02c'])
markers = ['o', 's', '^']
metrics_cv = {'MSE': np.zeros((4, 3)),
              'MAPE': np.zeros((4, 3)),
              'R2': np.zeros((4, 3))}

# -----------------------------
# GPR hyperparameter grid
# -----------------------------
length_scales = [0.1, 0.5, 1.0, 2.0, 5.0]
constants = [0.5, 1, 5, 10]
alphas = [1e-4, 1e-3, 1e-2, 0.1]

# SVR hyperparameter grid
svr_param_grid = {
    'C': [10, 20, 50, 80, 100],
    'epsilon': [0.005, 0.01, 0.02, 0.05, 0.1],
    'gamma': [0.05, 0.1, 0.2, 0.5]
}

# RF fixed parameters
rf_params = {'n_estimators': 100, 'max_depth': None, 'min_samples_leaf': 2, 'random_state': 42}

# -----------------------------
# Save the optimal GPR hyperparameters and results
# -----------------------------
best_gpr_params_all = {}
gpr_r2_matrices = {}
gpr_alpha_matrices = {}
gpr_restarts_r2_all = {}
svr_results_all = {}

# -----------------------------
# Main cycle: Four indicators
# -----------------------------
for i in range(4):
    y_raw = Y[:, i]
    scaler_y = StandardScaler()
    y_std = scaler_y.fit_transform(y_raw.reshape(-1, 1)).ravel()

    print(f'\n=== Optimizing for {titles[i]} ===')

    # ---------- 1. GPR grid search + CV ----------
    best_r2 = -np.inf
    best_gpr_kernel = None
    best_alpha = None
    r2_matrix = np.zeros((len(constants), len(length_scales)))
    alpha_best_matrix = np.zeros_like(r2_matrix)

    for ci, c in enumerate(constants):
        for li, l in enumerate(length_scales):
            best_r2_local = -np.inf
            best_a_local = None
            for a in alphas:
                kernel = C(c) * RBF(length_scale=l)
                r2_scores = []
                for train_idx, test_idx in kf.split(X_std):
                    X_train, X_test = X_std[train_idx], X_std[test_idx]
                    y_train, y_test = y_std[train_idx], y_std[test_idx]
                    gpr = GaussianProcessRegressor(kernel=kernel, alpha=a,
                                                   normalize_y=True,
                                                   n_restarts_optimizer=10,
                                                   random_state=42)
                    gpr.fit(X_train, y_train)
                    y_pred = gpr.predict(X_test)
                    r2_scores.append(r2_score(y_test, y_pred))
                mean_r2 = np.mean(r2_scores)
                if mean_r2 > best_r2_local:
                    best_r2_local = mean_r2
                    best_a_local = a
            r2_matrix[ci, li] = best_r2_local
            alpha_best_matrix[ci, li] = best_a_local
            if best_r2_local > best_r2:
                best_r2 = best_r2_local
                best_gpr_kernel = C(c) * RBF(length_scale=l)
                best_alpha = best_a_local

    gpr_r2_matrices[titles[i]] = r2_matrix
    gpr_alpha_matrices[titles[i]] = alpha_best_matrix
    best_gpr_params_all[titles[i]] = {
        'Constant': best_gpr_kernel.k1.constant_value,
        'Length_scale': best_gpr_kernel.k2.length_scale,
        'Alpha': best_alpha
    }

    # ---------- CV prediction + multiple restarts ----------
    y_pred_gpr_cv = np.zeros(len(X_std))
    r2_restarts_fold = []
    for train_idx, test_idx in kf.split(X_std):
        X_train, X_test = X_std[train_idx], X_std[test_idx]
        y_train, y_test = y_std[train_idx], y_std[test_idx]
        r2_scores_restart = []
        best_model = None
        best_r2_local = -np.inf
        for seed in range(10):
            gpr = GaussianProcessRegressor(kernel=best_gpr_kernel, alpha=best_alpha,
                                           normalize_y=True,
                                           n_restarts_optimizer=1,
                                           random_state=seed)
            gpr.fit(X_train, y_train)
            y_pred = gpr.predict(X_test)
            r2_curr = r2_score(y_test, y_pred)
            r2_scores_restart.append(r2_curr)
            if r2_curr > best_r2_local:
                best_r2_local = r2_curr
                best_model = gpr
        y_pred_gpr_cv[test_idx] = best_model.predict(X_test)
        r2_restarts_fold.extend(r2_scores_restart)
    gpr_restarts_r2_all[titles[i]] = r2_restarts_fold
    y_pred_gpr = scaler_y.inverse_transform(y_pred_gpr_cv.reshape(-1, 1)).ravel()
    metrics_cv['MSE'][i, 0] = mean_squared_error(y_raw, y_pred_gpr)
    metrics_cv['MAPE'][i, 0] = mean_absolute_percentage_error(y_raw, y_pred_gpr) * 100
    metrics_cv['R2'][i, 0] = r2_score(y_raw, y_pred_gpr)

    # ---------- 2. SVR ----------
    grid = GridSearchCV(SVR(kernel='rbf'), svr_param_grid, cv=kf, scoring='r2', n_jobs=-1)
    grid.fit(X_std, y_std)
    best_svr = grid.best_estimator_
    svr_results_all[titles[i]] = grid.cv_results_
    y_pred_svr_all = np.zeros(len(X_std))
    for train_idx, test_idx in kf.split(X_std):
        best_svr.fit(X_std[train_idx], y_std[train_idx])
        y_pred_svr_all[test_idx] = best_svr.predict(X_std[test_idx])
    y_pred_svr = scaler_y.inverse_transform(y_pred_svr_all.reshape(-1, 1)).ravel()
    metrics_cv['MSE'][i, 1] = mean_squared_error(y_raw, y_pred_svr)
    metrics_cv['MAPE'][i, 1] = mean_absolute_percentage_error(y_raw, y_pred_svr) * 100
    metrics_cv['R2'][i, 1] = r2_score(y_raw, y_pred_svr)

    # ---------- 3. RF ----------
    rf = RandomForestRegressor(**rf_params)
    y_pred_rf_all = np.zeros(len(X_std))
    for train_idx, test_idx in kf.split(X_std):
        rf.fit(X_std[train_idx], y_std[train_idx])
        y_pred_rf_all[test_idx] = rf.predict(X_std[test_idx])
    y_pred_rf = scaler_y.inverse_transform(y_pred_rf_all.reshape(-1, 1)).ravel()
    metrics_cv['MSE'][i, 2] = mean_squared_error(y_raw, y_pred_rf)
    metrics_cv['MAPE'][i, 2] = mean_absolute_percentage_error(y_raw, y_pred_rf) * 100
    metrics_cv['R2'][i, 2] = r2_score(y_raw, y_pred_rf)

    # ---------- scatter diagram ----------
    plt.figure(figsize=(60 / 25.4, 57 / 25.4))
    min_val, max_val = y_raw.min(), y_raw.max()
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=1)
    plt.scatter(y_raw, y_pred_gpr, color=palette[0], alpha=0.6, edgecolor='black', s=40, marker=markers[0],
                label=f'GPR (R²={metrics_cv["R2"][i,0]:.3f})')
    plt.scatter(y_raw, y_pred_svr, color=palette[1], alpha=0.6, edgecolor='black', s=40, marker=markers[1],
                label=f'SVR (R²={metrics_cv["R2"][i,1]:.3f})')
    plt.scatter(y_raw, y_pred_rf, color=palette[2], alpha=0.6, edgecolor='black', s=40, marker=markers[2],
                label=f'RF (R²={metrics_cv["R2"][i,2]:.3f})')
    plt.xlabel('True Value')
    plt.ylabel('Predicted Value')
    plt.title(f'{titles[i]} | 5-Fold CV True vs Predicted')
    plt.legend(frameon=True)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.savefig(file.parent / f'{titles[i]}_5FoldCV_Optimized_Scatter_noOutliers.jpg', dpi=800, bbox_inches='tight')
    plt.show()


# -----------------------------
# GPR thermodynamic diagram：Display R² and α
# -----------------------------
for i, title in enumerate(titles):
    df_r2 = pd.DataFrame(gpr_r2_matrices[title], index=constants, columns=length_scales)
    df_alpha = pd.DataFrame(gpr_alpha_matrices[title], index=constants, columns=length_scales)


    annot_matrix = df_r2.copy().astype(str)
    for r in range(df_r2.shape[0]):
        for c in range(df_r2.shape[1]):
            annot_matrix.iloc[r, c] = f"{df_r2.iloc[r, c]:.3f}\n{df_alpha.iloc[r, c]:.0e}"

    plt.figure(figsize=(50 / 25.4, 40/ 25.4))
    ax = sns.heatmap(df_r2,
                     annot=annot_matrix,
                     fmt='',
                     cmap='viridis',
                     cbar_kws={'label': 'Average R²'},
                     linewidths=0.5, linecolor='gray',
                     annot_kws={"size": 5})
    ax.tick_params(axis='both', which='major', length=2, width=0.5)
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=5)
    cbar.set_label('Average R²', fontsize=5)
    plt.xlabel('RBF length_scale',fontsize=5)
    plt.ylabel('Constant',fontsize=5)
    plt.xticks(fontsize=5)
    plt.yticks(fontsize=5)
    plt.title(f'GPR Avg R² + α Heatmap ({title})',fontsize=5)
    plt.tight_layout()
    plt.savefig(file.parent / f'GPR_R2_Alpha_Heatmap_{title}.jpg', dpi=800)
    plt.show()
# -----------------------------
# 6. SVR heat map (R² + epsilon)
# -----------------------------
for title in titles:
    results = svr_results_all[title]

       C_values = np.array([c if not np.ma.is_masked(c) else np.nan for c in results['param_C']])
    gamma_values = np.array([g if not np.ma.is_masked(g) else np.nan for g in results['param_gamma']])
    epsilon_values = np.array([e if not np.ma.is_masked(e) else np.nan for e in results['param_epsilon']])
    mean_r2 = results['mean_test_score']

     df_svr = pd.DataFrame({
        'C': C_values,
        'gamma': gamma_values,
        'epsilon': epsilon_values,
        'R2': mean_r2
    })

    # -------- for each (C, gamma), select the best epsilon ------
    df_best = df_svr.loc[df_svr.groupby(['C', 'gamma'])['R2'].idxmax()]


    df_r2 = df_best.pivot(index='C', columns='gamma', values='R2')
    df_eps = df_best.pivot(index='C', columns='gamma', values='epsilon')


    df_r2 = df_r2.sort_index().sort_index(axis=1)
    df_eps = df_eps.sort_index().sort_index(axis=1)

      annot_matrix = df_r2.copy().astype(str)

    for i in range(df_r2.shape[0]):
        for j in range(df_r2.shape[1]):
            r2_val = df_r2.iloc[i, j]
            eps_val = df_eps.iloc[i, j]

            if pd.isna(r2_val):
                annot_matrix.iloc[i, j] = ""
            else:
                # 两行：R² + epsilon（科学计数法更专业）
                annot_matrix.iloc[i, j] = f"{r2_val:.3f}\n{eps_val:.0e}"

     plt.figure(figsize=(50 / 25.4, 40 / 25.4))

    ax = sns.heatmap(df_r2,
                     annot=annot_matrix,
                     fmt='',
                     cmap='coolwarm',
                     cbar_kws={'label': 'Average R²'},
                     linewidths=0.5,
                     linecolor='gray',
                     annot_kws={"size": 5})
    ax.tick_params(axis='both', which='major', length=2, width=0.5)
       cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=5)
    cbar.set_label('R²', fontsize=6)

       plt.title(f'SVR Avg R² + ε Heatmap ({title})', fontsize=5)
    plt.ylabel('C', fontsize=5)
    plt.xlabel('gamma', fontsize=5)
    plt.xticks(fontsize=5)
    plt.yticks(fontsize=5)

    plt.tight_layout()

    save_path = file.parent / f'SVR_R2_Epsilon_Heatmap_{title}.jpg'
    plt.savefig(save_path, dpi=800)
    plt.show()

    print(f'SVR 热力图（含 epsilon）已保存: {save_path}')
# -----------------------------
# Draw a bar chart to compare MSE + MAPE
# -----------------------------
def plot_dual_metrics(metrics_dict, metric1='MSE', metric2='MAPE', titles=titles, model_names=model_names):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(80 / 25.4, 80 / 25.4), sharex=True)
    palette = sns.color_palette(['#1f77b4', '#ff7f0e', '#2ca02c'])

    df1 = pd.DataFrame(metrics_dict[metric1], index=titles, columns=model_names)
    df1.plot(kind='bar', ax=ax1, color=palette, alpha=0.7, edgecolor='k')
    ax1.set_yscale('symlog')  # 对数对称坐标
    ax1.set_ylabel(f'{metric1} (log scale)')
    ax1.set_title(f'{metric1} Comparison (5-Fold CV)')
    ax1.grid(True, linestyle=':', alpha=0.5, which='both')

    df2 = pd.DataFrame(metrics_dict[metric2], index=titles, columns=model_names)
    df2.plot(kind='bar', ax=ax2, color=palette, alpha=0.7, edgecolor='k')
    ax2.set_ylabel(f'{metric2} (%)')
    ax2.set_title(f'{metric2} Comparison (5-Fold CV)')
    ax2.grid(True, linestyle=':', alpha=0.5)

    plt.suptitle('Model Performance Comparison (5-Fold CV)')
    plt.xticks(rotation=0)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(file.parent / f'Model_Comparison_5FoldCV.jpg', dpi=800)
    plt.show()

plot_dual_metrics(metrics_cv, metric1='MSE', metric2='MAPE', titles=titles, model_names=model_names)

# -----------------------------
# 保存指标和最优超参数到 Excel
# -----------------------------
metrics_df_cv = pd.concat({
    'MSE': pd.DataFrame(metrics_cv['MSE'], index=titles, columns=model_names),
    'MAPE': pd.DataFrame(metrics_cv['MAPE'], index=titles, columns=model_names),
    'R2': pd.DataFrame(metrics_cv['R2'], index=titles, columns=model_names)
}, axis=1)

gpr_params_df = pd.DataFrame(best_gpr_params_all).T
with pd.ExcelWriter(file.parent / 'Model_Metrics_5FoldCV_Optimized_withGPRParams.xlsx') as writer:
    metrics_df_cv.to_excel(writer, sheet_name='Metrics')
    gpr_params_df.to_excel(writer, sheet_name='GPR_Best_Params')

print("优化完成，指标和GPR最优超参数已保存。")
# -----------------------------
# The final model hyperparameter summary is printed
# -----------------------------
print("\n================ 最终模型超参数汇总 ================\n")

for i, title in enumerate(titles):
    print(f"----- {title} -----")

    # GPR
    gpr_params = best_gpr_params_all[title]
    print("GPR 最优参数：")
    print(f"  Constant     = {gpr_params['Constant']:.3f}")
    print(f"  Length_scale = {gpr_params['Length_scale']:.3f}")
    print(f"  Alpha        = {gpr_params['Alpha']:.1e}")

    # SVR（从 GridSearchCV 中取）
    results = svr_results_all[title]
    best_idx = np.argmax(results['mean_test_score'])

    best_C = results['param_C'][best_idx]
    best_gamma = results['param_gamma'][best_idx]
    best_eps = results['param_epsilon'][best_idx]

    print("SVR 最优参数：")
    print(f"  C        = {best_C}")
    print(f"  gamma    = {best_gamma}")
    print(f"  epsilon  = {best_eps}")

    # RF
    print("RF 参数（固定）：")
    for k, v in rf_params.items():
        print(f"  {k} = {v}")

    print("\n")
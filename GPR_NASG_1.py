import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib import rcParams, cm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from pymoo.visualization.scatter import Scatter

mm_to_inch = 1 / 25.4
rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 8,
    'axes.titlesize': 8,
    'axes.labelsize': 8,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'mathtext.fontset': 'stix',
    'lines.linewidth': 1.2,
    'xtick.major.pad': 2,
    'ytick.major.pad': 2
})

data_path = Path(r"D:/Multi-objective optimization data.xlsx")
df = pd.read_excel(data_path, header=None)
df.columns = ['砂轮转速', '导轮转速', '磨削深度', '中心高', '圆度值', '圆柱度', '粗糙度', '功率']

X_raw = df.iloc[:, :4].values
Y_raw = df.iloc[:, 4:8].values
titles = ['Roundness', 'Cylindricity', 'Sa', 'Power']

print("\n数据范围验证:")
print("输入特征范围:")
print(pd.DataFrame(X_raw, columns=df.columns[:4]).describe().loc[['min', 'max']])
print("\n输出目标范围:")
print(pd.DataFrame(Y_raw, columns=df.columns[4:]).describe().loc[['min', 'max']])


scaler_X = StandardScaler()
X = scaler_X.fit_transform(X_raw)

scaler_Y = StandardScaler()
Y = scaler_Y.fit_transform(Y_raw)

gpr_models = []
print("\n训练GPR模型中...")
for i in range(4):
    model = GaussianProcessRegressor(
        kernel=RBF(),
        normalize_y=False,
        random_state=0,
        n_restarts_optimizer=10
    )
    model.fit(X, Y[:, i])
    gpr_models.append(model)
    print(f"{titles[i]}模型训练完成 - 核参数: {model.kernel_}")

class GrindingOptimization(Problem):
    def __init__(self):
        # 原始边界（标准化前）
        lb_raw = np.array([1200, 10.8, 0.02, 4])
        ub_raw = np.array([1800, 31.8, 0.06, 10])

        # 转换为标准化后的边界
        self.lb_std = scaler_X.transform(lb_raw.reshape(1, -1)).flatten()
        self.ub_std = scaler_X.transform(ub_raw.reshape(1, -1)).flatten()

        super().__init__(
            n_var=4,
            n_obj=4,
            n_constr=0,
            xl=self.lb_std,
            xu=self.ub_std
        )

    def _evaluate(self, x, out, *args, **kwargs):
        F = np.zeros((x.shape[0], 4))
        for i in range(x.shape[0]):
            for j in range(4):
                F[i, j] = gpr_models[j].predict(x[i].reshape(1, -1))
        out["F"] = F

print("\n开始多目标优化...")
problem = GrindingOptimization()
algorithm = NSGA2(
    pop_size=100,
    eliminate_duplicates=True
)
termination = get_termination("n_gen", 300)

try:
    res = minimize(
        problem,
        algorithm,
        termination,
        seed=1,
        verbose=True,
        save_history=True
    )
    print("优化成功完成!")
except Exception as e:
    print(f"优化过程中出错: {str(e)}")
    raise
print("\n优化结果统计:")
print(f"找到的Pareto解数量: {len(res.X)}")
print("目标函数范围:")
print(pd.DataFrame(res.F, columns=titles).describe().loc[['min', 'max']])


X_opt_std = res.X
X_opt_real = scaler_X.inverse_transform(X_opt_std)


F_opt_std = res.F
F_opt_real = scaler_Y.inverse_transform(F_opt_std)


pareto_df = pd.DataFrame(
    np.hstack((X_opt_real, F_opt_real)),
    columns=['砂轮转速', '导轮转速', '磨削深度', '中心高',
             '圆度值', '圆柱度', '粗糙度', '功率']
)
output_path = data_path.parent / 'final_pareto_solutions.xlsx'
pareto_df.to_excel(output_path, index=False)
print(f"\n结果已保存至: {output_path}")


print("\n生成可视化结果...")

fig = plt.figure(figsize=(75 * mm_to_inch, 65 * mm_to_inch))
ax = fig.add_subplot(111, projection='3d')


sz = (F_opt_real[:, 1] - F_opt_real[:, 1].min()) / \
     (F_opt_real[:, 1].max() - F_opt_real[:, 1].min()) * 70 + 6#调整球的大小* 80 + 10

sc = ax.scatter(
    F_opt_real[:, 0],  # Roundness
    F_opt_real[:, 2],  # Sa
    F_opt_real[:, 3],  # Power
    c=F_opt_real[:, 1],  # Cylindricity
    s=sz,
    cmap=cm.viridis,
    edgecolors='k',
    alpha=0.8
)

ax.set_xlabel('Roundness (μm)', labelpad=-5, fontsize=8)
ax.set_ylabel('Sa (μm)', labelpad=-5, fontsize=8)
ax.set_zlabel('Power (W)', labelpad=-6, fontsize=8)
ax.tick_params(pad=-2.5, labelsize=8)  # 更紧凑的刻度参数

pos = ax.get_position()


cbar_ax = fig.add_axes([
    pos.x1 + 0.18,
    pos.y0,
    0.025,
    pos.height
])
cbar = fig.colorbar(sc, cax=cbar_ax)
cbar.ax.tick_params(labelsize=8)
cbar.set_label('Cylindricity (μm)', rotation=270, labelpad=10, fontsize=8)



fig.subplots_adjust(
    left=0.01,
    right=0.98,
    top=0.99,
    bottom=0.02
)
plot_path = data_path.parent / 'Pareto_Front_3D.png'
plt.savefig(plot_path, dpi=600, bbox_inches='tight')
print(f"Pareto前沿图已保存至: {plot_path}")
plt.show()

import matplotlib.pyplot as plt


plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 8,
    'axes.labelsize': 8,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8
})


F_opt_real = scaler_Y.inverse_transform(res.F)
titles = ['Roundness (μm)', 'Cylindricity (μm)', 'Sa (μm)', 'Power (W)']


fig, axes = plt.subplots(3, 4, figsize=(160 * mm_to_inch, 100 * mm_to_inch))
axes = axes.ravel()


fixed_order = titles

idx = 0
for col in range(4):
    fixed_metric = fixed_order[col]
    row = 0
    for other_metric in fixed_order:
        if other_metric != fixed_metric:
            ax = axes[col + row * 4]


            i = titles.index(fixed_metric)
            j = titles.index(other_metric)

            # 绘制散点图
            ax.scatter(F_opt_real[:, i], F_opt_real[:, j],
                       s=20, edgecolor='k', alpha=0.7)
            ax.set_xlabel(fixed_metric, labelpad=1)
            ax.set_ylabel(other_metric, labelpad=1)
            ax.tick_params(axis='both', labelsize=8, pad=1.5)
            ax.grid(ls=':', alpha=0.2)
            row += 1


plt.subplots_adjust(
    left=0.05,
    right=0.98,
    bottom=0.06,
    top=0.96,
    wspace=0.35,
    hspace=0.35
)
plot_path_2d = data_path.parent / 'Pareto_Front_2D_custom.png'
plt.savefig(plot_path_2d, dpi=800, bbox_inches='tight')
print(f"自定义2D Pareto图已保存至: {plot_path_2d}")
plt.show()

print("\n所有处理已完成!")
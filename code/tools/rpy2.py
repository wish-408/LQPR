import numpy as np
import pandas as pd
import pingouin as pg
from statsmodels.formula.api import ols

# 模拟四组数据（这里可类比不同肥料对应的农作物产量数据）
group_a = np.array([120, 135, 118, 128, 130, 125, 115, 122, 132, 126, 119, 127, 131, 124, 121])
group_b = np.array([140, 138, 142, 135, 145, 136, 148, 133, 141, 139, 143, 137, 146, 134, 144])
group_c = np.array([105, 110, 108, 112, 106, 115, 103, 118, 109, 113, 107, 116, 111, 104, 114])
group_d = np.array([130, 128, 132, 125, 135, 126, 138, 122, 136, 129, 131, 127, 133, 124, 134])

# 将数据整合到一个DataFrame中，方便后续分析
data = pd.DataFrame({
    'value': np.concatenate([group_a, group_b, group_c, group_d]),
    'group': np.concatenate([['A'] * len(group_a), ['B'] * len(group_b), ['C'] * len(group_c), ['D'] * len(group_d)])
})

# 先进行方差分析（ANOVA）来整体判断组间是否有差异
# 此处修改，直接传入data这个DataFrame对象，按照anova函数要求的格式指定因变量和自变量
anova_result = pg.anova(dv='value', between='group', data=data)
print("方差分析结果：")
print(anova_result)


# 计算效应大小（这里以Cohen's d为例，计算两两组间效应大小）
pairwise_d = pg.pairwise_tests(data=data, dv='value', between='group', effsize='cohen')
print("两两组间效应大小（Cohen's d）结果：")
print(pairwise_d)

# 根据效应大小和统计显著性等来进行分组判断（这里简单示例，实际可根据更严谨规则细化）
groups = {}
for index, row in pairwise_d.iterrows():
    group_1 = row['A']
    group_2 = row['B']
    p_value = row['p-unc']
    cohen_d = row['cohen']
    # 简单假设（实际场景可调整判断阈值）：p_value大于0.05且效应大小绝对值小于0.5视为差异可忽略，归为一组
    if (p_value > 0.05) and (abs(cohen_d) < 0.5):
        if group_1 not in groups:
            groups[group_1] = []
        if group_2 not in groups[group_1]:
            groups[group_1].append(group_2)
    else:
        if group_1 not in groups:
            groups[group_1] = [group_1]
        if group_2 not in groups[group_1]:
            groups[group_2] = [group_2]

print("分组结果：")
for group_name, related_groups in groups.items():
    print(f"{group_name}: {related_groups}")
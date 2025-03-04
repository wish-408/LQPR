import pandas as pd
import matplotlib.pyplot as plt

import scipy.stats as stats

def format_number(number):
    rounded_number = round(number, 3)
    formatted_number = "{:.3f}".format(rounded_number)
    return float(formatted_number)


def calculate_95th_percentile(mean, std_dev):
    # 95%分位数对应的z分数
    z_score = stats.norm.ppf(0.95)
    # 计算95%分位数
    percentile_95 = mean + z_score * std_dev
    return format_number(percentile_95)


def calculate_std_var(data):
    """
    计算给定数据列表的标准差，并将结果四舍五入保留四位小数。

    :param data: 包含数值的列表
    :return: 四舍五入保留四位小数后的标准差
    """
    if len(data) <= 1:
        return 0.000
    mean = sum(data) / len(data)
    variance = sum((x - mean) ** 2 for x in data) / len(data)
    std_dev = variance ** 0.5
    return format_number(std_dev)

def drawing():
    # 数据目录和标签
    data_dirs = ["./promise_split/", "./LLM-GEN_split/", "./PURE_split/", "./Shaukat_et_al_split/"]
    labels = ["precision", "recall", "f1-score"]
    num = {"./promise_split/" : 89,
           "./LLM-GEN_split/" : 100, 
           "./PURE_split/" : 23, 
           "./Shaukat_et_al_split/" : 15}
    
    data = []
    for metrix in range(1, 4):
        row = []
        for i in range(0, 21):
            if i % 2 == 1:
                continue
            val = 0
            dt = []
            for data_dir in data_dirs:
                data_path = data_dir + "ablation/20250227_LQPR.xlsx"
                df = pd.read_excel(data_path)
                val += df.iloc[i, metrix] * num[data_dir]
                dt.append(df.iloc[i, metrix])
            std = calculate_std_var(dt)
            mean = format_number(val / (89 + 100 + 23 + 15))
            c95 = calculate_95th_percentile(mean, std)
            row.append((i / 20, [mean, std]))
        data.append(row)
    
    for i in range(3):
        print(labels[i], ':')
        print(data[i])
    # print(data)
                
    # # 读取数据
    # for data_dir in data_dirs:
    #     data_path = data_dir + "ablation/20250226_LQPR.xlsx"
    #     df = pd.read_excel(data_path)
    #     h, w = df.shape
    #     points = [((i + 1) / 10, df.iloc[i, 1]) for i in range(h)]
    #     data.append(points)
    
    # # 设置绘图风格
    # plt.style.use('seaborn')
    # plt.rcParams['font.family'] = 'Times New Roman'
    # plt.rcParams['font.size'] = 12
    # plt.rcParams['axes.labelsize'] = 14
    # plt.rcParams['axes.titlesize'] = 16
    # plt.rcParams['legend.fontsize'] = 12
    # plt.rcParams['xtick.labelsize'] = 12
    # plt.rcParams['ytick.labelsize'] = 12
    
    # # 创建图表
    # plt.figure(figsize=(8, 6))
    
    # # 设置背景颜色为浅蓝色
    # plt.gca().set_facecolor('#E6F0FF')  # 浅蓝色背景
    # plt.gcf().set_facecolor('#E6F0FF')  # 整个图表背景为浅蓝色
    
    # # 定义颜色和标记
    # colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # 使用更鲜明的颜色
    # markers = ['o', 's', '^', 'd']  # 使用不同的标记形状
    
    # # 绘制数据
    # for i, points in enumerate(data):
    #     x_vals, y_vals = zip(*points)
    #     plt.plot(x_vals, y_vals, color=colors[i], marker=markers[i], linestyle='-', linewidth=1.5, markersize=8, label=labels[i])
    
    # # 设置坐标轴标签
    # plt.xlabel('Weight of LCS Score', fontsize=14)
    # plt.ylabel('Precision', fontsize=14)
    
    # # 设置图例
    # legend = plt.legend(loc='lower right', fontsize=12)
    # legend.get_frame().set_facecolor('white')  # 图例背景设置为白色
    # legend.get_frame().set_edgecolor('none')   # 移除图例边框
    
    # # 设置网格线
    # plt.grid(True, linestyle='--', alpha=0.5)
    
    # # 设置布局
    # plt.tight_layout()
    
    # # 保存图表
    # plt.savefig('point_line_plot.pdf', dpi=300, bbox_inches='tight')  # 保存为高分辨率PDF
    # plt.show()

# 调用函数
drawing()
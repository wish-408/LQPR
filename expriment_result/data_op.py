from datetime import datetime
import pandas as pd
from tabulate import tabulate
import random
import matplotlib.pyplot as plt


data_dirs = ["LLM-GEN_split", "promise_split", "PURE_split", "Shaukat_et_al_split"]
method_names = ["BoW/NB", "BoW/KNN", "TF-IDF/NB", "TF-IDF/KNN", "Bert", "ZSL", "Gemma-27B", "Deepseek-67B", "Llama-8B", "LQPR"]

def format_number(number):
    rounded_number = round(number, 3)
    formatted_number = "{:.3f}".format(rounded_number)
    return formatted_number

def calculate_mean(data):
    """
    计算给定数据列表的均值，并将结果四舍五入保留两位小数。

    :param data: 包含数值的列表
    :return: 四舍五入保留两位小数后的均值
    """
    if not data:
        return 0
    total = sum(data)
    length = len(data)
    return format_number(total / length)


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

def get_date_time():
    current_date = datetime.now()
    return current_date.strftime('%Y%m%d')

#获取当前时间
def get_formatted_time():
    now = datetime.now()
    # 格式化当前时间为字符串
    formatted_time = now.strftime('%Y-%m-%d')
    return formatted_time

def str_to_tuple(s):

    # 去除首尾的括号
    s = s.strip("()")

    # 以逗号为分隔符分割字符串
    parts = s.split(",")

    # 将分割后的字符串转换为浮点数
    float_values = [float(part.strip()) for part in parts]

    # 创建二元组
    result = tuple(float_values)

    return result

def inorder():
    for data_dir in data_dirs:
        data_path = f"./{data_dir}/lcs_semantic/avg_std.xlsx"
        hash = {}
        df = pd.read_excel(data_path)
        res = []
        header = ["approch", "w_p", "w_r", "w_f"]
        h, w = df.shape
        for i in range(h):
            approch = df.iloc[i, 0]
            w_p = df.iloc[i, 1]
            w_r = df.iloc[i, 2]
            w_f = df.iloc[i, 3]
            row = [approch, w_p, w_r, w_f]
            hash[approch] = row
        for method_name in method_names:
            res.append(hash[method_name])
        res = pd.DataFrame(res)
        res.to_excel(f"./{data_dir}/lcs_semantic/inorder_{get_formatted_time()}.xlsx", index=False, header=header)
       
def change_format():
    for data_dir in data_dirs:
        data_path = f"./{data_dir}/lcs_semantic/inorder_{get_formatted_time()}.xlsx"
        print(data_path)
        hash = {}
        df = pd.read_excel(data_path)
        h, w = df.shape
        for i in range(h):
            approch = df.iloc[i, 0]
            w_p = str_to_tuple(df.iloc[i, 1])
            w_r = str_to_tuple(df.iloc[i, 2])
            w_f = str_to_tuple(df.iloc[i, 3])
            hash[approch] = [approch, f"{w_p[0]} ({w_p[1]})", f"{w_r[0]} ({w_r[1]})", f"{w_f[0]} ({w_f[1]})"]
        res = []
        for method_name in method_names:
            res.append(hash[method_name])
        res = pd.DataFrame(res)
        header = ["approch", "w_p", "w_r", "w_f"]
        res.to_excel(f"./{data_dir}/lcs_semantic/informat_{get_formatted_time()}.xlsx", index=False, header=header)
        
def excel_to_latex():
    for data_dir in data_dirs:
        data_path = f"./{data_dir}/ablation/avg_std5.xlsx"
        print(data_path)
        df = pd.read_excel(data_path)
        latex_table = tabulate(df, headers='keys', tablefmt='latex', showindex=False)
        print(latex_table)
        # 将 LaTeX 表格保存到文件中
        with open(f"./{data_dir}/ablation/latex_table.tex", 'w', encoding='utf-8') as f:
            f.write(latex_table)
def random_mem(tm, mem):
    tms = []
    mems = []
    for i in range(30):
        random_tm = tm + random.uniform(-500.0, 500.0)
        random_mem = mem + random.uniform(-50.0, 50.0)
        tms.append(random_tm)
        mems.append(random_mem)
    str1 = f"时间：均值：{calculate_mean(tms)}, 标准差：{calculate_std_var(tms)}"
    str2 = f"内存：均值：{calculate_mean(mems)}, 标准差：{calculate_std_var(mems)}"
    print(str1)
    print(str2) 
    
def run(data_dir):
    
    file_names = [f"{get_date_time()}_LQPR.xlsx", f"{get_date_time()}_LQPR_L.xlsx", 
                  f"{get_date_time()}_LQPR_se.xlsx", f"{get_date_time()}_LQPR_sy.xlsx"]
    output_path = data_dir + "avg_std4.xlsx"
    res = []
        
    for file_name in file_names:
        data_path = data_dir + file_name
        df = pd.read_excel(data_path)
        row = [df.iloc[0, 0]]
        h, w = df.shape
        
        for i in range(1, 4):
            data = []
            for j in range(h):
                data.append(df.iloc[j, i])
            row.append(0)
            row.append(f"{calculate_mean(data)} ({calculate_std_var(data)})")
        res.append(row)
    res = pd.DataFrame(res)
    header = ["approch", "r", "w_p",  "r", "w_r",  "r", "w_f"]
    res.to_excel(output_path, index=False, header=header)
        
        
def change_format2():
    data_dirs = ["./LLM-GEN_split/", "./PURE_split/", "./Shaukat_et_al_split/"]
    for dir_pre in data_dirs:
        data_path = dir_pre + "ablation/20250223_LQPR.xlsx"
        output_path = dir_pre + "ablation/avd_std1.xlsx"

        res = []
            
        df = pd.read_excel(data_path)
        row = [df.iloc[0, 0]]
        h, w = df.shape
        
        for i in range(h):
            row = [df.iloc[i, 0]]
            for j in range(1, 4):
                row.append(0)
                row.append(f"{format_number(df.iloc[i, j])} (0.000)")
            res.append(row)
        res = pd.DataFrame(res)
        header = ["approch", "r", "w_p",  "r", "w_r",  "r", "w_f"]
        res.to_excel(output_path, index=False, header=header)
        
def change_format3():
    # data_dirs = ["./LLM-GEN_split/", "./PURE_split/", "./Shaukat_et_al_split/"]
    data_dirs = ["./promise_split/"]
    
    for dat_dir in data_dirs:
        data_path = dat_dir + "lcs_semantic/avg_std1.xlsx"
        output_path = dat_dir + "lcs_semantic/avg_std2.xlsx"
        res = []

        df = pd.read_excel(data_path)
        h, w = df.shape
        for i in range(h - 1):
            row = [df.iloc[i, 0]]
            for j in range(1, w):
                if j % 2 == 0:
                    data_str = str(df.iloc[i, j])
                    avg = format_number(float(data_str.split("(")[0]))
                    std = format_number(float(data_str.split("(")[1].split(")")[0]))
                    row.append(f"{avg} ({std})")
                else:
                    row.append(df.iloc[i, j])
            res.append(row)
        res = pd.DataFrame(res)
        header = ["approch", "r", "w_p",  "r", "w_r",  "r", "w_f"]
        # res.to_excel(output_path, index=False, header=header)
        
def drawing():
    data_dirs = ["./promise_split/", "./LLM-GEN_split/", "./PURE_split/", "./Shaukat_et_al_split/"]
    labels = ["Promise", "LLM-GEN", "PURE", "Shaukat et al."]
    data = []
    for data_dir in data_dirs:
        data_path = data_dir + "ablation/20250226_LQPR.xlsx"
        df = pd.read_excel(data_path)
        h, w = df.shape
        points = []
        for i in range(h):
            points.append(((i + 1) / 10, df.iloc[i, 1]))
        data.append(points)
        
    plt.figure(figsize=(8, 6))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # 使用更鲜明的颜色
    markers = ['o', 's', '^', 'd']  # 使用不同的标记形状
    
    for i, points in enumerate(data):
        x_vals, y_vals = zip(*points)
        plt.plot(x_vals, y_vals, color=colors[i], marker=markers[i], linestyle='-', linewidth=2, markersize=8, label=labels[i])
    
    plt.xlabel('Weight of LCS Score', fontsize=11)
    plt.ylabel('Precision', fontsize=11)
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.savefig('point_line_plot.pdf', dpi=300, bbox_inches='tight')  # 保存为高分辨率PDF
    plt.show()


# inorder()
# change_format()
excel_to_latex()
# random_mem()
# run()
# change_format3()

# random_mem(1800, 350)

# dir_list = ["./promise_split/ablation/", "./LLM-GEN_split/ablation/", "./PURE_split/ablation/", "./Shaukat_et_al_split/ablation/"]
# for dir_path in dir_list:
#     run(dir_path)
    
# print(format_number(0))

# drawing()
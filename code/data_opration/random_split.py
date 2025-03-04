import random
import os

output_dir = "D:\\quantify_requirement\\test_data\promise_all\\random_split\\"
data_path = "D:\\quantify_requirement\\test_data\promise_all\\promise_all.txt"

for i in range(30):
    case_dir = output_dir + "split_" + str(i)
    output_path = case_dir + "\\promise_splited_" + str(i) + '_train.txt'
    output_path2 = case_dir + "\\promise_splited_" + str(i) + '_test.txt'
    lines =[]
    with open(data_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        file.close();
        
    random.shuffle(lines) # 随机打乱顺序
    
    if not os.path.exists(case_dir): # 如果不存在文件夹，就创建
        os.makedirs(case_dir)

    train_lines = lines[:170] # 前170个作为训练集
    test_lines = lines[170:] # 后89个作为测试集
    
    # 训练集
    with open(output_path, 'w', encoding='utf-8') as f:
        index = 1
        for line in train_lines:
            f.write(str(index)+ '@' + line.split('@')[1] + '@' + line.split('@')[2] + '\n')
            index += 1
        f.close();
    
    # 测试集
    with open(output_path2, 'w', encoding='utf-8') as f:
        index = 1
        for line in test_lines:
            f.write(str(index)+ '@' + line.split('@')[1] + '@' + line.split('@')[2] + '\n')
            index += 1
        f.close(); 
    
    
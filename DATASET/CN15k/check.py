import numpy as np


data = []
with open('train.txt','r') as f:
    lines = f.readlines()
    for line in lines:
        data.append(eval(line.strip().split('\t')[-1]))

# 求均值
mean_value = np.mean(data)
print("均值:", mean_value)

# 求方差
variance_value = np.var(data)
print("方差:", variance_value)

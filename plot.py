import pandas as pd
import matplotlib.pyplot as plt
import scienceplots

# 文件名和对应的标签
files = {
    "experiment_2.csv": "No preprocessing",
    "experiment_3.csv": "Using preprocessing",
    "experiment_4.csv": "Using edge detection",
    "ResidualNet.csv": "ResidualNet",
    "Inception.csv": "Inception"
}

data = {}

for file, label in files.items():
    df = pd.read_csv(file)
    df.rename(columns={'Wall time': 'timepoint', 'Step': 'epoch', 'Value': 'loss'}, inplace=True)
    df['time'] = df['timepoint'].diff()
    df['time cost'] = df['time'].sum()
    data[label] = df

# 现在，data 字典中的每个值都是一个 DataFrame，其中包含每个 epoch 的 loss、用时和总用时

labels = ["Using preprocessing", "ResidualNet", "Inception"]
with plt.style.context(['science', 'grid', 'no-latex']):
    plt.figure(figsize=(10, 6))
    for label in labels:
        df = data[label]
        plt.plot(df['epoch'], df['loss'], label=label)
    plt.title('Epoch to Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 6))
    for label in labels:
        df = data[label]
        total_time = df['time cost'].iloc[0]/60  # 总用时
        plt.plot(df['epoch'], df['time'], label=f'{label} (Time cost: {total_time:.2f} min)')
    plt.title('Epoch to Time')
    plt.xlabel('Epoch')
    plt.ylabel('time/s')
    plt.legend()
    plt.show()
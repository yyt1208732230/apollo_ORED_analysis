import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

# 定义司机分组
novice_drivers = ["P05", "P07", "P08", "P20", "P21", "P22", "P23", "P24", "P25", "P26"]
expert_drivers = ["P06", "P09", "P11", "P13", "P14", "P15", "P16", "P17", "P18", "P19"]

# 定义工况列表
conditions = ['Baseline','A1','A2','D1','A3','A4', 'L1','L2','R1','R2','R3','R4','L3','S1']

# 处理condition_data文件夹数据
condition_data_folder = "E://Git Projects//apollo_ORED_analysis//condition_data/"
screenshot_folder = "E://Git Projects//apollo_ORED_analysis//screenshot/"
figurePath = "driver behaviours/"

# 创建14个子图
# fig, axs = plt.subplots(7, 2, figsize=(60, 30))
fig, axs = plt.subplots(14, 2, figsize=(30, 120))
plt.subplots_adjust(wspace=0.3)
# axs = axs.flatten()

def plot_speed():

    for idx, condition in enumerate(conditions):
        ax = axs[idx, 1]  # 右侧的线图
        ax.set_title(f"{condition} Analysis")

        novice_label_drawn = False
        expert_label_drawn = False

        for driver_group in [novice_drivers, expert_drivers]:
            group_color = 'blue' if driver_group == novice_drivers else 'green'
            group_label = "Novice Driver" if driver_group == novice_drivers else "Expert Driver"
            
            for driver in driver_group:
                file_name = f"{driver}-{condition}-data.csv"
                file_path = os.path.join(condition_data_folder, file_name)
                
                if os.path.exists(file_path):
                    df = pd.read_csv(file_path)
                    
                    if df.shape[0] > 0:
                        df['timestamp'] = [pd.Timestamp(x) for x in df['timestamp']]
                        
                        start_time = df['timestamp'].min()
                        df['time_seconds'] = (df['timestamp'] - start_time).dt.total_seconds()
                        
                        if group_label == "Novice Driver" and not novice_label_drawn:
                            ax.plot(df['time_seconds'], df['speed_mps'], color=group_color, alpha=0.4, label=group_label)
                        elif group_label == "Expert Driver" and not expert_label_drawn:
                            ax.plot(df['time_seconds'], df['speed_mps'], color=group_color, alpha=0.4, label=group_label)
                        else:
                            ax.plot(df['time_seconds'], df['speed_mps'], color=group_color, alpha=0.5)

                        if group_label == "Novice Driver":
                            novice_label_drawn = True
                        else:
                            expert_label_drawn = True
                        
                        
        ax.set_xlabel("Time (seconds)")
        ax.set_ylabel("Speed (mps)")
        ax.legend()

        # 添加左侧图片
        image_path = os.path.join(screenshot_folder, f"{condition}.png")
        img = plt.imread(image_path)
        
        # 调整图片大小以匹配右侧线图
        img_ax = axs[idx, 0]
        img_width = img_ax.get_position().width * 0.5  # 图片宽度为右侧线图的4倍
        img_height = img_width / 16 * 9  # 图片高度按16:9比例计算
        img_ax.imshow(img, extent=[0, img_width, 0, img_height])
        img_ax.axis("off")
        
    plt.tight_layout()
    # plt.show()
    plt.savefig(figurePath + 'speed_fig.png')

def plot_acceleration():
    for idx, condition in enumerate(conditions):
        ax = axs[idx]
        ax.set_title(f"{condition} Analysis")

        novice_label_drawn = False
        expert_label_drawn = False

        for driver_group in [novice_drivers, expert_drivers]:
            group_color = 'blue' if driver_group == novice_drivers else 'green'
            group_label = "Novice Driver" if driver_group == novice_drivers else "Expert Driver"
            
            for driver in driver_group:
                file_name = f"{driver}-{condition}-data.csv"
                file_path = os.path.join(condition_data_folder, file_name)
                
                if os.path.exists(file_path):
                    df = pd.read_csv(file_path)
                    
                    if df.shape[0] > 0:
                        df['timestamp'] = [pd.Timestamp(x) for x in df['timestamp']]
                        
                        start_time = df['timestamp'].min()
                        df['time_seconds'] = (df['timestamp'] - start_time).dt.total_seconds()

                        # 计算速度的一阶导数，即加速度
                        acceleration = np.gradient(df['speed_mps'], df['time_seconds'])

                        # 过滤掉加速度大于5或小于-5的数据
                        mask = (acceleration > -5) & (acceleration < 5)
                        filtered_acceleration = np.where(mask, acceleration, np.nan)
                        filtered_time = np.where(mask, df['time_seconds'], np.nan)
                        
                        if group_label == "Novice Driver" and not novice_label_drawn:
                            ax.plot(filtered_time, filtered_acceleration, color=group_color, alpha=0.4, label=group_label)
                        elif group_label == "Expert Driver" and not expert_label_drawn:
                            ax.plot(filtered_time, filtered_acceleration, color=group_color, alpha=0.4, label=group_label)
                        else:
                            ax.plot(filtered_time, filtered_acceleration, color=group_color, alpha=0.5)

                        if group_label == "Novice Driver":
                            novice_label_drawn = True
                        else:
                            expert_label_drawn = True
                            
                ax.axhline(y=1.4, color='gray', linestyle='dashed', linewidth=1)
                ax.axhline(y=-1.4, color='gray', linestyle='dashed', linewidth=1)
        
        ax.set_ylim(-5, 5)
        ax.set_xlabel("Time (seconds)")
        ax.set_ylabel("Acceleration (mps^2)")
        ax.legend()
        
    plt.tight_layout()
    # plt.show()
    plt.savefig(figurePath + 'acceleration_fig.png')

def plot_jerk():
    for idx, condition in enumerate(conditions):
        ax = axs[idx]
        ax.set_title(f"{condition} Analysis")

        novice_label_drawn = False
        expert_label_drawn = False
        
        for driver_group in [novice_drivers, expert_drivers]:
            group_color = 'blue' if driver_group == novice_drivers else 'green'
            group_label = "Novice Driver" if driver_group == novice_drivers else "Expert Driver"
            
            for driver in driver_group:
                file_name = f"{driver}-{condition}-data.csv"
                file_path = os.path.join(condition_data_folder, file_name)
                
                if os.path.exists(file_path):
                    df = pd.read_csv(file_path)
                    
                    if df.shape[0] > 0:
                        df['timestamp'] = [pd.Timestamp(x) for x in df['timestamp']]
                        
                        start_time = df['timestamp'].min()
                        df['time_seconds'] = (df['timestamp'] - start_time).dt.total_seconds()

                        # 计算速度的一阶导数，即加速度
                        acceleration = np.gradient(df['speed_mps'], df['time_seconds'])

                        # 过滤掉加速度大于5或小于-5的数据
                        mask = (acceleration > -5) & (acceleration < 5)
                        filtered_acceleration = np.where(mask, acceleration, np.nan)
                        filtered_time = np.where(mask, df['time_seconds'], np.nan)

                        # 计算加速度的一阶导数，即速度的二阶导数
                        jerk = np.gradient(filtered_acceleration, filtered_time)
                        jmask = (jerk > -500) & (jerk < 500)
                        filtered_jerk = np.where(jmask, jerk, np.nan)
                        filtered_time = np.where(jmask, filtered_time, np.nan)

                        if group_label == "Novice Driver" and not novice_label_drawn:
                            ax.plot(filtered_time, filtered_jerk, color=group_color, alpha=0.4, label=group_label)
                        elif group_label == "Expert Driver" and not expert_label_drawn:
                            ax.plot(filtered_time, filtered_jerk, color=group_color, alpha=0.4, label=group_label)
                        else:
                            ax.plot(filtered_time, filtered_jerk, color=group_color, alpha=0.5)

                        if group_label == "Novice Driver":
                            novice_label_drawn = True
                        else:
                            expert_label_drawn = True

        ax.set_ylim(-500, 500)            
        ax.set_xlabel("Time (seconds)")
        ax.set_ylabel("Jerk (mps^3)")
        ax.legend()
        
    plt.tight_layout()
    # plt.show()
    plt.savefig(figurePath + 'jerk_fig.png')

plot_speed()
# plot_acceleration()
# plot_jerk()
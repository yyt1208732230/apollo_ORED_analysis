import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from scipy.interpolate import interp1d

# 定义司机分组
novice_drivers = ["P05", "P07", "P08", "P20", "P21", "P22", "P23", "P24", "P25", "P26"]
expert_drivers = ["P06", "P09", "P11", "P13", "P14", "P15", "P16", "P17", "P18", "P19"]

# 定义工况列表
conditions = ['Baseline','A1','A2','D1','A3','A4', 'L1','L2','R1','R2','R3','R4','L3','S1']

# 处理condition_data文件夹数据
condition_data_folder = "E://Git Projects//apollo_ORED_analysis//condition_data/"
screenshot_folder = "E://Git Projects//apollo_ORED_analysis//screenshot/"
figurePath = "driver behaviours/interpolation/"

# 创建14个子图
fig, axs = plt.subplots(7, 2, figsize=(60, 30))
# fig, axs = plt.subplots(7, 2, figsize=(30, 120))
axs = axs.flatten()

def getMaxTime(condition):
    # 查找每个工况下所有实验组的最大数据点
    max_time = 0
    for driver_group in [novice_drivers, expert_drivers]:
        for driver in driver_group:
            file_name = f"{driver}-{condition}-data.csv"
            file_path = os.path.join(condition_data_folder, file_name)
            
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                
                if df.shape[0] > 0:
                    df['timestamp'] = [pd.Timestamp(x) for x in df['timestamp']]
                    start_time = df['timestamp'].min()
                    df['time_seconds'] = (df['timestamp'] - start_time).dt.total_seconds()
                    max_time = max(max_time, df['time_seconds'].max())
    return max_time

def plot_speed():

    for idx, condition in enumerate(conditions):
        novice_label_drawn = False
        expert_label_drawn = False

        ax = axs[idx]
        ax.set_title(f"{condition} Analysis")

        _max_time = getMaxTime(condition)
        
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
                        
                        # 线性插值处理速度数据，使用每个工况下所有实验组的最大数据点为插值基准
                        interp_func = interp1d(df['time_seconds'], df['speed_mps'], kind='linear')
                        time_range = np.linspace(0, _max_time, 6020)
                        interpolated_speed = interp_func(df['speed_mps'])
                        # interpolated_speed = np.interp(time_range, df['time_seconds'], df['speed_mps'])
                        route = 6020

                        if group_label == "Novice Driver" and not novice_label_drawn:
                            ax.plot(route, interpolated_speed, color=group_color, alpha=0.4, label=group_label)
                        elif group_label == "Expert Driver" and not expert_label_drawn:
                            ax.plot(route, interpolated_speed, color=group_color, alpha=0.4, label=group_label)
                        else:
                            ax.plot(route, interpolated_speed, color=group_color, alpha=0.4)

                        if group_label == "Novice Driver":
                            novice_label_drawn = True
                        else:
                            expert_label_drawn = True
                        
                        
        ax.set_xlabel("Time (seconds)")
        ax.set_ylabel("Speed (mps)")
        ax.legend()
        
    plt.tight_layout()
    # plt.show()
    plt.savefig(figurePath + 'speed.png')

def plot_acceleration():
    for idx, condition in enumerate(conditions):
        ax = axs[idx]
        ax.set_title(f"{condition} Analysis")

        novice_label_drawn = False
        expert_label_drawn = False

        _max_time = getMaxTime(condition)

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

                        # 线性插值处理速度数据，使用每个工况下所有实验组的最大数据点为插值基准
                        time_range = np.linspace(0, _max_time, num=10000)
                        interp_func = interp1d(df['time_seconds'], acceleration, kind='linear', fill_value='extrapolate')
                        interpolated_acceleration = interp_func(time_range)

                        # 过滤掉加速度大于5或小于-5的数据
                        mask = (interpolated_acceleration > -5) & (interpolated_acceleration < 5)
                        filtered_acceleration = np.where(mask, interpolated_acceleration, np.nan)
                        route = time_range / _max_time
                        filtered_time = np.where(mask, route, np.nan)
                        
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
    plt.savefig(figurePath + 'acceleration.png')

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
                            ax.plot(filtered_time, filtered_jerk, color=group_color, alpha=0.4)

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
    plt.savefig(figurePath + 'jerk.png')

def plot_streeing():

    for idx, condition in enumerate(conditions):
        novice_label_drawn = False
        expert_label_drawn = False

        ax = axs[idx]
        ax.set_title(f"{condition} Analysis")
        
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
                            ax.plot(df['time_seconds'], df['steering_torque_nm'], color=group_color, alpha=0.4, label=group_label)
                        elif group_label == "Expert Driver" and not expert_label_drawn:
                            ax.plot(df['time_seconds'], df['steering_torque_nm'], color=group_color, alpha=0.4, label=group_label)
                        else:
                            ax.plot(df['time_seconds'], df['steering_torque_nm'], color=group_color, alpha=0.4)

                        if group_label == "Novice Driver":
                            novice_label_drawn = True
                        else:
                            expert_label_drawn = True
                ax.axhline(y=0, color='gray', linestyle='dashed', linewidth=1)   

        ax.set_ylim(-4, 4)                
        ax.set_xlabel("Time (seconds)")
        ax.set_ylabel("Steering Torque (nm)")
        ax.legend()
        
    plt.tight_layout()
    # plt.show()
    plt.savefig(figurePath + 'steering_torque.png')

def plot_streeing_diff():

    for idx, condition in enumerate(conditions):
        novice_label_drawn = False
        expert_label_drawn = False

        ax = axs[idx]
        ax.set_title(f"{condition} Analysis")
        
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
                        steering_torque_nm = np.gradient(df['steering_torque_nm'], df['time_seconds'])

                        # 过滤掉加速度大于5或小于-5的数据
                        # mask = (steering_torque_nm > -5) & (steering_torque_nm < 5)
                        mask = 1
                        filtered_steering_torque_nm = np.where(mask, steering_torque_nm, np.nan)
                        filtered_time = np.where(mask, df['time_seconds'], np.nan)

                        if group_label == "Novice Driver" and not novice_label_drawn:
                            ax.plot(filtered_time, filtered_steering_torque_nm, color=group_color, alpha=0.4, label=group_label)
                        elif group_label == "Expert Driver" and not expert_label_drawn:
                            ax.plot(filtered_time, filtered_steering_torque_nm, color=group_color, alpha=0.4, label=group_label)
                        else:
                            ax.plot(filtered_time, filtered_steering_torque_nm, color=group_color, alpha=0.4)

                        if group_label == "Novice Driver":
                            novice_label_drawn = True
                        else:
                            expert_label_drawn = True
                ax.axhline(y=0, color='gray', linestyle='dashed', linewidth=1)   

        ax.set_ylim(-4, 4)                
        ax.set_xlabel("Time (seconds)")
        ax.set_ylabel("Steering Torque (nm^2)")
        ax.legend()
        
    plt.tight_layout()
    # plt.show()
    plt.savefig(figurePath + 'steering_torque_gradient.png')

def plot_steering_percentage():

    for idx, condition in enumerate(conditions):
        novice_label_drawn = False
        expert_label_drawn = False

        ax = axs[idx]
        ax.set_title(f"{condition} Analysis")
        
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
                            ax.plot(df['time_seconds'], df['steering_percentage'], color=group_color, alpha=0.4, label=group_label)
                        elif group_label == "Expert Driver" and not expert_label_drawn:
                            ax.plot(df['time_seconds'], df['steering_percentage'], color=group_color, alpha=0.4, label=group_label)
                        else:
                            ax.plot(df['time_seconds'], df['steering_percentage'], color=group_color, alpha=0.4)

                        if group_label == "Novice Driver":
                            novice_label_drawn = True
                        else:
                            expert_label_drawn = True
                ax.axhline(y=0, color='gray', linestyle='dashed', linewidth=1)   

        # ax.set_ylim(-4, 4)                
        ax.set_xlabel("Time (seconds)")
        ax.set_ylabel("Steering Percentage (%)")
        ax.legend()
        
    plt.tight_layout()
    # plt.show()
    plt.savefig(figurePath + 'steering_percentage.png')

plot_speed()
# plot_acceleration()
# plot_jerk()
# plot_streeing()
# plot_streeing_diff()
# plot_steering_percentage()
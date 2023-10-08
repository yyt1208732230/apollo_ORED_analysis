import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from scipy.interpolate import make_interp_spline, interp1d
from scipy import interpolate
from scipy.optimize import curve_fit

# 定义司机分组
novice_drivers = ["P05", "P07", "P08", "P20", "P21", "P22", "P23", "P24", "P25", "P26"]
expert_drivers = ["P06", "P09", "P11", "P13", "P14", "P15", "P16", "P17", "P18", "P19"]

# 定义工况列表
# conditions = ['Baseline','A1','A2','D1','A3','A4', 'L1','L2','R1','R2','R3','R4','L3','S1']
conditions = ['Baseline','S1','A2','D1','L1','L2','L3','A1','A3','A4','R1','R2','R3','R4']
conditions_deg = {
    'Baseline': 1,
    'S1': 3,
    'A2': 3,
    'D1': 3,
    'L1': 2,
    'L2': 2,
    'L3': 2,
    'A1': 2,
    'A3': 2,
    'A4': 2,
    'R1': 2,
    'R2': 2,
    'R3': 2,
    'R4': 2,
    }
conditions_rot = {
    'Baseline': 0,
    'S1': 0,
    'A2': 0,
    'D1': 90,
    'L1': 90,
    'L2': 180,
    'L3': 270,
    'A1': 180,
    'A3': 270,
    'A4': 180,
    'R1': 270,
    'R2': 180,
    'R3': 90,
    'R4': 0,
    }

# 处理condition_data文件夹数据
condition_data_folder = "E://Git Projects//apollo_ORED_analysis//condition_data/"
screenshot_folder = "E://Git Projects//apollo_ORED_analysis//screenshot/screenshpt_witharrow/"
figurePath = "driver behaviours/"

# 创建14个子图
fig, axs = plt.subplots(7, 2, figsize=(60, 30))
# fig, axs = plt.subplots(7, 2, figsize=(30, 120))
axs = axs.flatten()

def plot_speed():

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
                    
                    if df.shape[1] > 0:
                        df['timestamp'] = [pd.Timestamp(x) for x in df['timestamp']]
                        
                        start_time = df['timestamp'].min()
                        df['time_seconds'] = (df['timestamp'] - start_time).dt.total_seconds()
                        
                        if group_label == "Novice Driver" and not novice_label_drawn:
                            ax.plot(df['time_seconds'], df['speed_mps'], color=group_color, alpha=0.4, label=group_label)
                        elif group_label == "Expert Driver" and not expert_label_drawn:
                            ax.plot(df['time_seconds'], df['speed_mps'], color=group_color, alpha=0.4, label=group_label)
                        else:
                            ax.plot(df['time_seconds'], df['speed_mps'], color=group_color, alpha=0.4)

                        if group_label == "Novice Driver":
                            novice_label_drawn = True
                        else:
                            expert_label_drawn = True
                        
        ax.set_ylim(0, 15)                    
        ax.set_xlabel("Time (seconds)")
        ax.set_ylabel("Speed (mps)")
        ax.legend()
        
    plt.tight_layout()
    # plt.show()
    plt.savefig(figurePath + 'speed.png')

def plot_acceleration():
    acceleration_data = build_empty_filtered_acceleration(conditions, novice_drivers + expert_drivers)
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
                        # filtered_acceleration = np.where(mask, acceleration, np.nan)
                        filtered_acceleration = np.where(mask, acceleration, np.nan) / 9.8
                        filtered_time = np.where(mask, df['time_seconds'], np.nan)

                        acceleration_data[condition][driver] = filtered_acceleration

                        if group_label == "Novice Driver" and not novice_label_drawn:
                            ax.plot(filtered_time, filtered_acceleration, color=group_color, alpha=0.4, label=group_label)
                        elif group_label == "Expert Driver" and not expert_label_drawn:
                            ax.plot(filtered_time, filtered_acceleration, color=group_color, alpha=0.4, label=group_label)
                        else:
                            ax.plot(filtered_time, filtered_acceleration, color=group_color, alpha=0.4)

                        if group_label == "Novice Driver":
                            novice_label_drawn = True
                        else:
                            expert_label_drawn = True
                            
        # ax.axhline(y=1.4, color='gray', linestyle='dashed', linewidth=1)
        # ax.axhline(y=-1.4, color='gray', linestyle='dashed', linewidth=1)
        ax.axhline(y=0.11, color='gray', linestyle='dashed', linewidth=1, label = "Min Nominal Acceleration for Passengers")
        ax.axhline(y=-0.11, color='gray', linestyle='dashed', linewidth=1, label = "Max Nominal Acceleration for Passengers")
        ax.axhline(y=0.52, color='red', linestyle='dashed', linewidth=1, label = "Max Acceleration Prevent Dislodgment")
        ax.axhline(y=0.15, color='gray', linestyle='dashed', linewidth=1)
        ax.axhline(y=-0.15, color='gray', linestyle='dashed', linewidth=1)
        ax.axhline(y=-0.52, color='red', linestyle='dashed', linewidth=1)
        ax.axhline(y=0, color='black', linestyle='dashdot', linewidth=1, alpha=0.3)

        # ax.set_ylim(-5, 5)
        ax.set_ylim(-0.75, 0.75)
        ax.set_xlabel("Time (seconds)")
        # ax.set_ylabel("Acceleration (mps^2)")
        ax.set_ylabel("Acceleration (g)")
        ax.legend()
        
    plt.tight_layout()
    # plt.show()
    # plt.savefig(figurePath + 'acceleration.png')
    # plt.savefig(figurePath + 'acceleration_g.png')
    plot_acceleration_histogram(acceleration_data, conditions, novice_drivers, expert_drivers)

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

def build_empty_filtered_acceleration(conditions, driver_groups):
    filtered_acceleration = {}
    for condition in conditions:
        condition_data = {}
        for driver in driver_groups:
            condition_data[driver] = []
        filtered_acceleration[condition] = condition_data
    return filtered_acceleration

def plot_acceleration_histogram(filtered_acceleration, conditions, novice_drivers, expert_drivers):
    # 处理condition_data文件夹数据
    condition_data_folder = "condition_data"

    # 创建14行的子图布局
    fig, axs = plt.subplots(len(conditions), 3, figsize=(18, len(conditions) * 5))
    plt.subplots_adjust(wspace=0.3)

    # 定义加速度范围和直方图的bins
    acceleration_range = (-0.6, 0.6)
    bins = np.linspace(acceleration_range[0], acceleration_range[1], num=50)

    for idx, condition in enumerate(conditions):
        # 加载工况图片
        screenshot_path = os.path.join(screenshot_folder, f"{condition}.png")
        screenshot = plt.imread(screenshot_path)
        
        # 在左侧子图中绘制工况图片
        axs[idx, 0].imshow(screenshot)
        axs[idx, 0].axis('off')  # 关闭坐标轴

        axs[idx, 1].set_title(f"Novice Driver - {condition} Acceleration Distribution")
        axs[idx, 2].set_title(f"Expert Driver - {condition} Acceleration Distribution")
        
        for driver_group in [novice_drivers, expert_drivers]:
            for driver in driver_group:
                file_name = f"{driver}-{condition}-data.csv"
                file_path = os.path.join(condition_data_folder, file_name)
                
                if os.path.exists(file_path):
                    # 获取对应的加速度数据
                    filtered_acc = filtered_acceleration[condition][driver]
                    axsidx = 0
                    if driver_group == novice_drivers:
                        # 绘制加速度频率分布直方图
                        axs[idx, 1].hist(filtered_acc, bins=bins, alpha=0.5, label=driver, density=True)
                        axs[idx, 1].set_xlabel("Acceleration (g)")
                        axs[idx, 1].set_ylabel("Frequency")
                        axs[idx, 1].legend()
                        axsidx = 1
                    else:
                        # 绘制加速度频率分布直方图
                        axs[idx, 2].hist(filtered_acc, bins=bins, alpha=0.5, label=driver, density=True)
                        axs[idx, 2].set_xlabel("Acceleration (g)")
                        axs[idx, 2].set_ylabel("Frequency")
                        axs[idx, 2].legend()
                        axsidx = 2

                    axs[idx, axsidx].axvline(x=0.11, color='gray', linestyle='dashed', linewidth=1)
                    axs[idx, axsidx].axvline(x=-0.11, color='gray', linestyle='dashed', linewidth=1)
                    axs[idx, axsidx].axvline(x=0.52, color='red', linestyle='dashed', linewidth=1)
                    axs[idx, axsidx].axvline(x=0.15, color='gray', linestyle='dashed', linewidth=1)
                    axs[idx, axsidx].axvline(x=-0.15, color='gray', linestyle='dashed', linewidth=1)
                    axs[idx, axsidx].axvline(x=-0.52, color='red', linestyle='dashed', linewidth=1)
                    axs[idx, axsidx].axvline(x=0, color='black', linestyle='dashdot', linewidth=1, alpha=0.3)

    plt.tight_layout()
    # plt.show()
    plt.savefig("acceleration_histogram.png")
    
# plot_speed()
# plot_acceleration()
# plot_jerk()
# plot_streeing()
# plot_streeing_diff()
# plot_steering_percentage()

def plot_trajectory():
    trajectory_data = build_empty_filtered_trajectory(conditions, novice_drivers + expert_drivers)
    for idx, condition in enumerate(conditions):
        ax = axs[idx]
        ax.set_title(f"{condition} Analysis")

        for driver_group in [novice_drivers, expert_drivers]:
            for driver in driver_group:
                file_name = f"{driver}-{condition}-data.csv"
                file_path = os.path.join(condition_data_folder, file_name)
                
                if os.path.exists(file_path):
                    df = pd.read_csv(file_path)
                    
                    if df.shape[0] > 0:
                        df['timestamp'] = [pd.Timestamp(x) for x in df['timestamp']]
                        
                        start_lat = df['gps.latitude'].min()
                        start_log = df['gps.longitude.fix'].min()
                        df['gps.latitude'] = (df['gps.latitude'] - start_lat) #y
                        df['gps.longitude.fix'] = (df['gps.longitude.fix'] - start_log) #y


                        trajectory_data[condition][driver]['y'] = df['gps.latitude']
                        trajectory_data[condition][driver]['x'] = df['gps.longitude.fix']

        ax.set_xlabel("Longitude")
        # ax.set_ylabel("trajectory (mps^2)")
        ax.set_ylabel("Latitude")
        
    plt.tight_layout()
    # plt.show()
    plot_trajectory_with_condition(trajectory_data, conditions, novice_drivers, expert_drivers)

def build_empty_filtered_trajectory(conditions, driver_groups):
    filtered_acceleration = {}
    for condition in conditions:
        condition_data = {}
        for driver in driver_groups:
            condition_data[driver] = {'x': [], 'y':[]}
        filtered_acceleration[condition] = condition_data
    return filtered_acceleration

def plot_trajectory_with_condition(trajectory_data, conditions, novice_drivers, expert_drivers):
    # 处理condition_data文件夹数据
    condition_data_folder = "condition_data"
    novice_label_drawn = False
    expert_label_drawn = False

    # 创建14行的子图布局
    fig, axs = plt.subplots(len(conditions), 2, figsize=(12, len(conditions) * 5))
    plt.subplots_adjust(wspace=0.3)

    for idx, condition in enumerate(conditions):
        # 加载工况图片
        screenshot_path = os.path.join(screenshot_folder, f"{condition}.png")
        screenshot = plt.imread(screenshot_path)
        
        # 在左侧子图中绘制工况图片
        axs[idx, 0].imshow(screenshot)
        axs[idx, 0].axis('off')  # 关闭坐标轴

        axs[idx, 1].set_title(f"Driver - {condition} Trajectory Tracking")
        
        for driver_group in [novice_drivers, expert_drivers]:
            group_color = 'blue' if driver_group == novice_drivers else 'green'
            group_label = "Novice Driver" if driver_group == novice_drivers else "Expert Driver"
            
            combined_trajectory_x = []
            combined_trajectory_y = []

            for driver in driver_group:
                file_name = f"{driver}-{condition}-data.csv"
                file_path = os.path.join(condition_data_folder, file_name)
                
                if os.path.exists(file_path):
                    # 获取对应的加速度数据
                    trajectory_y = trajectory_data[condition][driver]['y']
                    trajectory_x = trajectory_data[condition][driver]['x']

                    if group_label == "Novice Driver" and not novice_label_drawn:
                        axs[idx, 1].plot(trajectory_x, trajectory_y, color=group_color, alpha=0.2, label=group_label)
                    elif group_label == "Expert Driver" and not expert_label_drawn:
                        axs[idx, 1].plot(trajectory_x, trajectory_y, color=group_color, alpha=0.2, label=group_label)
                    else:
                        axs[idx, 1].plot(trajectory_x, trajectory_y, color=group_color, alpha=0.2)

                    if group_label == "Novice Driver":
                        novice_label_drawn = True
                    else:
                        expert_label_drawn = True

                    if(condition is not 'Baseline'):
                        # 累积轨迹数据以便进行拟合，去除重复的 x 值
                        combined_trajectory_x.extend(trajectory_x)
                        combined_trajectory_y.extend(trajectory_y)
                        # axs[idx, axsidx].axvline(x=0.11, color='gray', linestyle='dashed', linewidth=1)

            if(condition is not 'Baseline'):
                # 拟合并绘制贝塞尔曲线
                if len(combined_trajectory_x) > 0:
                    pass
                    # avg_trajectory_x = np.mean(combined_trajectory_x, axis=0)
                    # avg_trajectory_y = np.mean(combined_trajectory_y, axis=0)
                    
                    # # 多项式拟合回归曲线
                    # poly_coeff = np.polyfit(combined_trajectory_x, combined_trajectory_y, deg=conditions_deg[condition])
                    # poly_fit = np.poly1d(poly_coeff)
                    # x_fit = np.linspace(min(combined_trajectory_x), max(combined_trajectory_x), num=100)
                    # y_fit = poly_fit(x_fit)
                    
                    # # 计算拟合曲线的置信区间
                    # std_deviation = np.std(combined_trajectory_y, axis=0)
                    # confidence_interval = 0.2 * std_deviation  # 95% 置信区间
                    
                    # upper_bound = y_fit + confidence_interval
                    # lower_bound = y_fit - confidence_interval
                    
                    # axs[idx, 1].plot(x_fit, y_fit, color=group_color, linestyle='dashed', label=f"{group_label} Regression")
                    # axs[idx, 1].fill_between(x_fit, lower_bound, upper_bound, color=group_color, alpha=0.1)

                    # # ----------进行回归曲线拟合
                    # popt, pcov = curve_fit(trajectory_model, combined_trajectory_x, combined_trajectory_y)
                    # fit_y = trajectory_model(combined_trajectory_x, *popt)

                    # # 绘制回归曲线及置信区间
                    # axs[idx, 1].plot(combined_trajectory_x, fit_y, color=group_color, linestyle='dashed', linewidth=2)
                    # # axs[idx, 1].fill_between(combined_trajectory_x, fit_y - np.sqrt(np.diag(pcov)),
                    # #                          fit_y + np.sqrt(np.diag(pcov)), color=group_color, alpha=0.1)

    plt.tight_layout()
    # plt.show()
    plt.savefig("trajectory_with_condition.png")

# 正弦函数作为拟合模型
def trajectory_model(x, a, b, c):
    fity = []
    for _x in x:
        _y = a * np.sin(b * _x) + c
        fity.append(_y)
    return fity

def plot_acceleration_trajectory():
    acceleration_data = build_empty_filtered_acceleration(conditions, novice_drivers + expert_drivers)
    trajectory_data = build_empty_filtered_trajectory(conditions, novice_drivers + expert_drivers)

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
                        # ------------------处理加速度
                        df['timestamp'] = [pd.Timestamp(x) for x in df['timestamp']]
                        
                        start_time = df['timestamp'].min()
                        df['time_seconds'] = (df['timestamp'] - start_time).dt.total_seconds()

                        # 计算速度的一阶导数，即加速度
                        acceleration = np.gradient(df['speed_mps'], df['time_seconds'])

                        # 过滤掉加速度大于5或小于-5的数据
                        mask = (acceleration > -5) & (acceleration < 5)
                        # filtered_acceleration = np.where(mask, acceleration, np.nan)
                        filtered_acceleration = np.where(mask, acceleration, np.nan) / 9.8
                        filtered_time = np.where(mask, df['time_seconds'], np.nan)

                        acceleration_data[condition][driver] = filtered_acceleration

                        # ----------处理轨迹
                        start_lat = df['gps.latitude'].min()
                        start_log = df['gps.longitude.fix'].min()
                        df['gps.latitude'] = (df['gps.latitude'] - start_lat) #y
                        df['gps.longitude.fix'] = (df['gps.longitude.fix'] - start_log) #y


                        trajectory_data[condition][driver]['y'] = df['gps.latitude']
                        trajectory_data[condition][driver]['x'] = df['gps.longitude.fix']

                        # ----------绘制加速度直方图和轨迹
    plot_acceleration_trajectory_v2(trajectory_data, acceleration_data, conditions, novice_drivers, expert_drivers)
    


def plot_acceleration_trajectory_v2(trajectory_data, filtered_acceleration, conditions, novice_drivers, expert_drivers):
    # 处理condition_data文件夹数据
    condition_data_folder = "condition_data"
    
    # 创建14行的子图布局
    fig, axs = plt.subplots(len(conditions), 4, figsize=(24, len(conditions) * 5))
    plt.subplots_adjust(wspace=0.3)

    # 定义加速度范围和直方图的bins
    acceleration_range = (-0.6, 0.6)
    bins = np.linspace(acceleration_range[0], acceleration_range[1], num=50)

    for idx, condition in enumerate(conditions):
        # 加载工况图片
        screenshot_path = os.path.join(screenshot_folder, f"{condition}.png")
        screenshot = plt.imread(screenshot_path)

        novice_label_drawn = False
        expert_label_drawn = False

        # 在左侧子图中绘制工况图片
        axs[idx, 0].imshow(screenshot)
        axs[idx, 0].axis('off')  # 关闭坐标轴

        axs[idx, 1].set_title(f"Novice Driver - {condition} Acceleration Distribution")
        axs[idx, 2].set_title(f"Expert Driver - {condition} Acceleration Distribution")
        axs[idx, 3].set_title(f"Driver - {condition} Trajectory Tracking")
        
        for driver_group in [novice_drivers, expert_drivers]:
            group_color = 'blue' if driver_group == novice_drivers else 'green'
            group_label = "Novice Driver" if driver_group == novice_drivers else "Expert Driver"

            for driver in driver_group:
                file_name = f"{driver}-{condition}-data.csv"
                file_path = os.path.join(condition_data_folder, file_name)
                
                if os.path.exists(file_path):
                    # 获取对应的加速度数据
                    filtered_acc = filtered_acceleration[condition][driver]
                    # 获取对应的轨迹数据
                    trajectory_y = trajectory_data[condition][driver]['y']
                    trajectory_x = trajectory_data[condition][driver]['x']
                    if(len(trajectory_y) > 0 and len(trajectory_x)>0):
                        trajectory_x, trajectory_y = scatterplot_rotate_flip(trajectory_x, trajectory_y, conditions_rot[condition])

                    axsidx = 0
                    if driver_group == novice_drivers:
                        # 绘制加速度频率分布直方图
                        axs[idx, 1].hist(filtered_acc, bins=bins, alpha=0.5, label=driver, density=True)
                        axs[idx, 1].set_xlabel("Acceleration (g)")
                        axs[idx, 1].set_ylabel("Frequency")
                        axs[idx, 1].legend()
                        axsidx = 1
                        # 绘制轨迹
                        if( not novice_label_drawn):
                            axs[idx, 3].plot(trajectory_x, trajectory_y, color=group_color, alpha=0.2, label=group_label)
                            novice_label_drawn = True
                        else:
                            axs[idx, 3].plot(trajectory_x, trajectory_y, color=group_color, alpha=0.2)
                        axs[idx, 3].set_xlabel("Longitude")
                        axs[idx, 3].set_ylabel("Latitude")
                        axs[idx, 3].legend()
                            
                    else:
                        # 绘制加速度频率分布直方图
                        axs[idx, 2].hist(filtered_acc, bins=bins, alpha=0.5, label=driver, density=True)
                        axs[idx, 2].set_xlabel("Acceleration (g)")
                        axs[idx, 2].set_ylabel("Frequency")
                        axs[idx, 2].legend()
                        axsidx = 2
                        # 绘制轨迹
                        if( not expert_label_drawn):
                            axs[idx, 3].plot(trajectory_x, trajectory_y, color=group_color, alpha=0.2, label=group_label)
                            expert_label_drawn = True
                        else:
                            axs[idx, 3].plot(trajectory_x, trajectory_y, color=group_color, alpha=0.2)
                        axs[idx, 3].set_xlabel("Longitude")
                        axs[idx, 3].set_ylabel("Latitude")
                        axs[idx, 3].legend()
                    

                    axs[idx, axsidx].axvline(x=0.11, color='gray', linestyle='dashed', linewidth=1)
                    axs[idx, axsidx].axvline(x=-0.11, color='gray', linestyle='dashed', linewidth=1)
                    axs[idx, axsidx].axvline(x=0.52, color='red', linestyle='dashed', linewidth=1)
                    axs[idx, axsidx].axvline(x=0.15, color='gray', linestyle='dashed', linewidth=1)
                    axs[idx, axsidx].axvline(x=-0.15, color='gray', linestyle='dashed', linewidth=1)
                    axs[idx, axsidx].axvline(x=-0.52, color='red', linestyle='dashed', linewidth=1)
                    axs[idx, axsidx].axvline(x=0, color='black', linestyle='dashdot', linewidth=1, alpha=0.3)

    plt.tight_layout()
    # plt.show()
    plt.savefig("acceleration_histogram_trajectory.png")
    
def matrix_transpose(x, y, angle):
    matrix = np.array([x, y])  # 将x和y合并成一个矩阵
    if angle == 0:
        transposed_matrix = matrix
    elif angle == 90:
        transposed_matrix = np.transpose(matrix)
    elif angle == 180:
        transposed_matrix = np.rot90(matrix, 2)
    elif angle == 270:
        transposed_matrix = np.rot90(matrix, 3)
    else:
        transposed_matrix = matrix
    
    transposed_x = transposed_matrix[0]
    transposed_y = transposed_matrix[1]
    
    return transposed_x, transposed_y

def scatterplot_rotate(x, y, angle):
    if angle == 0:
        rotated_x = x
        rotated_y = y
    elif angle == 90:
        rotated_x = y
        rotated_y = -x
    elif angle == 180:
        rotated_x = -x
        rotated_y = -y
    elif angle == 270:
        rotated_x = -y
        rotated_y = x
    else:
        rotated_x = x
        rotated_y = y
    
    return rotated_x, rotated_y

def scatterplot_rotate_flip(x, y, angle):
    # 统一镜像翻转
    x = -x
    y = -y
    
    if angle == 0:
        rotated_x = x
        rotated_y = y
    elif angle == 90:
        rotated_x = y
        rotated_y = -x
    elif angle == 180:
        rotated_x = -x
        rotated_y = -y
    elif angle == 270:
        rotated_x = -y
        rotated_y = x
    else:
        rotated_x = x
        rotated_y = y
    
    return rotated_x, rotated_y

plot_acceleration_trajectory()
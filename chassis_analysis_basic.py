import os
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt

# 定义司机分组
novice_drivers = ["P05", "P07", "P08", "P20", "P21", "P22", "P23", "P25", "P26", "P27"]
expert_drivers = ["P06", "P09", "P11", "P13", "P14", "P15", "P16", "P17", "P18", "P19"]

# 定义工况列表
conditions = ['Baseline','A1','A2','D1','S1','A3','A4', 'L1','L2','R1','R2','R3','R4','L3']

# 用于存储分析结果的字典
results = {}

# 用于存储分析结果的字典
results = {"Novice": {cond: {"avg_speed": [], "acceleration": [], "steering_reversals": [], "avg_steering_angle": []} for cond in conditions},
           "Expert": {cond: {"avg_speed": [], "acceleration": [], "steering_reversals": [], "avg_steering_angle": []} for cond in conditions}}

# 存储对比值的数据框
comparison_data = []

def getmindata(data1, data2):
    # 处理长度不一致的数据
    min_length = min(len(data1), len(data2))
    data1 = data1[:min_length]
    data2 = data2[:min_length]
    return data1, data2


# 处理chassis文件夹数据
chassis_folder = "G:/Can-bus Dataset/chassis/"
timedivision_folder = "G:/Can-bus Dataset/timedivisions/"
for file_name in os.listdir(chassis_folder):
    if file_name.endswith("-chassis.csv"):
        experiment_id = file_name.split("-")[0]  # 获取实验试次
        if experiment_id in novice_drivers or experiment_id in expert_drivers:
            df_chassis = pd.read_csv(os.path.join(chassis_folder, file_name))
            
            # 处理timestamp格式
            df_chassis['timestamp'] = [pd.Timestamp(x) for x in df_chassis['timestamp']]
            
            
            # 读取工况时间数据
            timedivision_file_name = f"{experiment_id}-timedivision.csv"
            df_timedivision = pd.read_csv(os.path.join(timedivision_folder, timedivision_file_name))
            
            for condition in conditions:
                condition_start_time = [pd.Timestamp(x) for x in df_timedivision[df_timedivision['state'] == condition]['start_timestamp']]
                condition_finish_time = [pd.Timestamp(x) for x in df_timedivision[df_timedivision['state'] == condition]['finish_timestamp']]
                
                # 根据时间截取数据
                condition_data = df_chassis[(df_chassis['timestamp'] >= condition_start_time[0]) &
                                           (df_chassis['timestamp'] <= condition_finish_time[0])]
                condition_data.to_csv(f"./condition_data/{experiment_id}-{condition}-data.csv", index=False)
                if condition_data.shape[0] > 0:
                    avg_speed = condition_data["speed_mps"].mean()
                    acceleration = np.diff(condition_data["speed_mps"]).mean()
                    steering_reversals = condition_data[condition_data["steering_percentage"] < 0]["steering_percentage"].count()
                    avg_steering_angle = condition_data["steering_percentage"].mean()
                    
                    group = "Novice" if experiment_id in novice_drivers else "Expert"
                    
                    if group not in results:
                        results[group] = {cond: {"avg_speed": [], "acceleration": [], "steering_reversals": [], "avg_steering_angle": []} for cond in conditions}
                    
                    results[group][condition]["avg_speed"].append(avg_speed)
                    results[group][condition]["acceleration"].append(acceleration)
                    results[group][condition]["steering_reversals"].append(steering_reversals)
                    results[group][condition]["avg_steering_angle"].append(avg_steering_angle)

results.to_csv("./data/canbus_summary.csv", index=False)

# # 可视化结果
# for condition in conditions:
#     plt.figure(figsize=(12, 6))
#     plt.title(f"{condition} Analysis")
    
#     # 可视化平均主车速度
#     plt.subplot(2, 2, 1)
#     novice_speeds = results["Novice"][condition]["avg_speed"]
#     expert_speeds = results["Expert"][condition]["avg_speed"]
#     plt.boxplot([novice_speeds, expert_speeds], labels=["Novice", "Expert"])
#     plt.xticks([1], ["Average Speed"])
#     plt.ylabel("mps")
    
#     # 可视化加速度变量情况
#     plt.subplot(2, 2, 2)
#     novice_accelerations = results["Novice"][condition]["acceleration"]
#     expert_accelerations = results["Expert"][condition]["acceleration"]
#     plt.boxplot([novice_accelerations, expert_accelerations], labels=["Novice", "Expert"])
#     plt.xticks([1], ["Acceleration"])
#     plt.ylabel("mps^2")
    
#     # 可视化Steering reversal rate
#     plt.subplot(2, 2, 3)
#     novice_reversals = results["Novice"][condition]["steering_reversals"]
#     expert_reversals = results["Expert"][condition]["steering_reversals"]
#     plt.boxplot([novice_reversals, expert_reversals], labels=["Novice", "Expert"])
#     plt.xticks([1], ["Steering Reversals"])
#     plt.ylabel("Count")
    
#     # 可视化平均Steering angle
#     plt.subplot(2, 2, 4)
#     novice_angles = results["Novice"][condition]["avg_steering_angle"]
#     expert_angles = results["Expert"][condition]["avg_steering_angle"]
#     plt.boxplot([novice_angles, expert_angles], labels=["Novice", "Expert"])
#     plt.xticks([1], ["Average Steering Angle"])
#     plt.ylabel("Percentage")
    
#     plt.tight_layout()
#     # plt.show()
    
#     plt.title(f"{condition} Analysis")
#     plt.savefig(f"{condition} Analysis.png")
    
def export_mean_csv():
    # 计算每个条件的平均值并存储
    for condition in conditions:
        expert_speeds = results["Expert"][condition]["avg_speed"]
        novice_speeds = results["Novice"][condition]["avg_speed"]
        expert_accelerations = np.concatenate([data[condition]["acceleration"] for driver, data in results.items() if driver == "Expert"])
        novice_accelerations = np.concatenate([data[condition]["acceleration"] for driver, data in results.items() if driver == "Novice"])
        expert_reversals = np.concatenate([data[condition]["steering_reversals"] for driver, data in results.items() if driver == "Expert"])
        novice_reversals = np.concatenate([data[condition]["steering_reversals"] for driver, data in results.items() if driver == "Novice"])
        expert_angles = results["Expert"][condition]["avg_steering_angle"]
        novice_angles = results["Novice"][condition]["avg_steering_angle"]

        # 计算平均值
        avg_speed_mean_expert = np.mean(expert_speeds)
        avg_speed_mean_novice = np.mean(novice_speeds)
        acceleration_mean_expert = np.mean(expert_accelerations)
        acceleration_mean_novice = np.mean(novice_accelerations)
        steering_reversals_mean_expert = np.mean(expert_reversals)
        steering_reversals_mean_novice = np.mean(novice_reversals)
        avg_steering_angle_mean_expert = np.mean(expert_angles)
        avg_steering_angle_mean_novice = np.mean(novice_angles)
        
        # 存储对比值
        comparison_data.append([condition, avg_speed_mean_expert, avg_speed_mean_novice,
                                acceleration_mean_expert, acceleration_mean_novice,
                                steering_reversals_mean_expert, steering_reversals_mean_novice,
                                avg_steering_angle_mean_expert, avg_steering_angle_mean_novice])

    # 创建DataFrame
    comparison_df = pd.DataFrame(comparison_data, columns=["condition", "Avg_Speed_mean(expert_group)", "Avg_Speed_mean(novice_group)",
                                                        "Acceleration_mean(expert_group)", "Acceleration_mean(novice_group)",
                                                        "Steering_Reversals_mean(expert_group)", "Steering_Reversals_mean(novice_group)",
                                                        "Avg_Steering_Angle_mean(expert_group)", "Avg_Steering_Angle_mean(novice_group)"])

    # 保存为CSV文件
    comparison_df.to_csv("./analysis_results/combined_analysis_results.csv", index=False)






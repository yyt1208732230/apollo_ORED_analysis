import os
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import csv

# 定义司机分组
novice_drivers = ["P05", "P07", "P08", "P20", "P21", "P22", "P23", "P25", "P26", "P27"]
expert_drivers = ["P06", "P09", "P11", "P13", "P14", "P15", "P16", "P17", "P18", "P19"]

# 定义工况列表
conditions = ['Baseline','A1','A2','D1','S1','A3','A4', 'L1','L2','R1','R2','R3','R4','L3']

# 用于存储分析结果的字典
results = {}

# 用于存储分析结果的字典
results = {"Novice": {cond: {"avg_speed": {}, "acceleration": {}, "steering_reversals": {}, "avg_steering_angle": {}} for cond in conditions},
           "Expert": {cond: {"avg_speed": {}, "acceleration": {}, "steering_reversals": {}, "avg_steering_angle": {}} for cond in conditions}}

field_names = ['trial', 'driver group', 'driving condition', 'avg speed (m/s)', 'acceleration (g)', 'jerk', 'steering reversals', 'avg steering angle']

# 存储对比值的数据框
comparison_data = []

sterring_angle_unit = 0.0139 #5 deg out of 360 into percentage

def write_csv_header(filename):
    # with open('chassis_analysis_export.csv', 'w') as csvfile:
    with open(filename, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames = field_names)
        writer.writeheader()

def getmindata(data1, data2):
    # 处理长度不一致的数据
    min_length = min(len(data1), len(data2))
    data1 = data1[:min_length]
    data2 = data2[:min_length]
    return data1, data2

def process_chassis():
    # 处理chassis文件夹数据
    chassis_folder = "G:/Can-bus Dataset/chassis/"
    timedivision_folder = "G:/Can-bus Dataset/timedivisions/"
    csv_filename = 'chassis_analysis_export.csv'
    # writer = write_csv_header(csv_filename)
    each_result = []
    results = {"Novice": {cond: {"avg_speed": {}, "acceleration": {}, "steering_reversals_rate": {}, "avg_steering_angle": {}, "jerk": {}} for cond in conditions},
           "Expert": {cond: {"avg_speed": {}, "acceleration": {}, "steering_reversals_rate": {}, "avg_steering_angle": {}, "jerk": {}} for cond in conditions}}
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
                    
                    start_time = condition_data['timestamp'].min()
                    condition_data['time_seconds'] = (condition_data['timestamp'] - start_time).dt.total_seconds()

                    condition_data.to_csv(f"./condition_data/{experiment_id}-{condition}-data.csv", index=False)
                    if condition_data.shape[0] > 0:
                        avg_speed = condition_data["speed_mps"].mean()
                        acceleration = np.gradient(condition_data['speed_mps']/9.8, condition_data['time_seconds'])
                        acceleration_filtered = list(filter(lambda x: x < 0.51 and x > -0.51, acceleration))
                        acceleration_mean = sum(acceleration_filtered) / len(acceleration_filtered)
                        jerk = np.gradient(acceleration, condition_data['time_seconds'])
                        jerk_mean = jerk.mean()
                        avg_steering_angle = condition_data["steering_percentage"].mean()
                        event_count, interval_count, turn_events = calculate_turn_events(condition_data["steering_percentage"], condition_data['time_seconds'])
                        steering_reversals_rate = 0 if interval_count == 0 else event_count / interval_count
                        group = "Novice" if experiment_id in novice_drivers else "Expert"
                        
                        if group not in results:
                            results[group] = {cond: {"avg_speed": {}, "acceleration": {}, "steering_reversals": {}, "avg_steering_angle": {}} for cond in conditions}
                        
                        results[group][condition]["avg_speed"][experiment_id] = avg_speed
                        results[group][condition]["acceleration"][experiment_id] = acceleration_mean
                        results[group][condition]["steering_reversals_rate"][experiment_id] = steering_reversals_rate
                        results[group][condition]["avg_steering_angle"][experiment_id] = avg_steering_angle
                        results[group][condition]["jerk"][experiment_id] = jerk_mean
                        each_result.append([experiment_id, 
                                            group, 
                                            condition,
                                            avg_speed, 
                                            acceleration_mean, 
                                            jerk_mean, steering_reversals_rate, avg_steering_angle])
    with open(csv_filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerow(field_names)
            for row in each_result:
                writer.writerow(row)                   

    # results.to_csv("./data/canbus_summary.csv", index=False)

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


def process_chassis_value_4spss(savefolder, valueName):
    # 处理chassis文件夹数据
    chassis_folder = "G:/Can-bus Dataset/chassis/"
    timedivision_folder = "G:/Can-bus Dataset/timedivisions/"
    csv_filename = savefolder + '/chassis_' + valueName + '.csv'
    # writer = write_csv_header(csv_filename)
    results = {"Novice": {cond: {"avg_speed": {}, "acceleration": {}, "steering_reversals_rate": {}, "avg_steering_angle": {}, "jerk": {}} for cond in conditions},
           "Expert": {cond: {"avg_speed": {}, "acceleration": {}, "steering_reversals_rate": {}, "avg_steering_angle": {}, "jerk": {}} for cond in conditions}}
    each_result = []
    field_name_steering_reversals = ['trial', 'driver']
    for cond in conditions:
        field_name_steering_reversals.append(cond + '_sr')

    with open(csv_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(field_name_steering_reversals)

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
                        
                        start_time = condition_data['timestamp'].min()
                        condition_data['time_seconds'] = (condition_data['timestamp'] - start_time).dt.total_seconds()

                        # condition_data.to_csv(f"./condition_data/{experiment_id}-{condition}-data.csv", index=False)

                        if condition_data.shape[0] > 0:
                            avg_speed = condition_data["speed_mps"].mean()
                            acceleration = np.gradient(condition_data['speed_mps']/9.8, condition_data['time_seconds'])
                            acceleration_filtered = list(filter(lambda x: x < 0.51 and x > -0.51, acceleration))
                            acceleration_mean = sum(acceleration_filtered) / len(acceleration_filtered)
                            jerk = np.gradient(acceleration, condition_data['time_seconds'])
                            jerk_mean = jerk.mean()
                            avg_steering_angle = condition_data["steering_percentage"].mean()
                            event_count, interval_count, turn_events = calculate_turn_events(condition_data["steering_percentage"], condition_data['time_seconds'])
                            steering_reversals_rate = 0 if interval_count == 0 else event_count / interval_count
                            group = "Novice" if experiment_id in novice_drivers else "Expert"
                            
                            if group not in results:
                                results[group] = {cond: {"avg_speed": {}, "acceleration": {}, "steering_reversals": {}, "avg_steering_angle": {}} for cond in conditions}
                            
                            results[group][condition]["avg_speed"][experiment_id] = avg_speed
                            results[group][condition]["acceleration"][experiment_id] = acceleration_mean
                            results[group][condition]["steering_reversals_rate"][experiment_id] = steering_reversals_rate
                            results[group][condition]["avg_steering_angle"][experiment_id] = avg_steering_angle
                            results[group][condition]["jerk"][experiment_id] = jerk_mean
                            each_result.append([experiment_id, 
                                                group, 
                                                condition,
                                                avg_speed, 
                                                acceleration_mean, 
                                                jerk_mean, steering_reversals_rate, avg_steering_angle])
                        else:
                            print(experiment_id + condition + 'not found')

                wrote_id = []
                for row in each_result:
                    if(row[0] == experiment_id and experiment_id not in wrote_id):
                        _group = "Novice" if experiment_id in novice_drivers else "Expert"
                        _row = [experiment_id, _group]
                        for cond in conditions:
                            # _row.append(results[_group][cond]["steering_reversals_rate"][experiment_id])
                            _row.append(results[_group][cond][valueName][experiment_id])
                        writer.writerow(_row)
                        wrote_id.append(experiment_id)           

        # results.to_csv("./data/canbus_summary.csv", index=False)



def calculate_turn_events(steering_data_raw, speed_data_raw):
    steering_data = steering_data_raw.tolist()
    speed_data = speed_data_raw.tolist()
    threshold_angle = 5 / 360  # Convert 5° to steering percentage
    time_interval = 0.1  # seconds

    turn_events = []
    event_count = 0
    interval_count = 0
    prev_angle = steering_data[0]
    prev_time = 0

    for angle, time in zip(steering_data[1:], speed_data[1:]):
        time_difference = time - prev_time
        angle_difference = abs(angle - prev_angle)

        if time_difference >= time_interval:
            if angle_difference >= threshold_angle:
                event_count += 1
                turn_events.append(prev_time)
            prev_time = time
            prev_angle = angle
            interval_count += 1

    return event_count, interval_count, turn_events

if __name__ == "__main__":
    # process_chassis_value_4spss('./analysis_results/for each/', 'steering_reversals_rate')
    # process_chassis_value_4spss('./analysis_results/for each/', 'avg_speed')
    # process_chassis_value_4spss('./analysis_results/for each/', 'acceleration')
    # process_chassis_value_4spss('./analysis_results/for each/', 'avg_steering_angle')
    process_chassis()
    pass



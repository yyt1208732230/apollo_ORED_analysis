import os
import shutil

def copyfile(source_path, target_path):
    try:
        shutil.copy(source_path, target_path)
        print(f"{source_path} 拷贝成功")
    except Exception as e:
        print(f"{source_path} 拷贝失败: {str(e)}")

def copy_files(source_folder, target_folder):
    try:
        # 获取Real Road AIR Dataset文件夹下所有的被试文件夹名
        subject_folders = [folder for folder in os.listdir(source_folder) if folder.startswith("P")]

        # 遍历每个被试文件夹
        for subject_folder in subject_folders:
            # 构建被试文件夹路径
            subject_path = os.path.join(source_folder, subject_folder)
            
            # 构建“7.CANBus”文件夹路径
            canbus_folder_path = os.path.join(subject_path, "7.CANBus")
            
            # 检查路径是否存在
            if os.path.exists(canbus_folder_path):
                # 获取符合条件的csv和png文件列表
                file_list = os.listdir(canbus_folder_path)
                # matching_files = [file for file in file_list if file.startswith("timedivision--") or file.endswith("Tracking Route.png")]
                # matching_files = [file for file in file_list if file.startswith("fixgpschassis--")]
                matching_files = [file for file in file_list if file.startswith("timedivision--")]
                
                # 创建被试文件夹在目标文件夹中的路径
                # target_subject_folder = os.path.join(target_folder, subject_folder)
                target_subject_folder = os.path.join(target_folder)
                os.makedirs(target_subject_folder, exist_ok=True)
                
                # 拷贝符合条件的文件到目标文件夹
                for file_name in matching_files:
                    source_file_path = os.path.join(canbus_folder_path, file_name)
                    # target_file_path = os.path.join(target_subject_folder, subject_folder + '-chassis.csv')
                    target_file_path = os.path.join(target_subject_folder, subject_folder + '-timedivision.csv')
                    shutil.copy2(source_file_path, target_file_path)
                
    except Exception as e:
        print("Error:", e)

# 调用函数进行拷贝
source_folder = "G:/Real Road AIR Dataset/"
target_folder = "G:/Can-bus Dataset/timedivisions/"
copy_files(source_folder, target_folder)

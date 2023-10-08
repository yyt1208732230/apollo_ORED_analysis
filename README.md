# apollo_ORED_analysis
The scripts for analysis expert driver dataset.

1. 数据清洗：我们筛选了20个司机后，将所有解析后的gps和chassis数据，以及工况时间分割数据进行整理。这些处理后的数据放在位置：Can-bus Dataset 的文件夹/压缩包内。（批量拷贝代码：move_canfiles.py）
[Image]
2. 数据分类：我们按照SPSS分析时所需的参数，通过输入Can-bus Dataset的chassis&GPS解析数据以及工况时间戳数据，分别输出为平均速度、加速度、平均方向盘转向角、方向盘逆转率的数值汇总CSV文件，名称分别为
  - chassis_avg_speed.csv，
  - chassis_acceleration.csv，
  - chassis_avg_steering_angle.csv，
  - chassis_steering_reversals_rate.csv和
  - chassis_analysis_result.csv。并使用这些数据进行初步的SPSS分析。
（SPSS数据分类代码可参考：以每个工况分类的 chassis_analysis_basic.py 或 以每个chassis变量分类的chassis_analysis_export.py）数据放在位置：Can-bus Dataset\1_Can-bus SPSS analysis\ 的文件夹内
[Image]
3. 数据可视化：我们使用上述分类后的的数据分别对速度、加速度、jerk、方向转动比率和方向盘转动距离进行了一系列的可视化尝试。（代码可参考 普通可视化 speed_visulization.py，带截图的可视化 speed_visulization_withfig.py， 带插值的可视化 speed_visulization_interpolation.py 或 重新编写）数据放在位置：Can-bus Dataset\2_datapreprocess_visulization\ 的文件夹内
[Image]
  - 效果如下：
  速度
[Image]
  加速度
[Image]
  jerk
[Image]
  方向盘比率
[Image]
  方向盘转动距离
[Image]
  不舒适分析：加速度频率分布图+工况轨迹
[Image]
4. 显著性分析（工况）：我们结合SPSS数据和可视化结果，对部分常用的舒适度评估参数进行了纵向以及横向数据的显著性可视化。
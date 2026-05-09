# 关键文件
实验日志：/home/fudan222/ct/openpi_agibot/agi_bot/experiment_log.md
openpi官方README：/home/fudan222/ct/openpi_agibot/README.md

# 数据处理
原始数据：agi_bot/data/cartesian_grasp_routeB_5pt_clean_v2
数据处理：/home/fudan222/ct/openpi_agibot/agi_bot/convert_agibot_data_to_lerobot.py
处理后数据参考：agi_bot/test_converted_data

# 微调规则
智元官方数据格式：（x,y,z,四元数，gripper）
action_horizon ：预测动作块的长度。通常设置为 10 到 50 。
冻结主干网络（PaliGemma），仅微调 Action Expert。
夹爪数据只有0和1，分别表示不夹爪和夹爪，需要处理。
末端位姿使用6d旋转矩阵。6d旋转矩阵效果最好且不用归一化。

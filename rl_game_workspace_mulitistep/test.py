import gymnasium as gym
import ale_py

# 注册 ALE 环境
gym.register_envs(ale_py)

# 创建 AirRaid 环境
env = gym.make('ALE/AirRaid-v5', render_mode='human')  # 设置 render_mode 为 'human' 以显示图形界面

# 重置环境，获取初始观测信息
obs, info = env.reset()

# 执行随机动作
for _ in range(1000):  # 可以根据需要更改循环次数
    action = env.action_space.sample()  # 随机选择一个动作
    obs, reward, terminated, truncated, info = env.step(action)  # 执行动作并获取反馈
    env.render()  # 渲染图形界面

    if terminated or truncated:
        break  # 如果游戏结束或被截断，退出循环

# 关闭环境
env.close()

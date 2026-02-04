import gymnasium as gym
import ale_py
import numpy as np
from collections import deque

# 注册 ALE 环境
gym.register_envs(ale_py)

class AtariEnvVector(gym.Env):
    def __init__(self, env_name='ALE/AirRaid-v5', frame_skip=4, frame_stack=4, screen_size=84):
        self.env = gym.wrappers.TimeLimit(gym.make(env_name), max_episode_steps=1000)  # Removed render_mode for vector compatibility
        self.frame_skip = frame_skip
        self.frame_stack = frame_stack
        self.screen_size = screen_size
        self.frames = deque(maxlen=frame_stack)
        self.previous_lives = None  # 跟踪上一帧的生命数

        # 获取动作空间大小
        self.action_space = self.env.action_space
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(frame_stack, screen_size, screen_size), dtype=np.uint8)

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed)
        obs = self._process_frame(obs)
        self.frames.clear()
        for _ in range(self.frame_stack):
            self.frames.append(obs)
        self.previous_lives = self.env.unwrapped.ale.lives()
        return np.stack(self.frames, axis=0), info

    def step(self, action):
        total_reward = 0
        for _ in range(self.frame_skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        obs = self._process_frame(obs)
        self.frames.append(obs)
        state = np.stack(self.frames, axis=0)

        # Bullet detection and reward shaping
        current_lives = self.env.unwrapped.ale.lives()
        if self.previous_lives is not None and current_lives < self.previous_lives:
            total_reward -= 500  # Penalty for losing a life
        self.previous_lives = current_lives

        # Bullet detection removed for simplicity
        # frame = obs
        # hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
        # lower_white = np.array([0, 0, 200])
        # upper_white = np.array([180, 30, 255])
        # mask = cv2.inRange(hsv, lower_white, upper_white)
        # contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # bullet_count = 0
        # for contour in contours:
        #     area = cv2.contourArea(contour)
        #     if 5 < area < 50:  # Bullet size
        #         bullet_count += 1
        # if bullet_count > 0:
        #     total_reward += bullet_count * 10  # Reward for shooting bullets

        return state, total_reward, terminated, truncated, info

    def _process_frame(self, frame):
        frame = np.mean(frame, axis=2).astype(np.uint8)  # RGB to grayscale (210,160)
        # Simple nearest neighbor resize to (84,84)
        h, w = frame.shape
        new_h, new_w = self.screen_size, self.screen_size
        resized = np.zeros((new_h, new_w), dtype=frame.dtype)
        for i in range(new_h):
            for j in range(new_w):
                orig_i = int(i * h / new_h)
                orig_j = int(j * w / new_w)
                resized[i, j] = frame[orig_i, orig_j]
        return resized

    def close(self):
        self.env.close()
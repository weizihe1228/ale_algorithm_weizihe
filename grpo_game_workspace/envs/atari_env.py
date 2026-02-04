import gymnasium as gym
import ale_py
import numpy as np
import cv2
from collections import deque

# 注册 ALE 环境
gym.register_envs(ale_py)

class AtariEnv:
    def __init__(self, env_name='ALE/AirRaid-v5', frame_skip=4, frame_stack=4, screen_size=84, render_mode='rgb_array', death_penalty=0):
        # 将 render_mode 传递给 gym.make，这样 env.render() 会使用正确的模式
        self.env = gym.make(env_name, render_mode=render_mode)
        self.frame_skip = frame_skip
        self.frame_stack = frame_stack
        self.screen_size = screen_size
        self.frames = deque(maxlen=frame_stack)
        self.previous_lives = None  # 跟踪上一帧的生命数
        self.death_penalty = death_penalty

        # 获取动作空间大小
        self.action_space = self.env.action_space
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(screen_size, screen_size, frame_stack), dtype=np.uint8
        )

    def _preprocess_frame(self, frame):
        # 转换为灰度
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        # 调整大小
        resized = cv2.resize(gray, (self.screen_size, self.screen_size), interpolation=cv2.INTER_AREA)
        return resized.astype(np.uint8)

    def _get_plane_x(self, obs):
        # obs is (84, 84, 4), take first channel
        gray = obs[:, :, 0]
        # Assume plane is in bottom rows, find brightest x
        bottom = gray[-10:, :]  # bottom 10 rows
        _, thresh = cv2.threshold(bottom, 200, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            # Find largest contour
            largest = max(contours, key=cv2.contourArea)
            M = cv2.moments(largest)
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                return cx
        return 42  # center if not found

    def _get_bullets(self, obs):
        # Detect enemy bullets using motion: bullets moving downward in upper area
        gray_current = obs[:, :, 0]  # Current frame
        gray_prev = obs[:, :, 3]  # Previous frame (last in stack)
        diff = cv2.absdiff(gray_current, gray_prev)
        _, thresh = cv2.threshold(diff, 10, 255, cv2.THRESH_BINARY)  # Higher threshold to reduce noise
        # Clean up noise with morphology
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)  # Remove small noise
        enemy_area = thresh  # Use full image for detection
        contours, _ = cv2.findContours(enemy_area, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        bullets = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 1 < area < 30 and len(cnt) > 4:  # Stricter size and shape for bullet-like objects
                M = cv2.moments(cnt)
                if M['m00'] != 0:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])  # Full image, no offset
                    # Check motion direction and brightness: enemy bullet if moving down and bright white dot
                    if gray_current[cy, cx] > gray_prev[cy, cx] and gray_current[cy, cx] > 150:
                        bullets.append((cx, cy))
        return bullets
    def _get_enemy_planes(self, obs):
        # Detect enemy planes: assume medium bright areas in middle-upper part
        gray = obs[:, :, 0]
        middle = gray[20:70, :]  # middle to upper rows
        _, thresh = cv2.threshold(middle, 180, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        enemies = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 50 < area < 300:  # Enemy plane size range (larger than bullets)
                M = cv2.moments(cnt)
                if M['m00'] != 0:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00']) + 20  # Adjust for middle region
                    enemies.append((cx, cy))
        return enemies

    def reset(self):
        obs, info = self.env.reset()
        frame = self._preprocess_frame(obs)
        # 初始化帧堆叠
        for _ in range(self.frame_stack):
            self.frames.append(frame)
        # 使用 ALE 直接获取生命数，避免 info 中缺失
        try:
            self.previous_lives = self.env.unwrapped.ale.lives()
        except Exception:
            self.previous_lives = info.get('lives', 3)
        return self._get_obs(), info

    def step(self, action):
        total_reward = 0
        for _ in range(self.frame_skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            if terminated or truncated:
                break

        # 检查生命数减少（玩家死亡）
        try:
            current_lives = self.env.unwrapped.ale.lives()
        except Exception:
            current_lives = info.get('lives', 3)
        if self.previous_lives is not None and current_lives < self.previous_lives:
            total_reward += float(self.death_penalty)  # 死亡惩罚（可配置，默认0表示不惩罚）
        self.previous_lives = current_lives

        frame = self._preprocess_frame(obs)
        self.frames.append(frame)
        obs_processed = self._get_obs()

        # Bullet avoidance penalty
        # plane_x = self._get_plane_x(obs_processed)
        # bullets = self._get_bullets(obs_processed)
        # for bx, by in bullets:
        #     distance = abs(bx - plane_x)
        #     if distance < 5:  # Close horizontal distance
        #         total_reward -= 5  # Penalty for being close to bullet

        # Enemy plane alignment penalty
        # enemies = self._get_enemy_planes(obs_processed)
        # for ex, ey in enemies:
        #     if ey > 60:  # Enemy close vertically (near bottom)
        #         distance = abs(ex - plane_x)
        #         if distance < 20:  # Horizontal distance less than plane width
        #             total_reward -= 10  # Penalty for being too close

        return obs_processed, total_reward, terminated, truncated, info

    def _get_obs(self):
        return np.stack(self.frames, axis=-1)

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()
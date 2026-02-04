import torch
import numpy as np
from envs.atari_env import AtariEnv
import pickle
import os
import keyboard
import time
import pyautogui
import cv2
import threading

class PIDController:
    def __init__(self, Kp=0.5, Ki=0.1, Kd=0.05):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.prev_error = 0
        self.integral = 0

    def compute(self, error):
        self.integral += error
        derivative = error - self.prev_error
        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        self.prev_error = error
        return output

def get_plane_x(obs):
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

def collect_human_demos(env_name='ALE/AirRaid-v5', num_demos=5, save_path='human_demos.pkl', use_mouse=False):
    env = AtariEnv(env_name, render_mode='human')
    num_actions = env.action_space.n

    demos = []

    prev_plane_x = 42  # Initial center

    for demo in range(num_demos):
        print(f"Demo {demo + 1}/{num_demos}: Play the game. Controls: A/D for manual, or mouse for tracking. Space to fire, Q to quit")
        state, _ = env.reset()
        done = False
        episode_data = []

        while not done:
            env.render()
            time.sleep(0.01)  # Very fast loop for responsiveness

            current_action = 1  # Default FIRE

            if keyboard.is_pressed('q'):
                done = True
                break

            if use_mouse:
                # Mouse tracking
                mouse_x, _ = pyautogui.position()
                screen_width = pyautogui.size()[0]
                target_x = int(mouse_x / screen_width * 84)
                plane_x = get_plane_x(state)

                # Simple proportional control
                if plane_x < target_x:
                    current_action = 2  # RIGHT
                elif plane_x > target_x:
                    current_action = 3  # LEFT
                # Else stay FIRE

                # Visualization
                vis_img = state[:, :, 0].copy()
                cv2.line(vis_img, (target_x, 0), (target_x, 83), 255, 1)
                cv2.imshow('Mouse Target Visualization', vis_img)
                cv2.waitKey(1)
            else:
                # Manual control
                if keyboard.is_pressed('a') or keyboard.is_pressed('A'):
                    current_action = 3  # LEFT
                elif keyboard.is_pressed('d') or keyboard.is_pressed('D'):
                    current_action = 2  # RIGHT
                elif keyboard.is_pressed('s') or keyboard.is_pressed('S'):
                    current_action = 0  # NOOP

            next_state, reward, terminated, truncated, _ = env.step(current_action)
            done = terminated or truncated

            episode_data.append({
                'state': state,
                'action': current_action,
                'reward': reward,
                'next_state': next_state,
                'done': done
            })

            state = next_state

        demos.append(episode_data)
        print(f"Demo {demo + 1} completed, length: {len(episode_data)}")

    env.close()
    cv2.destroyAllWindows()

    with open(save_path, 'wb') as f:
        pickle.dump(demos, f)
    print(f"Saved {len(demos)} human demos to {save_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_mouse', action='store_true', help='Use mouse PID tracking')
    args = parser.parse_args()
    collect_human_demos(use_mouse=args.use_mouse)
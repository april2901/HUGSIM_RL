import gymnasium as gym
import numpy as np
from gymnasium import spaces

class SimpleAttackEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.dt = 0.1
        self.max_steps = 200

        self.action_space = spaces.Box(
            low=np.array([-5.0, -0.5], dtype=np.float32),
            high=np.array([5.0, 0.5], dtype=np.float32),
            dtype=np.float32,
        )

        self.observation_space = spaces.Box(
            low=np.array([-1000., -1000., -np.pi, 0., -1000., -1000., -1000., -1000.], dtype=np.float32),
            high=np.array([1000., 1000., np.pi, 30., 1000., 1000., 1000., 1000.], dtype=np.float32),
            dtype=np.float32,
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.attacker_state = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self.ego_state = np.array([
            np.random.uniform(10, 30),
            np.random.uniform(-10, 10),
        ], dtype=np.float32)
        # ego의 속도와 방향 (학습 환경에서는 constant velocity로 가정)
        self.ego_velocity = np.random.uniform(-5, 5)
        self.ego_heading = np.random.uniform(-np.pi, np.pi)
        self.prev_dist = np.linalg.norm(self.attacker_state[:2] - self.ego_state)
        return self._get_obs(), {}

    def step(self, action):
        self.current_step += 1
        a, steer = action
        x, y, yaw, v = self.attacker_state
        L = 2.8

        next_x = x + v * np.cos(yaw) * self.dt
        next_y = y + v * np.sin(yaw) * self.dt
        next_yaw = yaw + (v / L) * np.tan(steer) * self.dt
        next_v = np.clip(v + a * self.dt, 0.0, 30.0)

        self.attacker_state = np.array([next_x, next_y, next_yaw, next_v], dtype=np.float32)

        # ego도 constant velocity로 업데이트
        ego_next_x = self.ego_state[0] + self.ego_velocity * np.cos(self.ego_heading) * self.dt
        ego_next_y = self.ego_state[1] + self.ego_velocity * np.sin(self.ego_heading) * self.dt
        self.ego_state = np.array([ego_next_x, ego_next_y], dtype=np.float32)

        current_dist = np.linalg.norm(self.attacker_state[:2] - self.ego_state)
        reward = self.prev_dist - current_dist
        self.prev_dist = current_dist

        terminated = False
        truncated = False
        if current_dist < 2.0:
            reward += 100.0
            terminated = True
        elif self.current_step >= self.max_steps:
            truncated = True

        return self._get_obs(), reward, terminated, truncated, {}

    def _get_obs(self):
        # ego의 현재 위치와 미래 위치(T=max_steps 후)를 포함
        ego_future_x = self.ego_state[0] + self.ego_velocity * np.cos(self.ego_heading) * self.max_steps * self.dt
        ego_future_y = self.ego_state[1] + self.ego_velocity * np.sin(self.ego_heading) * self.max_steps * self.dt
        return np.concatenate([
            self.attacker_state,
            self.ego_state,
            np.array([ego_future_x, ego_future_y])
        ]).astype(np.float32)
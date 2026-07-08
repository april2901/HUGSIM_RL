import gymnasium as gym
import numpy as np
from gymnasium import spaces


DEFAULT_REWARD_WEIGHTS = {
    "approach": 1.0,
    "collision": 1.0,
}

DEFAULT_REWARD_PARAMS = {
    "approach_scale": 10.0,
    "collision_bonus": 10.0,
    "collision_dist": 2.0,
    "vmin": 0.0,
    "vmax": 30.0,
}


class SimpleAttackEnv(gym.Env):
    def __init__(self, reward_weights=None, reward_params=None):
        super().__init__()

        self.dt = 0.1
        self.max_steps = 200
        self.L = 2.8

        self.acc_limit = 5.0
        self.steer_limit = 0.5

        self.reward_weights = {**DEFAULT_REWARD_WEIGHTS}
        if reward_weights:
            self.reward_weights.update(reward_weights)

        self.cfg = {**DEFAULT_REWARD_PARAMS}
        if reward_params:
            self.cfg.update(reward_params)

        self.action_space = spaces.Box(
            low=np.array([-self.acc_limit, -self.steer_limit], dtype=np.float32),
            high=np.array([self.acc_limit, self.steer_limit], dtype=np.float32),
            dtype=np.float32,
        )

        self.observation_space = spaces.Box(
            low=np.array(
                [-1000., -1000., -np.pi, 0., -1000., -1000., -1000., -1000.],
                dtype=np.float32,
            ),
            high=np.array(
                [1000., 1000., np.pi, 30., 1000., 1000., 1000., 1000.],
                dtype=np.float32,
            ),
            dtype=np.float32,
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.current_step = 0

        self.ego_state = np.array(
            [
                np.random.uniform(10.0, 30.0),
                np.random.uniform(-2.0, 2.0),
            ],
            dtype=np.float32,
        )

        self.ego_velocity = np.random.uniform(5.0, 12.0)
        self.ego_heading = 0.0

        lane_width = 3.5
        lane_offset = np.random.choice([-lane_width, 0.0, lane_width])

        self.attacker_state = np.array(
            [
                self.ego_state[0] + np.random.uniform(-25.0, 15.0),
                self.ego_state[1] + lane_offset,
                self.ego_heading,
                np.random.uniform(5.0, 12.0),
            ],
            dtype=np.float32,
        )

        self.prev_dist = float(
            np.linalg.norm(self.attacker_state[:2] - self.ego_state)
        )

        return self._get_obs(), {}

    def step(self, action):
        self.current_step += 1

        a = float(np.clip(action[0], -self.acc_limit, self.acc_limit))
        steer = float(np.clip(action[1], -self.steer_limit, self.steer_limit))

        x, y, yaw, v = self.attacker_state

        next_v = np.clip(v + a * self.dt, self.cfg["vmin"], self.cfg["vmax"])

        next_yaw = yaw + (v / self.L) * np.tan(steer) * self.dt
        next_yaw = self.normalize_angle(next_yaw)

        next_x = x + next_v * np.cos(next_yaw) * self.dt
        next_y = y + next_v * np.sin(next_yaw) * self.dt

        self.attacker_state = np.array(
            [next_x, next_y, next_yaw, next_v],
            dtype=np.float32,
        )

        ego_next_x = (
            self.ego_state[0]
            + self.ego_velocity * np.cos(self.ego_heading) * self.dt
        )
        ego_next_y = (
            self.ego_state[1]
            + self.ego_velocity * np.sin(self.ego_heading) * self.dt
        )

        self.ego_state = np.array(
            [ego_next_x, ego_next_y],
            dtype=np.float32,
        )

        current_dist = float(
            np.linalg.norm(self.attacker_state[:2] - self.ego_state)
        )

        collided = current_dist < self.cfg["collision_dist"]

        reward, reward_terms = self._compute_reward(
            current_dist=current_dist,
            collided=collided,
        )

        closing = self.prev_dist - current_dist
        self.prev_dist = current_dist

        terminated = bool(collided)
        truncated = bool(self.current_step >= self.max_steps)

        info = {
            "reward": reward,
            "dist": current_dist,
            "closing": closing,
            "collided": collided,
            **reward_terms,
        }

        return self._get_obs(), float(reward), terminated, truncated, info

    def _compute_reward(self, current_dist, collided):
        closing = self.prev_dist - current_dist

        r_approach = closing / self.cfg["approach_scale"]
        r_collision = self.cfg["collision_bonus"] if collided else 0.0

        reward = (
            self.reward_weights["approach"] * r_approach
            + self.reward_weights["collision"] * r_collision
        )

        terms = {
            "r_approach": float(r_approach),
            "r_collision": float(r_collision),
            "w_approach": float(self.reward_weights["approach"]),
            "w_collision": float(self.reward_weights["collision"]),
        }
        return float(reward), terms

    def _get_obs(self):
        ego_future_x = (
            self.ego_state[0]
            + self.ego_velocity
            * np.cos(self.ego_heading)
            * self.max_steps
            * self.dt
        )
        ego_future_y = (
            self.ego_state[1]
            + self.ego_velocity
            * np.sin(self.ego_heading)
            * self.max_steps
            * self.dt
        )

        obs = np.concatenate(
            [
                self.attacker_state,
                self.ego_state,
                np.array([ego_future_x, ego_future_y], dtype=np.float32),
            ]
        )

        return obs.astype(np.float32)

    @staticmethod
    def normalize_angle(angle):
        return (angle + np.pi) % (2 * np.pi) - np.pi

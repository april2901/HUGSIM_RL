import math
import random
import numpy as np
from trajdata.maps import VectorMap
from submodules.Pplan.Sampling.spline_planner import SplinePlanner
import torch
import time
import math
from copy import deepcopy
from utils.dynamic_utils import unicycle
from stable_baselines3 import PPO


def constant_tracking(state, path, dt):
    '''
    Args:
        state: current state of the vehicle, of size [x, y, yaw, speed]
        path: the path to follow, of size (N, [x, y, yaw])
        dt: time duration
    '''

    # find the nearest point in the path
    dists = torch.norm(path[:, :2] - state[None, :2], dim=1)
    nearest_index = torch.argmin(dists)

    # find the target point
    lookahead_distance = state[3] * dt
    target = path[-1]
    is_end = True
    for i in range(nearest_index + 1, len(path)):
        if torch.norm(path[i, :2] - state[:2]) > lookahead_distance:
            target = path[i]
            is_end = False
            break

    # compute the new state
    target_distance = torch.norm(target[:2] - state[:2])
    ratio = lookahead_distance / target_distance.clamp(min=1e-6)
    ratio = ratio.clamp(max=1.0)

    new_state = deepcopy(state)
    new_state[:2] = state[:2] + ratio * (target[:2] - state[:2])
    new_state[2] = torch.atan2(
        state[2].sin() + ratio * (target[2].sin() - state[2].sin()),
        state[2].cos() + ratio * (target[2].cos() - state[2].cos())
    )
    if is_end:
        new_state[3] = 0

    return new_state


def constant_headaway(states, num_steps, dt):
    '''
    Args:
        states: current states of a batch of vehicles, of size (num_agents, [x, y, yaw, speed])
        num_steps: number of steps to move forward
        dt: time duration
    Return:
        trajs: the trajectories of the vehicles, of size (num_agents, num_steps, [x, y, yaw, speed])
    '''

    # state: [x, y, yaw, speed]
    x = states[:, 0]
    y = states[:, 1]
    yaw = states[:, 2]
    speed = states[:, 3]

    # Generate time steps
    t_steps = torch.arange(num_steps) * dt

    # Calculate dx and dy for each step
    dx = torch.outer(speed * torch.sin(yaw), t_steps)
    dy = torch.outer(speed * torch.cos(yaw), t_steps)

    # Update x and y positions
    x_traj = x.unsqueeze(1) + dx
    y_traj = y.unsqueeze(1) + dy

    # Replicate the yaw and speed for each time step
    yaw_traj = yaw.unsqueeze(1).repeat(1, num_steps)
    speed_traj = speed.unsqueeze(1).repeat(1, num_steps)

    # Stack the x, y, yaw, and speed components to form the trajectory
    trajs = torch.stack((x_traj, y_traj, yaw_traj, speed_traj), dim=-1)

    return trajs


class IDM:
    def __init__(
            self, v0=30.0, s0=5.0, T=2.0, a=2.0, b=4.0, delta=4.0,
            lookahead_path_length=100, lead_distance_threshold=1.0
    ):
        '''
        Args:
            v0: desired speed
            s0: minimum gap
            T: safe time headway
            a: max acceleration
            b: comfortable deceleration
            delta: acceleration exponent
            lookahead_path_length: the length of path to look ahead
            lead_distance_threshold: the distance to consider a vehicle as a lead vehicle
        '''
        self.v0 = v0
        self.s0 = s0
        self.T = T
        self.a = a
        self.b = b
        self.delta = delta
        self.lookahead_path_length = lookahead_path_length
        self.lead_distance_threshold = lead_distance_threshold

    def update(self, state, path, dt, neighbors):
        '''
        Args:
            state: current state of the vehicle, of size [x, y, yaw, speed]
            path: the path to follow, of size (N, [x, y, yaw])
            dt: time duration
            neighbors: the future states of the neighbors, of size (K, T, [x, y, yaw, speed])
        '''

        if path is None:
            return deepcopy(state)

        # find the nearest point in the path
        dists = torch.norm(path[:, :2] - state[None, :2], dim=1)
        nearest_index = torch.argmin(dists)


        # lookahead_distance = state[3] * dt
        # lookahead_targe = state[:2] + np.array([np.sin(state[2]) * lookahead_distance, np.cos(state[2]) * lookahead_distance])
        # # target = path[-1]
        # is_end = False
        # target_idx = torch.argmin(torch.norm(path[:, :2] - lookahead_targe, dim=-1))
        # target = path[target_idx]

        # find the target point
        lookahead_distance = state[3] * dt
        target = path[-1]
        is_end = True
        for i in range(nearest_index + 1, len(path)):
            if torch.norm(path[i, :2] - state[:2]) > lookahead_distance:
                target = path[i]
                is_end = False
                break

        # distance between neighbors and the path
        lookahead_path = path[nearest_index + 1:][:self.lookahead_path_length]
        lookahead_neighbors = neighbors[..., None, :].expand(
            -1, -1, lookahead_path.shape[0], -1
        )  # (K, T, n, 4)

        dists_neighbors = torch.norm(
            lookahead_neighbors[..., :2] - lookahead_path[None, None, :, :2], dim=-1
        )  # (K, T, n)
        indices_neighbors = torch.arange(
            lookahead_path.shape[0]
        )[None, None].expand_as(dists_neighbors)

        # determine lead vehicles
        is_lead = (dists_neighbors < self.lead_distance_threshold)
        if is_lead.any():
            # compute lead distance
            indices_lead = indices_neighbors[is_lead]  # (num_lead)
            lookahead_lengths = torch.cumsum(torch.norm(
                lookahead_path[1:, :2] - lookahead_path[:-1, :2], dim=1
            ), dim=0)
            lookahead_lengths = torch.cat([lookahead_lengths, lookahead_lengths[-1:]])
            lead_distance = lookahead_lengths[indices_lead]

            # compute lead speed
            states_lead = lookahead_neighbors[is_lead]  # (num_lead, 4)
            ori_speed_lead = states_lead[:, 3]
            yaw_lead = states_lead[:, 2]
            yaw_path = lookahead_path[indices_lead, 2]
            lead_speed = ori_speed_lead * (yaw_lead - yaw_path).cos()

            # compute acceleration
            ego_speed = state[3]
            delta_v = ego_speed - lead_speed
            s_star = self.s0 + \
                     (ego_speed * self.T + ego_speed * delta_v / (2 * math.sqrt(self.a * self.b))).clamp(min=0)
            acceleration = self.a * (1 - (ego_speed / self.v0) ** self.delta - (s_star / lead_distance) ** 2)
            acceleration = acceleration.min()
        else:
            acceleration = self.a * (1 - (state[3] / self.v0) ** self.delta)

        # compute the new state
        target_distance = torch.norm(target[:2] - state[:2])
        ratio = lookahead_distance / target_distance.clamp(min=1e-6)
        ratio = ratio.clamp(max=1.0)

        new_state = deepcopy(state)
        new_state[:2] = state[:2] + ratio * (target[:2] - state[:2])
        new_state[2] = torch.atan2(
            state[2].sin() + ratio * (target[2].sin() - state[2].sin()),
            state[2].cos() + ratio * (target[2].cos() - state[2].cos())
        )
        if is_end:
            new_state[3] = 0
        else:
            new_state[3] = (state[3] + acceleration * dt).clamp(min=0)

        return new_state


class AttackPlanner:
    def __init__(self, pred_steps=20, ATTACK_FREQ = 3, best_k=1, device='cpu'):
        self.device = device
        self.predict_steps = pred_steps
        self.best_k = best_k

        self.planner = SplinePlanner(
            device,
            N_seg=self.predict_steps,
            acce_grid=torch.linspace(-2, 5, 10).to(self.device),
            acce_bound=[-6, 5],
            vbound=[-2, 50]
        )
        self.planner.psi_bound = [-math.pi * 2, math.pi * 2]

        self.exec_traj = None
        self.exec_pointer = 1

    def update(
            self, state, unified_map, dt,
            neighbors, attacked_states,
            new_plan=True
    ):
        '''
        Args:
            state: current state of the vehicle, of size [x, y, yaw, speed]
            vector_map: the vector map
            attacked_states: future states of the attacked agent, of size (T, [x, y, yaw, speed])
            neighbors: future states of the neighbors, of size (K, T, [x, y, yaw, speed])
            new_plan: whether to generate a new plan
        '''
        assert self.exec_pointer > 0

        # directly execute the current plan
        if not new_plan:
            if self.exec_traj is not None and \
                    self.exec_pointer < self.exec_traj.shape[0]:
                next_state = self.exec_traj[self.exec_pointer]
                self.exec_pointer += 1
                return next_state
            else:
                new_plan = True

        assert attacked_states.shape[0] == self.predict_steps

        # state: [x, y, yaw, speed]
        x, y, yaw, speed = state

        # query vector map to get lanes
        query_xyzr = np.array([x, y, 0, yaw + np.pi / 2])
        # query_xyzr = unified_map.xyzr_local2world(np.array([x, y, 0, yaw]))
        # lanes = unified_map.vector_map.get_lanes_within(query_xyzr[:3], dist=30)
        # lanes = [unified_map.batch_xyzr_world2local(l.center.xyzh)[:, [0,1,3]] for l in lanes]
        # lanes = [l.center.xyzh[:, [0,1,3]] for l in lanes]
        lanes = None

        # for lane in lanes:
        #     plt.plot(lane[:, 0], lane[:, 1], 'k--', linewidth=0.5, alpha=0.5)

        # generate spline trajectories
        x0 = torch.tensor([query_xyzr[0], query_xyzr[1], speed, query_xyzr[3]], device=self.device)
        possible_trajs, xf_set = self.planner.gen_trajectories(x0, self.predict_steps * dt, lanes,
                                                               dyn_filter=True)  # (num_trajs, T-1, [x, y, v, a, yaw, r, t])
        if possible_trajs.shape[0] == 0:
            trajs = constant_headaway(state[None], self.predict_steps, dt)  # (1, T, [x, y, yaw, speed])
        else:
            trajs = torch.cat([
                state[None, None].expand(possible_trajs.shape[0], -1, -1),
                possible_trajs[..., [0, 1, 4, 2]]
            ], dim=1)

        # select the best trajectory
        attack_distance = torch.norm(attacked_states[None, :, :2] - trajs[..., :2], dim=-1)
        cost_attack = attack_distance.min(dim=1).values
        cost_collision = (
                    torch.norm(neighbors[None, ..., :2] - trajs[:, None, :, :2], dim=-1).min(dim=-1).values < 2.0).sum(
            dim=-1)
        cost = cost_attack + 0.1 * cost_collision
        values, indices = torch.topk(cost, self.best_k, largest=False)
        random_index = torch.randint(0, self.best_k, (1,)).item()
        selected_index = indices[random_index]
        traj_best = trajs[selected_index]

        # produce next state
        self.exec_traj = traj_best
        self.exec_traj[:, 2] -= np.pi / 2
        self.exec_pointer = 1
        next_state = self.exec_traj[self.exec_pointer]
        # next_state[0] = -next_state[0]
        self.exec_pointer += 1

        return next_state


class ConstantPlanner:
    def __init__(self):
        return

    def update(self, state, dt):
        a, b, yaw, v = state
        a = a - v * np.sin(yaw) * dt
        b = b + v * np.cos(yaw) * dt
        return torch.tensor([a, b, yaw, v])
    

class UnicyclePlanner:
    def __init__(self, uc_path, speed=1.0):
        self.uc_model = unicycle.restore(torch.load(uc_path, weights_only=False))
        self.t = 0
        self.speed = speed
    
    def update(self, dt):
        self.t += dt * self.speed
        a, b, v, pitchroll, yaw, h = self.uc_model.forward(self.t)
        # return torch.tensor([a, b, yaw, v]), pitchroll.detach().cpu(), h.item()
        return torch.tensor([a, b, yaw, v])


class RLAttackPlanner:
    def __init__(self, weight_path="/mnt/sda/projects/hugsim/HUGSIM/simple_attacker_ppo.zip", dt=0.1, device='cuda'):
        self.device = device
        self.dt = dt
        
        # 1. 오프라인에서 학습된 RL 모델 가중치 불러오기
        self.model = PPO.load(weight_path, device=self.device)
        
    def update(
            self, state, unified_map, dt,
            neighbors, attacked_states,
            new_plan=True
    ):
        '''
        기존 AttackPlanner와 동일한 입력값을 받습니다.
        '''
        
        # 1. 상태 관측(Observation) 구성하기
        # HUGSIM이 주는 날것의 데이터를 RL 모델이 학습했던 형태(State Vector)로 가공합니다.
        # 예: 현재 내 위치를 기준으로 상대 좌표계 변환
        obs = self._make_observation(state, neighbors, attacked_states)
        
        # 2. RL 모델로 추론 (Inference)
        # "현재 상황이 이런데, 가속도랑 조향각을 어떻게 할까?"
        # deterministic=True 를 주어 학습된 최적의 행동만 일관되게 뽑아냅니다.
        action, _states = self.model.predict(obs, deterministic=True)
        
        # action은 예를 들어 [acceleration, steering_rate] 형태
        accel, steer = action[0], action[1]
        
        # 3. 모델의 행동을 바탕으로 다음 프레임의 위치(Next State) 계산
        # (Kinematic Bicycle Model 적용)
        next_state = self._apply_kinematics(state, accel, steer, self.dt)
        
        # HUGSIM 시뮬레이터에게 "나 다음 프레임에 여기로 갈게" 하고 반환
        return next_state

    def _make_observation(self, state, neighbors, attacked_states):
        # attacked_states: ego의 미래 궤적 (T, [x, y, yaw, v])
        # constant_headaway나 navsim에서 계산된 ego의 동적 움직임을 활용합니다.
        # AttackPlanner와 유사하게 ego의 전체 미래 궤적을 고려합니다.
        
        if torch.is_tensor(state):
            state = state.detach().cpu().numpy()
        if torch.is_tensor(attacked_states):
            attacked_states = attacked_states.detach().cpu().numpy()

        attacker_state = np.asarray(state, dtype=np.float32)
        
        if attacked_states is None or attacked_states.size == 0:
            # fallback: 기본값
            ego_info = np.zeros(4, dtype=np.float32)
        else:
            # ego의 미래 궤적에서 현재와 최종 목표 위치를 추출
            # 현재 ego 위치 + 미래 예측 위치를 고려하여 공격 전략 수립
            ego_current = attacked_states[0, :2]   # 현재 ego 위치
            ego_future = attacked_states[-1, :2]   # 최종 예상 위치
            ego_info = np.concatenate([ego_current, ego_future], axis=0).astype(np.float32)

        obs = np.concatenate([attacker_state, ego_info], axis=0).astype(np.float32)
        return obs

    def _apply_kinematics(self, state, accel, steer, dt):
        if torch.is_tensor(state):
            state = state.detach().cpu().numpy()

        x, y, yaw, v = state
        L = 2.8

        next_x = x + v * np.cos(yaw) * dt
        next_y = y + v * np.sin(yaw) * dt
        next_yaw = yaw + (v / L) * np.tan(steer) * dt
        next_v = np.clip(v + accel * dt, 0.0, 30.0)

        return torch.tensor([next_x, next_y, next_yaw, next_v], dtype=torch.float32, device=self.device)

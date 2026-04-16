import json
import random
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@dataclass
class SignalPhase:
    direction: str
    green_duration: int
    min_green: int = 10
    ped_buffer: int = 5


@dataclass
class IntersectionState:
    intersection_id: str
    queue_NS: int = 0
    queue_EW: int = 0
    current_phase: str = 'NS'
    phase_elapsed: int = 0
    throughput: int = 0


@dataclass
class TrafficProfile:
    name: str
    peak_NS: float = 8.0
    peak_EW: float = 6.0
    offpeak_NS: float = 3.0
    offpeak_EW: float = 2.5
    peak_hours: Tuple[int, int] = (7, 9)


class TrafficEnvironment:

    def __init__(self, grid_size=2, profile=None, sim_hour=8, max_queue=50, timestep_sec=5):
        assert 2 <= grid_size <= 4
        self.grid_size = grid_size
        self.n_intersections = grid_size * grid_size
        self.profile = profile or TrafficProfile(name="default")
        self.sim_hour = sim_hour
        self.max_queue = max_queue
        self.timestep_sec = timestep_sec
        self.intersections = {}
        self.time_elapsed = 0
        self.metrics_log = []
        self._build_grid()

    def _build_grid(self):
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                iid = f"I{r}{c}"
                self.intersections[iid] = IntersectionState(iid)

    def _is_peak(self):
        h = self.sim_hour % 24
        return self.profile.peak_hours[0] <= h <= self.profile.peak_hours[1]

    def _arrival_rate(self, direction):
        p = self._is_peak()
        return self.profile.peak_NS if direction == 'NS' and p else \
               self.profile.offpeak_NS if direction == 'NS' else \
               self.profile.peak_EW if p else self.profile.offpeak_EW

    def step(self, actions):
        total_wait = 0
        info = {}

        for iid, state in self.intersections.items():
            phase = actions.get(iid, SignalPhase('NS', 30))
            phase.green_duration = max(phase.green_duration, phase.min_green)

            arrivals_NS = np.random.poisson(self._arrival_rate('NS') * self.timestep_sec / 60)
            arrivals_EW = np.random.poisson(self._arrival_rate('EW') * self.timestep_sec / 60)

            state.queue_NS = min(state.queue_NS + arrivals_NS, self.max_queue)
            state.queue_EW = min(state.queue_EW + arrivals_EW, self.max_queue)

            discharge_rate = int(phase.green_duration / self.timestep_sec * 2)

            if phase.direction == 'NS':
                cleared = min(state.queue_NS, discharge_rate)
                state.queue_NS -= cleared
                wait = state.queue_EW
            else:
                cleared = min(state.queue_EW, discharge_rate)
                state.queue_EW -= cleared
                wait = state.queue_NS

            state.throughput = cleared
            state.current_phase = phase.direction
            state.phase_elapsed += self.timestep_sec
            total_wait += wait

            info[iid] = {
                "queue_NS": state.queue_NS,
                "queue_EW": state.queue_EW,
                "throughput": cleared,
                "wait": wait,
                "phase": phase.direction,
                "green_sec": phase.green_duration,
            }

        self.time_elapsed += self.timestep_sec
        reward = -total_wait

        obs = {iid: (s.queue_NS, s.queue_EW) for iid, s in self.intersections.items()}

        self.metrics_log.append({"t": self.time_elapsed, "total_wait": total_wait, "info": info})

        return obs, reward, info

    def reset(self):
        self._build_grid()
        self.time_elapsed = 0
        self.metrics_log = []
        return {iid: (0, 0) for iid in self.intersections}

    def observation_space_size(self):
        return 2 * self.n_intersections

    def action_space_size(self):
        return self.n_intersections

    def get_state_vector(self):
        vals = []
        for s in self.intersections.values():
            vals.extend([s.queue_NS / self.max_queue, s.queue_EW / self.max_queue])
        return np.array(vals, dtype=np.float32)


class FixedTimingBaseline:

    def __init__(self, green_duration=30, intersections=None):
        self.green_duration = green_duration
        self.intersections = intersections or []
        self._cycle_counter = defaultdict(int)

    def act(self):
        actions = {}
        for iid in self.intersections:
            self._cycle_counter[iid] += 1
            direction = 'NS' if self._cycle_counter[iid] % 2 == 0 else 'EW'
            actions[iid] = SignalPhase(direction, self.green_duration)
        return actions

    def reset(self):
        self._cycle_counter = defaultdict(int)


class QLearningAgent:

    def __init__(self, intersections, alpha=0.1, gamma=0.95,
                 epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.05, n_buckets=5):
        self.intersections = intersections
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.n_buckets = n_buckets
        self.q_table = defaultdict(lambda: defaultdict(float))

    def _discretise(self, queue, max_q=50):
        return min(int(queue / max_q * self.n_buckets), self.n_buckets - 1)

    def _state_key(self, obs):
        return "|".join(
            f"{self._discretise(obs[i][0])}{self._discretise(obs[i][1])}"
            for i in self.intersections
        )

    def act(self, obs):
        actions = {}
        for iid in self.intersections:
            if random.random() < self.epsilon:
                direction = random.choice(['NS', 'EW'])
            else:
                qns, qew = obs[iid]
                direction = 'NS' if qns >= qew else 'EW'
            green = max(10, min(60, 20 + max(obs[iid]) * 2))
            actions[iid] = SignalPhase(direction, green)
        return actions

    def learn(self, obs, actions, reward, next_obs):
        sk = self._state_key(obs)
        nsk = self._state_key(next_obs)
        ak = "|".join(actions[i].direction for i in self.intersections)

        max_next = max(self.q_table[nsk].values()) if self.q_table[nsk] else 0.0
        old = self.q_table[sk][ak]

        self.q_table[sk][ak] = old + self.alpha * (reward + self.gamma * max_next - old)
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


class SimulationRunner:

    def __init__(self, grid_size=2, profile=None, sim_hour=8,
                 steps_per_episode=120, n_episodes=200, use_dqn=False):

        self.env = TrafficEnvironment(grid_size, profile, sim_hour)
        iids = list(self.env.intersections.keys())

        self.agent = QLearningAgent(iids)
        self.baseline = FixedTimingBaseline(30, iids)

        self.steps_per_episode = steps_per_episode
        self.n_episodes = n_episodes

        self.ai_rewards = []
        self.baseline_rewards = []
        self.ai_throughputs = []
        self.baseline_throughputs = []

        self.best_plan = None
        self.best_reward = float('-inf')

    def _run_episode_ai(self):
        obs = self.env.reset()
        total_reward = 0
        total_tp = 0

        for _ in range(self.steps_per_episode):
            actions = self.agent.act(obs)
            next_obs, reward, info = self.env.step(actions)
            self.agent.learn(obs, actions, reward, next_obs)
            obs = next_obs
            total_reward += reward
            total_tp += sum(v["throughput"] for v in info.values())

        return total_reward, total_tp

    def _run_episode_baseline(self):
        self.env.reset()
        self.baseline.reset()

        total_reward = 0
        total_tp = 0

        for _ in range(self.steps_per_episode):
            actions = self.baseline.act()
            _, reward, info = self.env.step(actions)
            total_reward += reward
            total_tp += sum(v["throughput"] for v in info.values())

        return total_reward, total_tp

    def run(self):
        for _ in range(self.n_episodes):
            ai_r, ai_tp = self._run_episode_ai()
            base_r, base_tp = self._run_episode_baseline()

            self.ai_rewards.append(ai_r)
            self.baseline_rewards.append(base_r)
            self.ai_throughputs.append(ai_tp)
            self.baseline_throughputs.append(base_tp)

        return {
            "avg_wait_reduction_pct": np.mean(self.baseline_rewards) - np.mean(self.ai_rewards)
        }


def main():
    profile = TrafficProfile(name="peak_morning")
    runner = SimulationRunner(profile=profile)
    summary = runner.run()
    print(summary)


if __name__ == "__main__":
    main()
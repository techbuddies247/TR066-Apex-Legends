"""
================================================
AI-Driven Urban Traffic Signal Optimization
Simulator - TENSOR '26 | PS16
Team: APEX LEGENDS
================================================

ARCHITECTURE OVERVIEW
---------------------
1. TrafficEnvironment   - SUMO-style grid simulation (2x2 to 4x4)
2. TrafficAgent         - RL agent (Q-learning / DQN) for signal control
3. FixedTimingBaseline  - Traditional fixed-phase controller
4. SimulationRunner     - Orchestrates env + agent, collects metrics
5. Visualizer           - Side-by-side throughput & queue plots
6. JSONExporter         - Exports optimized signal plan for controllers

Dependencies:
    pip install numpy matplotlib gymnasium torch
"""

import json
import random
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import requests

API_KEY = "J8Yf2b8hjemZoq37qDCSZ5zrZ48CNo98"

def fetch_live_data_from_api(env):
    data = {}

    for iid in env.intersections:
        # Example coordinates (replace with real ones)
        lat, lon = 28.6139, 77.2090  

        url = f"https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/10/json"
        params = {
            "point": f"{lat},{lon}",
            "key": API_KEY
        }

        response = requests.get(url, params=params).json()

        try:
            flow = response["flowSegmentData"]
            speed = flow["currentSpeed"]
            free_speed = flow["freeFlowSpeed"]

            congestion = 1 - (speed / free_speed)

            data[iid] = {
                "NS": int(congestion * env.max_queue),
                "EW": int((1 - congestion) * env.max_queue)
            }

        except:
            data[iid] = {"NS": 10, "EW": 10}

    return data
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@dataclass
class SignalPhase:
    """A single green-phase duration for one intersection approach."""
    direction: str          # 'NS' or 'EW'
    green_duration: int     # seconds
    min_green: int = 10     # pedestrian / safety minimum
    ped_buffer: int = 5     # all-red clearance buffer


@dataclass
class IntersectionState:
    """Snapshot of one intersection at a given timestep."""
    intersection_id: str
    queue_NS: int = 0       # vehicles queued on N-S approaches
    queue_EW: int = 0       # vehicles queued on E-W approaches
    current_phase: str = 'NS'
    phase_elapsed: int = 0
    throughput: int = 0     # vehicles cleared this cycle


@dataclass
class TrafficProfile:
    """
    Vehicle arrival rates (vehicles/min/lane) for peak & off-peak.
    Matches the problem's 'user-defined or pre-loaded scenarios'.
    """
    name: str
    peak_NS: float = 8.0
    peak_EW: float = 6.0
    offpeak_NS: float = 3.0
    offpeak_EW: float = 2.5
    peak_hours: Tuple[int, int] = (7, 9)   # 07:00-09:00


class TrafficEnvironment:
    """
    Simulates a configurable NxN intersection grid.
    - State  : queue lengths per lane per intersection
    - Action : green-time allocation per intersection
    - Reward : negative total wait time (minimise congestion)
    """
    def update_from_live_data(self, live_data: Dict[str, Dict[str, int]]):
        """
        Inject real-time queue data into environment
        Format:
        {
            "I00": {"NS": 10, "EW": 5},
            ...
        }
        """
        for iid, state in self.intersections.items():
            if iid in live_data:
                state.queue_NS = min(live_data[iid]["NS"], self.max_queue)
                state.queue_EW = min(live_data[iid]["EW"], self.max_queue)
    def __init__(
        self,
        grid_size: int = 2,
        profile: Optional[TrafficProfile] = None,
        sim_hour: int = 8,          # simulated hour (for peak/offpeak)
        max_queue: int = 50,
        timestep_sec: int = 5,
    ):
        assert 2 <= grid_size <= 4, "Grid must be 2x2 to 4x4"
        self.grid_size = grid_size
        self.n_intersections = grid_size * grid_size
        self.profile = profile or TrafficProfile(name="default")
        self.sim_hour = sim_hour
        self.max_queue = max_queue
        self.timestep_sec = timestep_sec

        self.intersections: Dict[str, IntersectionState] = {}
        self.time_elapsed = 0           # seconds
        self.metrics_log: List[dict] = []

        self._build_grid()

    def _build_grid(self):
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                iid = f"I{r}{c}"
                self.intersections[iid] = IntersectionState(
                    intersection_id=iid,
                    current_phase='NS',
                    phase_elapsed=0,
                )

    def _is_peak(self) -> bool:
        h = self.sim_hour % 24
        return self.profile.peak_hours[0] <= h <= self.profile.peak_hours[1]

    def _arrival_rate(self, direction: str) -> float:
        p = self._is_peak()
        if direction == 'NS':
            return self.profile.peak_NS if p else self.profile.offpeak_NS
        return self.profile.peak_EW if p else self.profile.offpeak_EW
    def fetch_live_data(env):
        data = {}
        for iid in env.intersections:
            data[iid] = {
                "NS": random.randint(0, env.max_queue),
                "EW": random.randint(0, env.max_queue),
            }
        return data
    def step(self, actions: Dict[str, SignalPhase]) -> Tuple[dict, float, dict]:
        """
        Apply one set of signal phases, advance by timestep_sec.

        Parameters
        ----------
        actions : {intersection_id: SignalPhase}

        Returns
        -------
        observation : dict  - new queue state
        reward      : float - negative total wait
        info        : dict  - per-intersection metrics
        """
        total_wait = 0
        info = {}

        for iid, state in self.intersections.items():
            phase = actions.get(iid, SignalPhase(direction='NS', green_duration=30))
            phase.green_duration = max(phase.green_duration, phase.min_green)

            arrivals_NS = np.random.poisson(
                self._arrival_rate('NS') * self.timestep_sec / 60
            )
            arrivals_EW = np.random.poisson(
                self._arrival_rate('EW') * self.timestep_sec / 60
            )

            state.queue_NS = min(state.queue_NS + arrivals_NS, self.max_queue)
            state.queue_EW = min(state.queue_EW + arrivals_EW, self.max_queue)

            discharge_rate = int(phase.green_duration / self.timestep_sec * 2)
            if phase.direction == 'NS':
                cleared = min(state.queue_NS, discharge_rate)
                state.queue_NS -= cleared
                wait = state.queue_EW   # EW is waiting
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
        reward = -total_wait + (0.5 * sum(v["throughput"] for v in info.values()))

        obs = {
            iid: (s.queue_NS, s.queue_EW)
            for iid, s in self.intersections.items()
        }
        self.metrics_log.append({
            "t": self.time_elapsed,
            "total_wait": total_wait,
            "info": info,
        })
        return obs, reward, info

    def reset(self):
        self._build_grid()
        self.time_elapsed = 0
        self.metrics_log = []
        return {iid: (0, 0) for iid in self.intersections}

    def observation_space_size(self) -> int:
        """Flat state vector length: 2 queue values x N intersections."""
        return 2 * self.n_intersections

    def action_space_size(self) -> int:
        """Binary action per intersection: 0=NS green, 1=EW green."""
        return self.n_intersections

    def get_state_vector(self) -> np.ndarray:
        vals = []
        for s in self.intersections.values():
            vals.extend([s.queue_NS / self.max_queue,
                         s.queue_EW / self.max_queue])
        return np.array(vals, dtype=np.float32)


class FixedTimingBaseline:
    """
    Traditional controller: alternates NS/EW on a fixed cycle.
    Used as the comparison baseline for evaluation metrics.
    """

    def __init__(self, green_duration: int = 30, intersections: List[str] = None):
        self.green_duration = green_duration
        self.intersections = intersections or []
        self._cycle_counter = defaultdict(int)

    def act(self) -> Dict[str, SignalPhase]:
        actions = {}
        for iid in self.intersections:
            self._cycle_counter[iid] += 1
            direction = 'NS' if self._cycle_counter[iid] % 2 == 0 else 'EW'
            actions[iid] = SignalPhase(
                direction=direction,
                green_duration=self.green_duration,
            )
        return actions

    def reset(self):
        self._cycle_counter = defaultdict(int)


class QLearningAgent:
    """
    Tabular Q-learning - fast, no GPU needed.
    State: discretised queue buckets per intersection.
    Action: NS-green or EW-green per intersection (independent).
    """
    def act(self, obs: dict) -> Dict[str, SignalPhase]:
        actions = {}
        sk = self._state_key(obs)

        for iid in self.intersections:
            if random.random() < self.epsilon:
                direction = random.choice(['NS', 'EW'])
            else:
                # Use learned Q-values
                q_values = self.q_table[sk]
                if not q_values:
                    direction = random.choice(['NS', 'EW'])
                else:
                    best_action = max(q_values, key=q_values.get)
                    direction = best_action.split("|")[0]  # simple mapping

            green = self._compute_green(obs.get(iid, (0, 0)), direction)
            actions[iid] = SignalPhase(direction=direction, green_duration=green)

        return actions
    def __init__(
        self,
        intersections: List[str],
        alpha: float = 0.1,
        gamma: float = 0.95,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.05,
        n_buckets: int = 5,
    ):
        self.intersections = intersections
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.n_buckets = n_buckets

        self.q_table: Dict[str, Dict[str, float]] = defaultdict(
            lambda: defaultdict(float)
        )
    class RealTimeRunner:

        def __init__(self, env, agent):
            self.env = env
            self.agent = agent

        def run(self, steps=50, delay=5):
            print("\n🚦 REAL-TIME MODE STARTED\n")

            obs = self.env.reset()

            for step in range(steps):

                # 1. Fetch live data
                live_data = fetch_live_data(self.env)

                # 2. Inject into environment
                self.env.update_from_live_data(live_data)

                # 3. Get state
                obs = {
                    iid: (s.queue_NS, s.queue_EW)
                    for iid, s in self.env.intersections.items()
                }

                # 4. Agent decision
                actions = self.agent.act(obs)

                # 5. Apply actions
                _, reward, info = self.env.step(actions)

                # 6. Print (for demo)
                print(f"\n--- Step {step+1} ---")
                for iid, v in info.items():
                    print(f"{iid} | NS:{v['queue_NS']:2d} EW:{v['queue_EW']:2d} -> {v['phase']}")

                print(f"Reward: {reward}")

                time.sleep(delay)
    def _discretise(self, queue: int, max_q: int = 50) -> int:
        return min(int(queue / max_q * self.n_buckets), self.n_buckets - 1)

    def _state_key(self, obs: dict) -> str:
        parts = []
        for iid in self.intersections:
            qns, qew = obs.get(iid, (0, 0))
            parts.append(f"{self._discretise(qns)}{self._discretise(qew)}")
        return "|".join(parts)

    def _action_key(self, actions: Dict[str, str]) -> str:
        return "|".join(actions[iid] for iid in self.intersections)

    def act(self, obs: dict, max_queue: int = 50) -> Dict[str, SignalPhase]:
        """e-greedy policy."""
        actions = {}
        for iid in self.intersections:
            if random.random() < self.epsilon:
                direction = random.choice(['NS', 'EW'])
            else:
                qns, qew = obs.get(iid, (0, 0))
                direction = 'NS' if qns >= qew else 'EW'
            green = self._compute_green(obs.get(iid, (0, 0)), direction)
            actions[iid] = SignalPhase(direction=direction, green_duration=green)
        return actions

    def _compute_green(self, queues: Tuple[int, int], direction: str,
                       base: int = 20, scale: int = 2) -> int:
        """Proportional green time based on queue length."""
        qns, qew = queues
        q = qns if direction == 'NS' else qew
        

    def learn(self, obs: dict, actions: Dict[str, SignalPhase],
              reward: float, next_obs: dict):
        sk = self._state_key(obs)
        nsk = self._state_key(next_obs)
        ak = "|".join(a.direction for a in
                      [actions[i] for i in self.intersections])

        next_vals = self.q_table[nsk]
        max_next = max(next_vals.values()) if next_vals else 0.0

        old = self.q_table[sk][ak]
        self.q_table[sk][ak] = old + self.alpha * (
            reward + self.gamma * max_next - old
        )
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def reset_epsilon(self):
        self.epsilon = 1.0


if TORCH_AVAILABLE:

    class DQNNetwork(nn.Module):
        def __init__(self, state_dim: int, action_dim: int):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(state_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, action_dim),
            )

        def forward(self, x):
            return self.net(x)


    class DQNAgent:
        """
        Deep Q-Network agent for larger grid sizes (3x3, 4x4).
        Uses experience replay + target network for stability.
        """

        def __init__(
            self,
            state_dim: int,
            action_dim: int,
            lr: float = 1e-3,
            gamma: float = 0.95,
            epsilon: float = 1.0,
            epsilon_decay: float = 0.997,
            epsilon_min: float = 0.05,
            batch_size: int = 64,
            memory_size: int = 10_000,
            target_update_freq: int = 50,
        ):
            self.state_dim = state_dim
            self.action_dim = action_dim
            self.gamma = gamma
            self.epsilon = epsilon
            self.epsilon_decay = epsilon_decay
            self.epsilon_min = epsilon_min
            self.batch_size = batch_size
            self.target_update_freq = target_update_freq
            self.step_count = 0

            self.policy_net = DQNNetwork(state_dim, action_dim)
            self.target_net = DQNNetwork(state_dim, action_dim)
            self.target_net.load_state_dict(self.policy_net.state_dict())
            self.target_net.eval()

            self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
            self.memory = deque(maxlen=memory_size)

        def act(self, state: np.ndarray) -> np.ndarray:
            """Returns binary action array: 0=NS, 1=EW per intersection."""
            if random.random() < self.epsilon:
                return np.random.randint(0, 2, size=self.action_dim)
            with torch.no_grad():
                q = self.policy_net(torch.FloatTensor(state))
            return (q.numpy() > 0).astype(int)

        def remember(self, s, a, r, ns):
            self.memory.append((s, a, r, ns))

        def replay(self):
            if len(self.memory) < self.batch_size:
                return
            batch = random.sample(self.memory, self.batch_size)
            states, actions, rewards, next_states = zip(*batch)

            S  = torch.FloatTensor(np.array(states))
            A  = torch.LongTensor(np.array(actions))
            R  = torch.FloatTensor(np.array(rewards))
            NS = torch.FloatTensor(np.array(next_states))

            Q_vals  = self.policy_net(S)
            Q_next  = self.target_net(NS).detach()

            target = R.unsqueeze(1).expand_as(Q_vals) + \
                     self.gamma * Q_next

            loss = nn.functional.mse_loss(Q_vals, target)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.step_count += 1
            if self.step_count % self.target_update_freq == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())

            self.epsilon = max(self.epsilon_min,
                               self.epsilon * self.epsilon_decay)


class SimulationRunner:
    """
    Orchestrates the environment + agent for N episodes.
    Collects metrics for both AI-optimized and fixed-baseline runs.
    Targets < 10 seconds per optimization cycle (PS16 requirement).
    """

    def __init__(
        self,
        grid_size: int = 2,
        profile: Optional[TrafficProfile] = None,
        sim_hour: int = 8,
        steps_per_episode: int = 120,   # 120 x 5s = 10 min simulation
        n_episodes: int = 200,
        use_dqn: bool = False,
    ):
        self.grid_size = grid_size
        self.steps_per_episode = steps_per_episode
        self.n_episodes = n_episodes
        self.use_dqn = use_dqn and TORCH_AVAILABLE

        self.env = TrafficEnvironment(
            grid_size=grid_size,
            profile=profile,
            sim_hour=sim_hour,
        )
        iids = list(self.env.intersections.keys())

        if self.use_dqn:
            self.agent = DQNAgent(
                state_dim=self.env.observation_space_size(),
                action_dim=self.env.action_space_size(),
            )
        else:
            self.agent = QLearningAgent(intersections=iids)

        self.baseline = FixedTimingBaseline(
            green_duration=30,
            intersections=iids,
        )

        self.ai_rewards: List[float] = []
        self.baseline_rewards: List[float] = []
        self.ai_throughputs: List[float] = []
        self.baseline_throughputs: List[float] = []
        self.best_plan: Optional[Dict] = None
        self.best_reward = float('-inf')

    def _run_episode_ai(self) -> Tuple[float, float]:
        obs = self.env.reset()
        total_reward = 0.0
        total_throughput = 0.0
        state_vec = self.env.get_state_vector()

        for _ in range(self.steps_per_episode):
            if self.use_dqn:
                action_arr = self.agent.act(state_vec)
                actions = {}
                for i, iid in enumerate(self.env.intersections):
                    direction = 'EW' if action_arr[i] else 'NS'
                    actions[iid] = SignalPhase(direction=direction, green_duration=30)
            else:
                actions = self.agent.act(obs)

            next_obs, reward, info = self.env.step(actions)
            next_state_vec = self.env.get_state_vector()

            if self.use_dqn:
                self.agent.remember(state_vec, action_arr, reward, next_state_vec)
                self.agent.replay()
                state_vec = next_state_vec
            else:
                self.agent.learn(obs, actions, reward, next_obs)

            obs = next_obs
            total_reward += reward
            total_throughput += sum(v["throughput"] for v in info.values())

        return total_reward, total_throughput

    def _run_episode_baseline(self) -> Tuple[float, float]:
        self.env.reset()
        self.baseline.reset()
        total_reward = 0.0
        total_throughput = 0.0

        for _ in range(self.steps_per_episode):
            actions = self.baseline.act()
            _, reward, info = self.env.step(actions)
            total_reward += reward
            total_throughput += sum(v["throughput"] for v in info.values())

        return total_reward, total_throughput

    def run(self, verbose: bool = True) -> dict:
        print(f"\n{'='*55}")
        print(f"  TENSOR '26 - Traffic Signal Optimizer (PS16)")
        print(f"  Grid: {self.grid_size}x{self.grid_size}  |  "
              f"Agent: {'DQN' if self.use_dqn else 'Q-Learning'}  |  "
              f"Episodes: {self.n_episodes}")
        print(f"{'='*55}\n")

        for ep in range(self.n_episodes):
            t0 = time.perf_counter()

            ai_r, ai_tp   = self._run_episode_ai()
            base_r, base_tp = self._run_episode_baseline()

            cycle_time = time.perf_counter() - t0

            self.ai_rewards.append(ai_r)
            self.baseline_rewards.append(base_r)
            self.ai_throughputs.append(ai_tp)
            self.baseline_throughputs.append(base_tp)

            if ai_r > self.best_reward:
                self.best_reward = ai_r
                self.best_plan = self._capture_signal_plan()

            if verbose and (ep + 1) % 20 == 0:
                imp = ((ai_r - base_r) / (abs(base_r) + 1e-9)) * 100
                print(
                    f"  Ep {ep+1:>4}/{self.n_episodes}  |  "
                    f"AI reward: {ai_r:>8.1f}  |  "
                    f"Baseline: {base_r:>8.1f}  |  "
                    f"Diff: {imp:+.1f}%  |  "
                    f"Cycle: {cycle_time:.2f}s"
                )

        return self._compute_summary()

    def _compute_summary(self) -> dict:
        ai_wait   = [-r for r in self.ai_rewards]
        base_wait = [-r for r in self.baseline_rewards]

        ai_avg   = np.mean(ai_wait[-20:])
        base_avg = np.mean(base_wait[-20:])
        wait_reduction = (base_avg - ai_avg) / (base_avg + 1e-9) * 100

        ai_tp   = np.mean(self.ai_throughputs[-20:])
        base_tp = np.mean(self.baseline_throughputs[-20:])
        tp_improvement = (ai_tp - base_tp) / (base_tp + 1e-9) * 100

        summary = {
            "grid_size": f"{self.grid_size}x{self.grid_size}",
            "agent": "DQN" if self.use_dqn else "Q-Learning",
            "avg_wait_reduction_pct": round(wait_reduction, 2),
            "throughput_improvement_veh_per_hour": round(tp_improvement, 2),
            "best_ai_reward": round(self.best_reward, 2),
            "best_signal_plan": self.best_plan,
        }

        print(f"\n{'-'*55}")
        print(f"  RESULTS SUMMARY")
        print(f"  Avg wait reduction vs baseline : {wait_reduction:.1f}%")
        print(f"  Throughput improvement         : {tp_improvement:.1f}%")
        print(f"{'-'*55}\n")
        return summary

    def _capture_signal_plan(self) -> Dict:
        """Snapshot current agent policy as an exportable signal plan."""
        plan = {}
        for iid, state in self.env.intersections.items():
            qns, qew = state.queue_NS, state.queue_EW
            direction = 'NS' if qns >= qew else 'EW'
            green = max(10, min(60, 20 + max(qns, qew) * 2))
            plan[iid] = {
                "phase": direction,
                "green_duration_sec": green,
                "ped_buffer_sec": 5,
            }
        return plan


class Visualizer:
    """
    Produces side-by-side plots comparing AI-optimized vs fixed baseline.
    Satisfies the problem output: 'Average wait time reduction (%)
    and queue length visualization'.
    """

    @staticmethod
    def plot(runner: SimulationRunner, save_path: str = "results.png"):
        fig = plt.figure(figsize=(14, 9), facecolor='#0f1117')
        gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.32)

        ORANGE = '#FF6B35'
        BLUE   = '#4FC3F7'
        BG     = '#0f1117'
        PANEL  = '#1a1d27'
        GRID_C = '#2a2d3a'

        def style_ax(ax, title):
            ax.set_facecolor(PANEL)
            ax.set_title(title, color='white', fontsize=11, fontweight='bold', pad=8)
            ax.tick_params(colors='#aaaaaa', labelsize=8)
            ax.spines[:].set_color(GRID_C)
            ax.grid(True, color=GRID_C, linewidth=0.5, alpha=0.7)
            ax.yaxis.label.set_color('#aaaaaa')
            ax.xaxis.label.set_color('#aaaaaa')

        eps = list(range(1, len(runner.ai_rewards) + 1))

        ax1 = fig.add_subplot(gs[0, 0])
        style_ax(ax1, "Total Wait Time per Episode")
        ai_wait   = [-r for r in runner.ai_rewards]
        base_wait = [-r for r in runner.baseline_rewards]
        window = max(1, len(eps) // 20)
        ai_smooth   = np.convolve(ai_wait,   np.ones(window)/window, mode='valid')
        base_smooth = np.convolve(base_wait, np.ones(window)/window, mode='valid')
        ax1.plot(eps[:len(base_smooth)], base_smooth, color=ORANGE,
                 linewidth=1.8, label='Fixed timing', alpha=0.85)
        ax1.plot(eps[:len(ai_smooth)],   ai_smooth,   color=BLUE,
                 linewidth=1.8, label='AI optimized', alpha=0.85)
        ax1.fill_between(eps[:len(ai_smooth)], ai_smooth, base_smooth[:len(ai_smooth)],
                         alpha=0.12, color=BLUE)
        ax1.set_xlabel("Episode")
        ax1.set_ylabel("Total wait (vehicle-steps)")
        ax1.legend(facecolor=PANEL, edgecolor=GRID_C,
                   labelcolor='white', fontsize=8)

        ax2 = fig.add_subplot(gs[0, 1])
        style_ax(ax2, "Throughput (Vehicles Cleared)")
        ai_tp_s   = np.convolve(runner.ai_throughputs,
                                np.ones(window)/window, mode='valid')
        base_tp_s = np.convolve(runner.baseline_throughputs,
                                np.ones(window)/window, mode='valid')
        ax2.plot(eps[:len(base_tp_s)], base_tp_s, color=ORANGE,
                 linewidth=1.8, label='Fixed timing', alpha=0.85)
        ax2.plot(eps[:len(ai_tp_s)],   ai_tp_s,   color=BLUE,
                 linewidth=1.8, label='AI optimized', alpha=0.85)
        ax2.set_xlabel("Episode")
        ax2.set_ylabel("Vehicles cleared / episode")
        ax2.legend(facecolor=PANEL, edgecolor=GRID_C,
                   labelcolor='white', fontsize=8)

        ax3 = fig.add_subplot(gs[1, 0])
        style_ax(ax3, "Queue Length Heatmap - AI (last episode)")
        n = runner.grid_size
        qmap = np.zeros((n, n))
        for iid, s in runner.env.intersections.items():
            r, c = int(iid[1]), int(iid[2])
            qmap[r, c] = s.queue_NS + s.queue_EW
        im = ax3.imshow(qmap, cmap='YlOrRd', vmin=0, vmax=runner.env.max_queue)
        for r in range(n):
            for c in range(n):
                ax3.text(c, r, f"{int(qmap[r,c])}", ha='center', va='center',
                         fontsize=10, fontweight='bold', color='#1a1d27')
        ax3.set_xticks(range(n)); ax3.set_yticks(range(n))
        ax3.set_xticklabels([f"Col {i}" for i in range(n)])
        ax3.set_yticklabels([f"Row {i}" for i in range(n)])
        plt.colorbar(im, ax=ax3, fraction=0.04, pad=0.04).ax.tick_params(
            colors='#aaaaaa', labelsize=7)

        ax4 = fig.add_subplot(gs[1, 1])
        style_ax(ax4, "Performance vs Fixed Baseline")
        metrics = ['Wait\nreduction', 'Throughput\nimprovement']
        ai_avg_wait   = np.mean(ai_wait[-20:])
        base_avg_wait = np.mean(base_wait[-20:])
        wait_pct = (base_avg_wait - ai_avg_wait) / (base_avg_wait + 1e-9) * 100
        ai_avg_tp   = np.mean(runner.ai_throughputs[-20:])
        base_avg_tp = np.mean(runner.baseline_throughputs[-20:])
        tp_pct = (ai_avg_tp - base_avg_tp) / (base_avg_tp + 1e-9) * 100
        bars = ax4.bar(metrics, [wait_pct, tp_pct],
                       color=[BLUE, ORANGE], width=0.45, zorder=3)
        for bar, val in zip(bars, [wait_pct, tp_pct]):
            ax4.text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + 0.5,
                     f"{val:.1f}%", ha='center', va='bottom',
                     color='white', fontsize=10, fontweight='bold')
        ax4.set_ylabel("Improvement (%)")
        ax4.axhline(0, color=GRID_C, linewidth=1)

        fig.suptitle(
            f"AI Traffic Signal Optimizer - {runner.grid_size}x{runner.grid_size} Grid  |  "
            f"TENSOR '26 PS16",
            color='white', fontsize=13, fontweight='bold', y=0.98
        )

        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=BG)
        print(f"  Plot saved -> {save_path}")
        plt.show()


class JSONExporter:
    """
    Exports the best signal plan to JSON compatible with
    real-world traffic controllers (PS16 requirement).
    """

    @staticmethod
    def export(summary: dict, path: str = "signal_plan.json"):
        payload = {
            "metadata": {
                "generated_by": "TENSOR-26-PS16-AI-Optimizer",
                "team": "APEX LEGENDS",
                "grid_size": summary["grid_size"],
                "agent": summary["agent"],
            },
            "performance": {
                "avg_wait_reduction_pct": summary["avg_wait_reduction_pct"],
                "throughput_improvement_pct": summary["throughput_improvement_veh_per_hour"],
            },
            "signal_plan": summary.get("best_signal_plan", {}),
        }
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"  Signal plan exported -> {path}")
        return payload


def main():
    GRID_SIZE   = 2          # 2, 3, or 4  (2x2 to 4x4)
    SIM_HOUR    = 8          # 8 = peak morning traffic
    N_EPISODES  = 200
    USE_DQN     = False      # True -> DQN (requires PyTorch), False -> Q-Learning

    profile = TrafficProfile(
        name="peak_morning",
        peak_NS=9.0,
        peak_EW=7.0,
        offpeak_NS=3.5,
        offpeak_EW=2.0,
        peak_hours=(7, 9),
    )

    runner = SimulationRunner(
        grid_size=GRID_SIZE,
        profile=profile,
        sim_hour=SIM_HOUR,
        steps_per_episode=120,
        n_episodes=N_EPISODES,
        use_dqn=USE_DQN,
    )

    summary = runner.run(verbose=True)

    Visualizer.plot(runner, save_path="traffic_results.png")

    JSONExporter.export(summary, path="signal_plan.json")

    print("\nDone. Files written:")
    print("  traffic_results.png   <- side-by-side comparison plots")
    print("  signal_plan.json      <- exportable controller plan")


if __name__ == "__main__":
    main()
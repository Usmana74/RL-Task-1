"""Smart Traffic Signal Control using Q-Learning (MVP).
Run:  streamlit run app.py
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st

MAXQ = 10         # queue length state 0..10 (cars, capped)
DISCHARGE = 2     # cars leaving per step on green
EPISODE_LEN = 200


class Intersection:
    """2-road intersection. Phase 0 = NS green, 1 = EW green."""
    def __init__(self, p_ns=0.3, p_ew=0.5, seed=None):
        self.p = (p_ns, p_ew)
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self):
        self.q = [0, 0]
        self.phase = 0
        self.t = 0
        return self._state()

    def _state(self):
        return (min(self.q[0] // 1, MAXQ), min(self.q[1] // 1, MAXQ), self.phase)

    def step(self, action):
        lost = 0
        if action == 1:                 # switch phase -> 1 step lost (yellow/all-red)
            self.phase = 1 - self.phase
            lost = 1
        if not lost:
            g = self.phase
            self.q[g] = max(0, self.q[g] - DISCHARGE)
        for i in range(2):              # random arrivals (up to 2 per step)
            self.q[i] += self.rng.binomial(2, self.p[i] / 2)
        self.q = [min(x, 30) for x in self.q]
        self.t += 1
        reward = -(self.q[0] + self.q[1]) - 0.5 * lost
        return self._state(), reward, self.t >= EPISODE_LEN


def train(episodes, alpha, gamma, eps_start, eps_end, p_ns, p_ew, seed=0):
    rng = np.random.default_rng(seed)
    Q = np.zeros((MAXQ + 1, MAXQ + 1, 2, 2))
    env = Intersection(p_ns, p_ew, seed)
    rewards = []
    for ep in range(episodes):
        eps = max(eps_end, eps_start - (eps_start - eps_end) * ep / (0.8 * episodes))
        s, total, done = env.reset(), 0, False
        while not done:
            a = rng.integers(2) if rng.random() < eps else int(np.argmax(Q[s]))
            s2, r, done = env.step(a)
            # Q-learning update: Q(s,a) += a*(r + g*max Q(s',.) - Q(s,a))
            Q[s][a] += alpha * (r + gamma * np.max(Q[s2]) - Q[s][a])
            s, total = s2, total + r
        rewards.append(total)
    return Q, rewards


def evaluate(policy, p_ns, p_ew, runs=30):
    avg = []
    for k in range(runs):
        env = Intersection(p_ns, p_ew, 1000 + k)
        s, done, qsum = env.reset(), False, 0
        while not done:
            s, _, done = env.step(policy(s, env))
            qsum += sum(env.q)
        avg.append(qsum / EPISODE_LEN)
    return float(np.mean(avg))


def fixed_policy(period=10):
    return lambda s, env: 1 if env.t % period == period - 1 else 0


def q_policy(Q):
    return lambda s, env: int(np.argmax(Q[s]))


st.set_page_config(page_title="RL Traffic Signal", page_icon="🚦")
st.title("🚦 Smart Traffic Signal Control with Q-Learning")
st.caption("MVP: a tabular Q-learning agent learns when to switch a traffic light to minimise waiting cars.")

with st.sidebar:
    st.header("Traffic")
    p_ns = st.slider("North-South arrival rate", 0.05, 0.9, 0.25)
    p_ew = st.slider("East-West arrival rate", 0.05, 0.9, 0.60)
    st.header("Q-Learning")
    episodes = st.slider("Training episodes", 100, 3000, 1500, 100)
    alpha = st.slider("Learning rate (alpha)", 0.01, 1.0, 0.1)
    gamma = st.slider("Discount (gamma)", 0.5, 0.99, 0.95)
    eps_start = st.slider("Initial exploration (epsilon)", 0.1, 1.0, 1.0)

if st.button("Train agent and compare", type="primary"):
    with st.spinner("Training..."):
        Q, rewards = train(episodes, alpha, gamma, eps_start, 0.05, p_ns, p_ew)
        res = {
            "Fixed timer (10 steps)": evaluate(fixed_policy(10), p_ns, p_ew),
            "Fixed timer (5 steps)": evaluate(fixed_policy(5), p_ns, p_ew),
            "Always keep phase": evaluate(lambda s, e: 0, p_ns, p_ew),
            "Q-Learning agent": evaluate(q_policy(Q), p_ns, p_ew),
        }
    base = min(res["Fixed timer (10 steps)"], res["Fixed timer (5 steps)"])
    imp = 100 * (base - res["Q-Learning agent"]) / base
    c1, c2, c3 = st.columns(3)
    c1.metric("Best fixed timer: avg queue", f"{base:.2f}")
    c2.metric("Q-Learning: avg queue", f"{res['Q-Learning agent']:.2f}")
    c3.metric("Improvement", f"{imp:.1f}%")

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
    k = max(1, episodes // 50)
    sm = np.convolve(rewards, np.ones(k) / k, mode="valid")
    ax[0].plot(sm); ax[0].set_title("Learning curve (smoothed reward)")
    ax[0].set_xlabel("Episode"); ax[0].set_ylabel("Total reward")
    ax[1].bar(range(4), list(res.values()), color=["gray", "silver", "lightcoral", "seagreen"])
    ax[1].set_xticks(range(4)); ax[1].set_xticklabels(["Fixed 10", "Fixed 5", "Never switch", "Q-Learn"])
    ax[1].set_title("Average cars waiting (lower = better)")
    plt.tight_layout(); st.pyplot(fig)

    st.subheader("Learned policy (when the agent switches)")
    st.write("Rows = NS queue length (cars), columns = EW queue length (cars). 1 = switch, 0 = keep. Shown for the current phase.")
    for ph, name in [(0, "NS currently green"), (1, "EW currently green")]:
        st.write(f"**{name}**")
        st.dataframe(np.argmax(Q[:, :, ph, :], axis=2))
else:
    st.info("Set parameters in the sidebar and press **Train agent and compare**.")

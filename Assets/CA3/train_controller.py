import json
from pathlib import Path
import time
import cma
import numpy as np
import torch
from torch import nn
from train import make_model
ROOT = Path(__file__).resolve().parent / "Data"
STATE = 28
ACTION = 14
FEAT = STATE
N_PARAMS = ACTION * FEAT + ACTION

RANDOM_SCALE = 0.35

VEL_PENALTY = 0.05
ACTION_SMOOTH_PENALTY = 0.1
ACTION_SMOOTH = 0.2       # must match WorldModelController.actionSmoothing in Unity
MAX_DELTA = 0.05          # must match WorldModelController.maxDeltaPerStep in Unity
HORIZON = 200
EPISODES = 16
CMA_ITERS = 150
SIGMA0 = 0.2              # initial step in the normalized search space (see PARAM_SCALE)

# ALOHA 2, one arm = 6 hinge joints + gripper (from aloha.xml); left arm then right arm.
ARM_HOME = [0, -0.96, 1.16, 0, -0.3, 0, 0.0084]
ARM_LO   = [-3.14158, -1.85005, -1.76278, -3.14158, -1.8675, -3.14158, 0.002]
ARM_HI   = [ 3.14158,  1.25664,  1.6057,   3.14158,  2.23402,  3.14158, 0.037]
HOME     = np.array(ARM_HOME * 2, dtype=np.float32)
JOINT_LO = np.array(ARM_LO * 2,   dtype=np.float32)
JOINT_HI = np.array(ARM_HI * 2,   dtype=np.float32)
DT = 0.02
HOLD_STEPS = int(3.0 / DT)          # new goal every 3 s, same as ArmJointController.holdSeconds
SCALE = JOINT_HI - JOINT_LO         # per-joint normalization (gripper is in metres, joints in rad)
DIVERGED_PENALTY = 1e4
# Penalize commands that stray from the goal. The world model was trained only on
# "action = goal" data, so aggressive commands (big overshoot gains) look great in
# imagination but oscillate in the real sim. This keeps the policy where the model is valid.
DEVIATION_PENALTY = 0.3

# Start states sampled from the recorded rollout (filled in main), so imagined
# episodes begin inside the distribution the world model was trained on.
START_STATES = None


def build_dynamics():
    model = make_model() ## fill in the model you used in the train.py file
    path = ROOT / "dynamics.pt"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run train.py first.")
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    model.eval()
    return model


def load_norm():
    path = ROOT / "norm.npz"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run train.py first.")
    data = np.load(path)
    return data["x_mean"], data["x_std"], data["y_mean"], data["y_std"]


def unpack(theta):
    w = theta[: ACTION * FEAT].reshape(ACTION, FEAT)
    b = theta[ACTION * FEAT :]
    return w, b


# CMA-ES searches in a normalized space z; the real parameters are theta = z * PARAM_SCALE.
# W[i, j] is scaled by (output range of joint i) / (std of feature j), and b[i] by the output
# range, so one unit of z means a comparable correction for every joint (rad or m) and every
# feature (angle error or velocity). Filled in main() from the recorded data.
PARAM_SCALE = None
OUTPUT_FRACTION = 0.1   # a unit z step moves the action by ~10% of the joint range


def make_param_scale(feature_std):
    # Normalized search space has 3 numbers per joint: [k_pos, k_vel, bias].
    #   W[i, i]          (gain on q_i - goal_i) scaled by range_i / std(q_i - goal_i)
    #   W[i, ACTION + i] (gain on qd_i)         scaled by range_i / std(qd_i)
    #   b[i]                                    scaled by range_i
    out = SCALE * OUTPUT_FRACTION
    return np.stack([out / feature_std[:ACTION], out / feature_std[ACTION:], out], axis=1).reshape(-1)


N_SEARCH = 3 * ACTION   # 42 search parameters instead of the full 406


def expand(z):
    # Structured (per-joint PD-style) policy -> full theta in the controller.json layout.
    # Off-diagonal weights stay 0, so Unity's generic W @ f + b computes exactly the same thing.
    p = (np.asarray(z, dtype=np.float64) * PARAM_SCALE).reshape(ACTION, 3)
    w = np.zeros((ACTION, FEAT))
    idx = np.arange(ACTION)
    w[idx, idx] = p[:, 0]
    w[idx, ACTION + idx] = p[:, 1]
    return np.concatenate([w.reshape(-1), p[:, 2]])


# Thanks for the instruction by claude
def policy(state, goal, w, b):
    # Linear policy on the feature "q_minus_goal_and_qd": f = [q - goal, qd].
    # Output is a correction on top of the goal, so theta = 0 means "action = goal"
    # (exactly the behaviour seen during data collection).
    # Works for a single state (STATE,) or a batch (n, STATE).
    q = state[..., :ACTION]
    qd = state[..., ACTION:]
    f = np.concatenate([q - goal, qd], axis=-1)
    action = goal + f @ w.T + b
    return np.clip(action, JOINT_LO, JOINT_HI).astype(np.float32)


@torch.no_grad()
def step_dynamics(model, state, action, x_mean, x_std, y_mean, y_std):
    # Same convention as train.py:  s' = s + model((concat(s, a) - x_mean) / x_std) * y_std + y_mean
    x = (np.concatenate([state, action], axis=-1) - x_mean) / x_std
    delta_n = model(torch.as_tensor(x, dtype=torch.float32)).numpy()
    return (state + delta_n * y_std + y_mean).astype(np.float32)


def sample_goal(rng):
    span = (JOINT_HI - JOINT_LO) * 0.5 * RANDOM_SCALE
    goal = HOME + rng.uniform(-span, span)
    return np.clip(goal, JOINT_LO, JOINT_HI).astype(np.float32)


def smooth_action(prev, raw):
    blended = (1.0 - ACTION_SMOOTH) * prev + ACTION_SMOOTH * raw
    delta = np.clip(blended - prev, -MAX_DELTA, MAX_DELTA)
    return np.clip(prev + delta, JOINT_LO, JOINT_HI).astype(np.float32)


def sample_goals(rng, n):
    return np.stack([sample_goal(rng) for _ in range(n)])


# Thanks for the instruction by claude
def rollout(model, theta, x_mean, x_std, y_mean, y_std, rng, n=EPISODES):
    # Imagine n episodes at once inside the world model (batched for speed) and
    # return the total return of each episode, shape (n,).
    w, b = unpack(np.asarray(theta, dtype=np.float32))

    state = START_STATES[rng.integers(0, len(START_STATES), size=n)].copy()
    goal = sample_goals(rng, n)
    prev = state[:, :ACTION].copy()   # first "previous action" = current joint angles (as in Unity)
    ret = np.zeros(n, dtype=np.float64)
    alive = np.ones(n, dtype=bool)

    for t in range(HORIZON):
        # Re-sample goals every HOLD_STEPS, like ArmJointController during collection.
        if t > 0 and t % HOLD_STEPS == 0:
            goal = sample_goals(rng, n)

        raw = policy(state, goal, w, b)
        action = smooth_action(prev, raw)          # identical smoothing to WorldModelController
        state = step_dynamics(model, state, action, x_mean, x_std, y_mean, y_std)

        q, qd = state[:, :ACTION], state[:, ACTION:]
        tracking = (((q - goal) / SCALE) ** 2).sum(axis=1)
        velocity = VEL_PENALTY * (qd ** 2).sum(axis=1) * DT
        jerk = ACTION_SMOOTH_PENALTY * (((action - prev) / SCALE) ** 2).sum(axis=1)
        deviation = DEVIATION_PENALTY * (((raw - goal) / SCALE) ** 2).sum(axis=1)
        ret[alive] -= (tracking + velocity + jerk + deviation)[alive]
        prev = action

        # If the imagined state blows up, punish hard and freeze that episode.
        # This stops CMA-ES from exploiting regions where the world model is wrong.
        bad = alive & (~np.isfinite(state).all(axis=1) | (np.abs(q) > 2 * np.pi).any(axis=1))
        if bad.any():
            ret[bad] -= DIVERGED_PENALTY
            alive &= ~bad
            state[bad] = START_STATES[0]
            prev[bad] = START_STATES[0, :ACTION]
        if not alive.any():
            break
    return ret


def fitness(theta, model, x_mean, x_std, y_mean, y_std, seed):
    # rollout is batched over EPISODES, so a single call covers all episodes.
    rng = np.random.default_rng(seed)
    returns = rollout(model, theta, x_mean, x_std, y_mean, y_std, rng)
    return -float(np.mean(returns))   # CMA-ES minimizes


def save_controller(theta):
    w, b = unpack(theta)
    payload = {
        "feature": "q_minus_goal_and_qd",
        "state_size": STATE,
        "action_size": ACTION,
        "W": w.astype(float).reshape(-1).tolist(),
        "b": b.astype(float).tolist(),
        "joint_lo": JOINT_LO.astype(float).tolist(),
        "joint_hi": JOINT_HI.astype(float).tolist(),
        "home": HOME.astype(float).tolist(),
        "vel_penalty": VEL_PENALTY,
        "horizon": HORIZON,
    }
    path = ROOT / "controller.json"
    path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {path}")


# Thanks for the instruction by claude
def main():
    global START_STATES, PARAM_SCALE

    model = build_dynamics()
    x_mean, x_std, y_mean, y_std = load_norm()
    data = np.loadtxt(ROOT / "rollout.csv", delimiter=",", dtype=np.float32)
    START_STATES = data[::50, :STATE].copy()
    norm = (x_mean, x_std, y_mean, y_std)

    # Feature scale [q - goal, qd] from the data (actions recorded = goals during collection).
    q, qd, goals = data[:, :ACTION], data[:, ACTION:STATE], data[:, STATE:STATE + ACTION]
    feature_std = np.concatenate([(q - goals).std(0), qd.std(0)]) + 1e-3
    PARAM_SCALE = make_param_scale(feature_std)

    def evaluate(z, seed):
        return fitness(expand(z), model, *norm, seed=seed)

    # Fixed evaluation seeds: used to pick the best parameters fairly across iterations.
    eval_seeds = [100, 101, 102, 103]

    def eval_score(z):
        return float(np.mean([evaluate(z, s) for s in eval_seeds]))

    # Baseline: theta = 0 means "send the goal straight to the actuators".
    zero = np.zeros(N_SEARCH)
    best_z, best_eval = zero.copy(), eval_score(zero)
    print(f"baseline (action = goal) eval return: {-best_eval:.3f}")

    es = cma.CMAEvolutionStrategy(zero, SIGMA0, {"seed": 0, "verbose": -9})
    for it in range(CMA_ITERS):
        zs = es.ask()
        # Same seed for every candidate in an iteration (same starts and goals) -> fair
        # comparison; a new seed each iteration -> no overfitting to one set of goals.
        fits = [evaluate(z, seed=it) for z in zs]
        es.tell(zs, fits)

        # Track the distribution mean on the fixed eval seeds (much less noisy than "best").
        mean_eval = eval_score(es.mean)
        if mean_eval < best_eval:
            best_eval, best_z = mean_eval, np.asarray(es.mean).copy()
        print(f"iter {it + 1:3d}/{CMA_ITERS}  pop best {-min(fits):9.3f}  pop mean {-np.mean(fits):9.3f}  "
              f"eval(mean) {-mean_eval:9.3f}  best eval {-best_eval:9.3f}  sigma {es.sigma:.4f}")

    best_theta = expand(best_z)
    zero = np.zeros(N_PARAMS)   # full-size zero theta for the baseline comparison below

    # Re-evaluate on unseen seeds: the number to report as the imagined return.
    held_out = [-fitness(best_theta, model, *norm, seed=1000 + k) for k in range(5)]
    held_base = [-fitness(zero, model, *norm, seed=1000 + k) for k in range(5)]
    print(f"imagined return (held-out seeds): controller {np.mean(held_out):.3f}  "
          f"vs baseline {np.mean(held_base):.3f}")
    save_controller(best_theta)


if __name__ == "__main__":
    current_time = time.time()
    main()
    end_time = time.time()
    print(f"Training time: {end_time - current_time} seconds")

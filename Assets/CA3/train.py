import numpy as np
import torch
import time
from pathlib import Path
from torch import nn

ROOT = Path(__file__).resolve().parent / "Data"
STATE = 28
ACTION = 14


def mean_std(z):
    return z.mean(0), z.std(0) + 1e-6


def norm(z, mean, std):
    return (z - mean) / std


def save_norm_json(x_mean, x_std, y_mean, y_std):
    arrays = {"x_mean": x_mean, "x_std": x_std, "y_mean": y_mean, "y_std": y_std}
    lines = ["{"]
    keys = list(arrays)
    for index, key in enumerate(keys):
        values = ", ".join(f"{float(value):.8f}" for value in arrays[key])
        comma = "," if index < len(keys) - 1 else ""
        lines.append(f'  "{key}": [{values}]{comma}')
    lines.append("}")
    (ROOT / "norm.json").write_text("\n".join(lines) + "\n")


def load_rollout():
    path = ROOT / "rollout.csv"
    if not path.exists() or path.stat().st_size == 0:
        raise FileNotFoundError(f"No rollout at {path}. Enable RolloutRecorder and Play.")
    data = np.loadtxt(path, delimiter=",", dtype=np.float32)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] != STATE + ACTION + STATE:
        raise ValueError(f"Expected {STATE + ACTION + STATE} columns, found {data.shape[1]}.")
    return data


def predict_delta(model, state, action, x_mean, x_std, y_mean, y_std):
    x = np.concatenate([state, action], axis=-1)
    x_n = (x - x_mean) / x_std
    with torch.no_grad():
        delta_n = model(torch.as_tensor(x_n)).numpy()
    return delta_n * y_std + y_mean


def joint_rmse(pred, true):
    err = pred[..., 0:ACTION] - true[..., 0:ACTION]
    return float(np.sqrt((err ** 2).sum(axis=-1).mean()))


def open_loop_rmse(model, data, start, x_mean, x_std, y_mean, y_std, horizon):
    end = len(data) - horizon
    if end <= start:
        return float("nan")
    idx = np.arange(start, end)
    if len(idx) > 2000:
        idx = idx[:: max(1, len(idx) // 2000)]
    state = data[idx, 0:STATE].copy()
    for t in range(horizon):
        action = data[idx + t, STATE:STATE + ACTION]
        state = state + predict_delta(model, state, action, x_mean, x_std, y_mean, y_std)
    return joint_rmse(state, data[idx + horizon, 0:STATE])

# Training hyperparameters
HIDDEN = 256          # width of each hidden layer
EPOCHS = 30           # passes over the training split
BATCH = 512           # one-step batch size
LR = 1e-3             # Adam learning rate
HORIZON = 10          # rollout length used by multi_step_loss
MULTI_BATCH = 64      # number of rollouts per multi-step batch
LAMBDA = 0.5          # weight of the multi-step term in the total loss
VAL_FRACTION = 0.1    # last 10% of the (time-ordered) data is held out


def make_model():
    # MLP: normalized [state, action] -> normalized state delta.
    # train_controller.build_dynamics must build exactly this architecture.
    return nn.Sequential(
        nn.Linear(STATE + ACTION, HIDDEN), nn.SiLU(),
        nn.Linear(HIDDEN, HIDDEN), nn.SiLU(),
        nn.Linear(HIDDEN, STATE),
    )


## Thanks for claude(I'm not familiaer with Torch)
def multi_step_loss(model, data, split, x_mean, x_std, y_mean, y_std, horizon, batch, loss_fn):
    # Roll the model forward on its own predictions for `horizon` steps and penalize
    # the error at every step. This teaches the model to keep compounding error small,
    # which is exactly how the ghost arm and the CMA-ES rollouts use it.
    if split - horizon <= 0:
        return torch.zeros(())

    # Normalization stats as tensors so gradients flow through the whole rollout.
    xm, xs = torch.as_tensor(x_mean), torch.as_tensor(x_std)
    ym, ys = torch.as_tensor(y_mean), torch.as_tensor(y_std)

    # Random start indices inside the training split so t + horizon never leaves it.
    starts = np.random.randint(0, split - horizon, size=batch)

    # Start from the true state, then feed back the model's own prediction (no detach).
    state = torch.as_tensor(data[starts, 0:STATE])
    total = torch.zeros(())
    for k in range(horizon):
        action = torch.as_tensor(data[starts + k, STATE:STATE + ACTION])  # recorded action
        x = (torch.cat([state, action], dim=1) - xm) / xs
        state = state + model(x) * ys + ym                                # imagined next state
        true_next = torch.as_tensor(data[starts + k, STATE + ACTION:])    # s' of row t+k
        # Scale by y_std so angles (rad), velocities (rad/s) and gripper (m) weigh equally.
        total = total + loss_fn((state - true_next) / ys, torch.zeros_like(state))
    return total / horizon


## Thanks for claude(I'm not familiaer with Torch)
def train_world_model(data):
    torch.manual_seed(0)
    np.random.seed(0)

    # Split each row into state, action and next state; the target is the state delta.
    s = data[:, 0:STATE]
    a = data[:, STATE:STATE + ACTION]
    s_next = data[:, STATE + ACTION:]
    x = np.concatenate([s, a], axis=1)
    y = s_next - s

    # Time-ordered split (no shuffling) so validation is an unseen stretch of time.
    split = int((1.0 - VAL_FRACTION) * len(data))
    if split < 2 or len(data) - split < 2:
        raise ValueError(f"Not enough rows ({len(data)}) to train. Record a longer rollout.")

    # Normalization statistics come from the training split only.
    x_mean, x_std = mean_std(x[:split])
    y_mean, y_std = mean_std(y[:split])
    x_mean, x_std, y_mean, y_std = (v.astype(np.float32) for v in (x_mean, x_std, y_mean, y_std))

    X = torch.as_tensor(norm(x, x_mean, x_std))
    Y = torch.as_tensor(norm(y, y_mean, y_std))

    model = make_model()
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    loss_fn = nn.MSELoss()

    print(f"rows={len(data)}  train={split}  val={len(data) - split}")
    for epoch in range(EPOCHS):
        model.train()
        perm = torch.randperm(split)  # shuffling inside the training split is fine
        running, batches = 0.0, 0
        for i in range(0, split, BATCH):
            idx = perm[i:i + BATCH]
            one_step = loss_fn(model(X[idx]), Y[idx])
            multi = multi_step_loss(model, data, split, x_mean, x_std, y_mean, y_std,
                                    HORIZON, MULTI_BATCH, loss_fn)
            loss = one_step + LAMBDA * multi
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += loss.item()
            batches += 1
        sched.step()

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(X[split:]), Y[split:]).item()
        print(f"epoch {epoch + 1:3d}/{EPOCHS}  train {running / batches:.5f}  val {val_loss:.5f}")

    # Evaluation on the held-out split (numbers for REPORT.md).
    model.eval()
    val = data[split:]
    pred = val[:, 0:STATE] + predict_delta(
        model, val[:, 0:STATE], val[:, STATE:STATE + ACTION], x_mean, x_std, y_mean, y_std)
    print(f"one-step joint RMSE: {joint_rmse(pred, val[:, STATE + ACTION:]):.6f}")
    for h in (10, 50, 100):
        rmse = open_loop_rmse(model, data, split, x_mean, x_std, y_mean, y_std, h)
        print(f"open-loop {h:3d}-step joint RMSE: {rmse:.6f}")

    # Save everything the rest of the pipeline needs.
    ROOT.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), ROOT / "dynamics.pt")            # train_controller.py
    np.savez(ROOT / "norm.npz", x_mean=x_mean, x_std=x_std,
             y_mean=y_mean, y_std=y_std)                            # train_controller.py
    save_norm_json(x_mean, x_std, y_mean, y_std)                     # WorldModelDriver (Unity)
    export_onnx(model)                                               # WorldModelDriver (Unity)
    print(f"saved dynamics.pt, norm.npz, norm.json, dynamics.onnx to {ROOT}")
    return model


def export_onnx(model):
    # Unity Inference Engine expects a [1, STATE + ACTION] input.
    dummy = torch.zeros(1, STATE + ACTION)
    path = ROOT / "dynamics.onnx"
    kwargs = dict(input_names=["input"], output_names=["output"], opset_version=15)
    try:
        # Prefer the classic TorchScript exporter: simple graph, broad Unity support.
        torch.onnx.export(model, dummy, path, dynamo=False, **kwargs)
    except TypeError:
        # Older PyTorch versions have no `dynamo` argument.
        torch.onnx.export(model, dummy, path, **kwargs)


if __name__ == "__main__":
    current_time = time.time()
    train_world_model(load_rollout())
    end_time = time.time()
    print(f"Training time: {end_time - current_time} seconds")

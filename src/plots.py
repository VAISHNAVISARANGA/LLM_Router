"""Figures sized for a two-column paper (about 3.4 inches wide)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

W, H = 3.4, 2.5
LABELS = {("q", "handcrafted"): "Q-learn + handcrafted", ("q", "embedding"): "Q-learn + embedding",
          ("q", "hybrid"): "Q-learn + hybrid",
          ("dqn", "handcrafted"): "DQN + handcrafted", ("dqn", "embedding"): "DQN + embedding",
          ("dqn", "hybrid"): "DQN + hybrid"}
STYLE = {("q", "handcrafted"): ("tab:blue", "-"), ("q", "embedding"): ("tab:cyan", "-"),
         ("q", "hybrid"): ("tab:green", "-"),
         ("dqn", "handcrafted"): ("tab:red", "--"), ("dqn", "embedding"): ("tab:orange", "--"),
         ("dqn", "hybrid"): ("tab:purple", "--")}
plt.rcParams.update({"font.size": 7, "axes.titlesize": 8, "legend.fontsize": 6})


def _save(fig, out_dir, name):
    fig.tight_layout()
    fig.savefig(f"{out_dir}/{name}.png", dpi=200)
    fig.savefig(f"{out_dir}/{name}.pdf")
    plt.close(fig)


def _smooth(x, w):
    w = min(w, len(x))
    return np.convolve(x, np.ones(w) / w, mode="valid")


def training_curves(histories, out_dir, window=20):
    """histories[(kind, feat)] = array (seeds, episodes) of mean reward per step."""
    fig, ax = plt.subplots(figsize=(W, H))
    for key, h in histories.items():
        sm = np.stack([_smooth(r, window) for r in h])
        m, s = sm.mean(0), sm.std(0)
        x = np.arange(len(m)) + window
        col, ls = STYLE[key]
        ax.plot(x, m, color=col, ls=ls, label=LABELS[key])
        ax.fill_between(x, m - s, m + s, color=col, alpha=0.15)
    ax.set_xlabel("training episode"); ax.set_ylabel("mean reward / query")
    ax.set_title("Learning curves (mean $\\pm$ std over seeds)"); ax.legend()
    _save(fig, out_dir, "fig_training_curves")


def pareto(sweep, baselines_pts, out_dir):
    """sweep[(kind, feat)] = list of (cost_per_1k, quality) for each lambda."""
    fig, ax = plt.subplots(figsize=(W, H))
    for key, pts in sweep.items():
        pts = sorted(pts)
        col, ls = STYLE[key]
        ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", ms=3,
                color=col, ls=ls, label=LABELS[key])
    marks = {"always_small": ("s", "gray"), "always_large": ("*", "black"),
             "random": ("x", "green"), "length_heuristic": ("^", "purple")}
    for name, (c, q) in baselines_pts.items():
        if name in marks:
            ax.scatter([c], [q], marker=marks[name][0], color=marks[name][1], s=30,
                       zorder=5, label=name.replace("_", " "))
    ax.set_xscale("symlog", linthresh=max(1e-3, min(p[0] for v in sweep.values() for p in v) + 1e-6))
    ax.set_xlabel("cost per 1k queries (USD)"); ax.set_ylabel("mean quality")
    ax.set_title("Cost-quality trade-off (each dot = one $\\lambda$)")
    ax.legend(ncol=2, loc="lower right")
    _save(fig, out_dir, "fig_pareto")


def reward_bars(rows, out_dir):
    """rows = list of (name, mean, std)."""
    fig, ax = plt.subplots(figsize=(W, H))
    names = [r[0] for r in rows]
    ax.barh(names, [r[1] for r in rows], xerr=[r[2] for r in rows], color="tab:blue", alpha=0.8)
    ax.set_xlabel("test reward / query"); ax.set_title("Test reward by method")
    ax.invert_yaxis()
    _save(fig, out_dir, "fig_reward_bars")


def per_task(tasks, agent_frac, oracle_frac, out_dir, title):
    fig, ax = plt.subplots(figsize=(W, max(H, 0.22 * len(tasks) + 0.8)))
    y = np.arange(len(tasks))
    ax.barh(y - 0.2, agent_frac, 0.4, label="learned router", color="tab:blue")
    ax.barh(y + 0.2, oracle_frac, 0.4, label="oracle", color="lightgray")
    ax.set_yticks(y); ax.set_yticklabels(tasks); ax.invert_yaxis()
    ax.set_xlabel("fraction routed to large model"); ax.set_title(title); ax.legend()
    _save(fig, out_dir, "fig_per_task")

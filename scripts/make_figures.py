"""Figures for the README, from results/ only. Writes results/figures/<name>_{light,dark}.png.

python scripts/make_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, LogLocator, MultipleLocator, NullFormatter

ROOT = Path(__file__).resolve().parent.parent
SIM = ROOT / "results" / "sim"
H200 = ROOT / "results" / "h200"
FIG = ROOT / "results" / "figures"

# Categorical slots in fixed order; a policy keeps its colour in every chart.
SLOTS = {
    "light": [
        "#2a78d6",
        "#eb6834",
        "#1baf7a",
        "#eda100",
        "#e87ba4",
        "#008300",
        "#6250d6",
        "#e34948",
    ],
    "dark": [
        "#3987e5",
        "#d95926",
        "#199e70",
        "#c98500",
        "#d55181",
        "#0ca30c",
        "#9085e9",
        "#e66767",
    ],
}
ORDER = [
    "ttft",
    "affinity",
    "chwbl",
    "least-request",
    "round-robin",
    "legacy",
    "threshold",
    "random",
]
LABEL = {
    "ttft": "TTFT-cost (ours)",
    "affinity": "SGLang affinity",
    "threshold": "SGLang threshold (old)",
    "chwbl": "Consistent hash (bounded)",
    "least-request": "Least request",
    "round-robin": "Round robin",
    "legacy": "Original repo rule",
    "random": "Random",
    "p2c": "Power of two",
}
THEME = {
    "light": {
        "surface": "#fcfcfb",
        "ink": "#0b0b0b",
        "ink2": "#52514e",
        "muted": "#898781",
        "grid": "#e1e0d9",
        "axis": "#c3c2b7",
    },
    "dark": {
        "surface": "#1a1a19",
        "ink": "#f0efec",
        "ink2": "#c3c2b7",
        "muted": "#898781",
        "grid": "#2c2c2a",
        "axis": "#383835",
    },
}
TRACE_TITLE = {
    "conversation": "Mooncake conversation trace",
    "toolagent": "Mooncake tool-agent trace",
}


def color(policy: str, mode: str) -> str:
    return SLOTS[mode][ORDER.index(policy)] if policy in ORDER else THEME[mode]["muted"]


def load(name: str) -> dict[str, Any] | None:
    p = SIM / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else None


def h200_rows() -> list[dict[str, Any]]:
    """Measured rows from both H200 runs, named as in build_readme.py."""
    import sys

    sys.path.insert(0, str(ROOT / "scripts"))
    from build_readme import h200

    return h200()


def style(ax: Any, mode: str) -> None:
    t = THEME[mode]
    ax.set_facecolor(t["surface"])
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(t["axis"])
    ax.tick_params(colors=t["ink2"], labelsize=9, length=3)
    ax.grid(True, color=t["grid"], linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.xaxis.label.set_color(t["ink2"])
    ax.yaxis.label.set_color(t["ink2"])
    ax.title.set_color(t["ink"])


def figure(mode: str, ncols: int, w: float = 10.5, h: float = 4.2) -> tuple[Any, Any]:
    fig, axes = plt.subplots(1, ncols, figsize=(w, h), squeeze=False)
    fig.patch.set_facecolor(THEME[mode]["surface"])
    for ax in axes[0]:
        style(ax, mode)
    return fig, axes[0]


def save(fig: Any, name: str, mode: str) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        FIG / f"{name}_{mode}.png", dpi=160, bbox_inches="tight", facecolor=fig.get_facecolor()
    )
    plt.close(fig)


def seconds(v: float, _pos: Any = None) -> str:
    return f"{v:g} s" if v >= 1 else f"{v * 1000:g} ms"


def legend(fig: Any, policies: list[str], mode: str, extra: list[Any] | None = None) -> None:
    t = THEME[mode]
    handles = [plt.Line2D([], [], color=color(p, mode), lw=2, marker="o", ms=5) for p in policies]
    labels = [LABEL[p] for p in policies]
    if extra:
        for h, lab in extra:
            handles.append(h)
            labels.append(lab)
    leg = fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=min(5, len(labels)),
        frameon=False,
        fontsize=9,
        bbox_to_anchor=(0.5, -0.17),
    )
    for txt in leg.get_texts():
        txt.set_color(t["ink2"])


# -- figures ---------------------------------------------------------------------------------

LOAD_POLICIES = ["ttft", "affinity", "threshold", "chwbl", "legacy", "least-request", "round-robin"]


def fig_load(mode: str) -> None:
    d = load("load")
    if not d:
        return
    rows = d["rows"]
    real = h200_rows()
    fig, axes = figure(mode, 2)
    t = THEME[mode]
    for ax, trace in zip(axes, ("conversation", "toolagent")):
        for p in LOAD_POLICIES:
            pts = sorted(
                (r["workload"]["scale"], r["summary"]["ttft_p99"])
                for r in rows
                if r["workload"]["kind"] == trace and r["policy"] == p
            )
            if not pts:
                continue
            xs, ys = zip(*pts)
            lw, z = (2.4, 5) if p == "ttft" else (1.6, 3)
            ax.plot(xs, ys, color=color(p, mode), lw=lw, marker="o", ms=4, zorder=z)
            meas = sorted(
                (r["time_scale"], r["ttft_p99"])
                for r in real
                if r["trace_name"] == trace and r["policy"] == p and "ttft_p99" in r
            )
            if meas:
                mx, my = zip(*meas)
                ax.scatter(
                    mx,
                    my,
                    s=70,
                    marker="D",
                    facecolor=t["surface"],
                    edgecolor=color(p, mode),
                    linewidth=2,
                    zorder=7 if p == "ttft" else 6,
                )
        ax.axhline(2.0, color=t["muted"], lw=1, ls=(0, (4, 3)), zorder=2)
        ax.text(
            0.01,
            2.15,
            "2 s SLO",
            transform=ax.get_yaxis_transform(),
            color=t["muted"],
            fontsize=8,
            va="bottom",
        )
        ax.set_yscale("log")
        ax.set_ylim(0.5, 300)
        ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
        ax.yaxis.set_major_formatter(FuncFormatter(seconds))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.xaxis.set_major_locator(MultipleLocator(1))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{v:g}x"))
        ax.set_xlabel("trace replay speed (1x = recorded arrival rate), 4 x H200")
        ax.set_title(TRACE_TITLE[trace], fontsize=11, loc="left", color=t["ink"])
    axes[0].set_ylabel("TTFT p99")
    extra = []
    if real:
        extra = [
            (
                plt.Line2D(
                    [],
                    [],
                    ls="",
                    marker="D",
                    ms=7,
                    markerfacecolor=t["surface"],
                    markeredgecolor=t["ink2"],
                    markeredgewidth=1.5,
                ),
                "measured on H200s",
            )
        ]
    legend(fig, LOAD_POLICIES, mode, extra)
    save(fig, "ttft_vs_load", mode)


def fig_capacity(mode: str) -> None:
    d = load("capacity")
    if not d:
        return
    rows = d["rows"]
    fig, axes = figure(mode, 2, h=3.6)
    t = THEME[mode]
    pols = ["ttft", "affinity", "threshold", "chwbl", "legacy", "least-request", "round-robin"]
    for ax, trace in zip(axes, ("conversation", "toolagent")):
        cap = {r["policy"]: r["capacity_scale"] for r in rows if r["trace"] == trace}
        ys = list(range(len(pols)))[::-1]
        vals = [cap.get(p, 0) for p in pols]
        ax.barh(ys, vals, height=0.62, color=[color(p, mode) for p in pols], zorder=3)
        base = cap.get("round-robin", 0)
        for y, v, p in zip(ys, vals, pols):
            rel = f"  ({(v / base - 1) * 100:+.0f}%)" if base and p != "round-robin" else ""
            ax.text(v, y, f"  {v:.2f}x{rel}", va="center", fontsize=8.5, color=t["ink2"])
        ax.set_yticks(ys, [LABEL[p] for p in pols], fontsize=9, color=t["ink2"])
        ax.set_xlim(0, max(vals) * 1.45)
        ax.grid(axis="y", visible=False)
        ax.set_xlabel("max load with TTFT p99 <= 2 s (x trace rate; % vs round robin)")
        ax.set_title(TRACE_TITLE[trace], fontsize=11, loc="left", color=t["ink"])
    fig.subplots_adjust(wspace=0.55)
    save(fig, "capacity", mode)


def fig_beta(mode: str) -> None:
    d = load("beta")
    if not d:
        return
    rows = d["rows"]
    fig, axes = figure(mode, 2, h=3.8)
    t = THEME[mode]
    ramp = ["#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95"]
    if mode == "dark":
        ramp = ["#184f95", "#256abf", "#3987e5", "#6da7ec", "#9ec5f4"]
    for ax, trace in zip(axes, ("conversation", "toolagent")):
        scales = sorted({r["workload"]["scale"] for r in rows if r["workload"]["kind"] == trace})
        for k, s in enumerate(scales):
            ref = [
                r
                for r in rows
                if r["workload"]["kind"] == trace
                and r["workload"]["scale"] == s
                and r["policy"] == "affinity"
            ]
            if not ref:
                continue
            base = ref[0]["summary"]["ttft_p99"]
            pts = sorted(
                (r["params"]["beta"], r["summary"]["ttft_p99"] / base)
                for r in rows
                if r["workload"]["kind"] == trace
                and r["workload"]["scale"] == s
                and r["policy"] == "ttft"
            )
            xs, ys = zip(*pts)
            xs = [max(x, 0.5) for x in xs]  # beta=0 drawn at 0.5 on the log axis
            ax.plot(xs, ys, color=ramp[k % len(ramp)], lw=2, marker="o", ms=4, label=f"{s:g}x load")
        ax.axhline(1.0, color=t["muted"], lw=1, ls=(0, (4, 3)))
        ax.axvline(8, color=t["muted"], lw=1, ls=":")
        ax.set_yscale("log")
        ax.set_ylim(0.75, 28)
        ticks = [0.8, 1, 1.25, 1.5, 2, 4, 8, 16]
        ax.set_yticks(ticks, [f"{v:g}" for v in ticks])
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_xscale("log", base=2)
        ax.set_xticks([0.5, 1, 2, 4, 8, 16, 32, 64], ["0", "1", "2", "4", "8", "16", "32", "64"])
        ax.set_xlabel("beta (weight on recomputation)")
        ax.set_title(TRACE_TITLE[trace], fontsize=11, loc="left", color=t["ink"])
        leg = ax.legend(frameon=False, fontsize=8, loc="upper right")
        for txt in leg.get_texts():
            txt.set_color(t["ink2"])
    axes[0].set_ylabel("TTFT p99 relative to SGLang affinity")
    save(fig, "beta", mode)


def fig_fidelity(mode: str) -> None:
    d = load("fidelity")
    real = h200_rows()
    if not d or not real:
        return
    from build_readme import sim_policy  # h200_rows put scripts/ on the path

    sim = {
        (r["workload"]["kind"], r["workload"]["scale"], r["policy"], r.get("variant", "")): r[
            "summary"
        ]
        for r in d["rows"]
    }
    fig, axes = figure(mode, 3, w=11.5, h=3.7)
    t = THEME[mode]
    metrics = [
        ("ttft_p50", "TTFT p50", True),
        ("ttft_p99", "TTFT p99", True),
        ("hit_ratio", "prompt tokens reused", False),
    ]
    for ax, (m, title, logscale) in zip(axes, metrics):
        xs, ys, cs = [], [], []
        for r in real:
            s = sim.get((r["trace_name"], r["time_scale"], *sim_policy(r["policy"])))
            if s is None or m not in r or m not in s:
                continue
            xs.append(s[m])
            ys.append(r[m])
            cs.append(color(sim_policy(r["policy"])[0], mode))
        if not xs:
            continue
        lo, hi = min(xs + ys) * 0.8, max(xs + ys) * 1.25
        ax.plot([lo, hi], [lo, hi], color=t["muted"], lw=1, ls=(0, (4, 3)))
        ax.scatter(xs, ys, c=cs, s=46, edgecolor=t["surface"], linewidth=1.5, zorder=4)
        if logscale:
            ax.set_xscale("log")
            ax.set_yscale("log")
            nice = (0.1, 0.2, 0.3, 0.5, 1, 2, 3, 5, 10, 20, 50, 100)
            ticks = [v for v in nice if lo <= v <= hi]
            for axis in (ax.xaxis, ax.yaxis):
                axis.set_major_formatter(FuncFormatter(seconds))
                axis.set_minor_formatter(NullFormatter())
            ax.set_xticks(ticks)
            ax.set_yticks(ticks)
        else:
            pct = FuncFormatter(lambda v, _p: f"{v * 100:.0f}%")
            ax.xaxis.set_major_formatter(pct)
            ax.yaxis.set_major_formatter(pct)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel("simulated")
        ax.set_title(title, fontsize=11, loc="left", color=t["ink"])
    axes[0].set_ylabel("measured on 4 x H200")
    shown = {sim_policy(r["policy"])[0] for r in real}
    legend(fig, [p for p in LOAD_POLICIES if p in shown], mode)
    save(fig, "fidelity", mode)


def main() -> None:
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 10})
    for mode in ("light", "dark"):
        for f in (fig_load, fig_capacity, fig_beta, fig_fidelity):
            f(mode)
    print("figures:", sorted(p.name for p in FIG.glob("*.png")) if FIG.exists() else [])


if __name__ == "__main__":
    main()

"""Render actual recorded state arrays; no simulated or decorative activity."""
from __future__ import annotations
import html
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def render(results: Path) -> None:
    summary = json.loads((results / "summary.json").read_text())
    fly = np.load(results / "p0_fly_fixed_states.npz", allow_pickle=False)["states"]
    fig, axs = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    fig.suptitle("LiquidAion | recorded FlyWire reservoir states", fontsize=16)
    im = axs[0, 0].imshow(fly.T, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest")
    axs[0, 0].set(title="512 simulated neurons, original index order", xlabel="Forward step", ylabel="Neuron index")
    fig.colorbar(im, ax=axs[0, 0], label="Rate state")
    for mode in ["none", "static", "random", "fly"]:
        data = np.load(results / f"p0_{mode}_fixed_states.npz", allow_pickle=False)["states"]
        axs[0, 1].plot(np.sqrt(np.mean(data**2, axis=1)), label=mode)
    axs[0, 1].set(title="Reservoir activity on the same token stream", xlabel="Forward step", ylabel="State RMS")
    axs[0, 1].legend()
    conditions = [r for r in summary["conditions"] if r["prompt_index"] == 0]
    axs[1, 0].bar([r["mode"] for r in conditions], [r["logit_rms_difference"] for r in conditions], color=["#8fa4b2", "#a09ed5", "#eeaa66", "#4eafa0"])
    axs[1, 0].set(title="Output logit change on matched contexts", ylabel="RMS difference")
    retained = np.load(results / "persistence_b_retained_states.npz")["states"]
    reset = np.load(results / "persistence_b_reset_states.npz")["states"]
    axs[1, 1].plot(np.linalg.norm(retained - reset, axis=1))
    axs[1, 1].set(title="Prompt B: retained vs reset recurrent state", xlabel="Forward step", ylabel="State L2 distance")
    fig.savefig(results / "overview.png", dpi=150)
    plt.close(fig)
    # Joint PCA is only a projection of recorded rates, not emotional localization.
    states = [np.load(results / f"p{i}_fly_fixed_states.npz")["states"] for i in range(len(summary["generated_examples"]))]
    combined = np.concatenate(states)
    center = combined.mean(axis=0)
    _, _, vt = np.linalg.svd(combined - center, full_matrices=False)
    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    for i, data in enumerate(states):
        xy = (data - center) @ vt[:2].T
        ax.plot(xy[:, 0], xy[:, 1], marker=".", label=f"Prompt {i}")
        ax.scatter(xy[0, 0], xy[0, 1], marker="x", s=70)
    ax.set(title="Joint PCA of recorded reservoir trajectories", xlabel="PC1", ylabel="PC2")
    ax.legend()
    fig.savefig(results / "trajectory.png", dpi=150)
    plt.close(fig)
    cases = {}
    for mode in ["none", "static", "random", "fly"]:
        trace = json.loads((results / f"p0_{mode}_fixed.json").read_text())["trace"]
        rates = np.load(results / f"p0_{mode}_fixed_states.npz")["states"]
        cases[mode] = {"trace": trace, "states": rates.tolist()}
    cases["feedback_off"] = {"trace": json.loads((results / "p0_fly_feedback_off.json").read_text())["trace"],
                             "states": np.load(results / "p0_fly_feedback_off_states.npz")["states"].tolist()}
    examples = "".join(f'<article><h3>{html.escape(e["prompt"])}</h3><div class="pair"><pre>BASELINE\n{html.escape(e["baseline"])}</pre><pre>FLY\n{html.escape(e["fly"])}</pre></div></article>' for e in summary["generated_examples"])
    payload = json.dumps(cases).replace("<", "\\u003c")
    document = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>LiquidAion recorded experiment</title>
<style>body{background:#101922;color:#e2eaf1;font:16px system-ui;margin:0 auto;padding:32px;max-width:1120px}h1{font-size:36px}h2{margin-top:36px}p{color:#b0c3d3;line-height:1.6}article{border:1px solid #2b4050;border-radius:12px;padding:16px;margin:16px 0}.pair{display:grid;grid-template-columns:1fr 1fr;gap:16px}pre{white-space:pre-wrap;background:#182633;padding:18px;border-radius:8px;font:15px system-ui}img{width:100%;border-radius:12px;background:white}select,input{margin:12px;padding:8px}canvas{width:100%;height:180px;background:#172737;border-radius:8px}small{color:#adc3d2}@media(max-width:700px){.pair{grid-template-columns:1fr}}</style>
<h1>LiquidAion / FlyWire + Qwen3-0.6B</h1><p>Recorded CPU experiment: 512 neurons, 16,731 directed biological edges. Frozen LLM, fixed untrained adapters. This viewer replays saved measurements; it does not run inference or apply live steering.</p>
<h2>Recorded recurrent activity</h2><label>Condition <select id="mode"><option value="fly">Fly topology / steering ON</option><option value="none">Steering OFF</option><option value="static">Static steering</option><option value="random">Random topology</option><option value="feedback_off">Post-intervention feedback OFF</option></select></label><label>Forward step <input id="step" type="range" min="0" value="0"></label><div id="stats"></div><canvas id="rates" width="1024" height="180"></canvas><p>Bars are the actual 512 simulated rates. Index ordering is not anatomical space. OFF still observes LLM activations but applies zero residual.</p>
<h2>Matched-context metrics and persistence</h2><img src="overview.png" alt="Measured recurrent rates, paired output logit differences and persistence"><h2>Trajectory projection</h2><img src="trajectory.png" alt="Joint PCA of recorded reservoir states"><p>Three prompts and one seed illustrate trajectories. They do not establish category separation, semantic control, topology superiority, or emotion.</p>
<h2>Greedy output examples</h2>''' + examples + '''<p>Outputs were limited to 32 tokens and may be truncated. Full token IDs, configuration, layer traces and source hashes accompany this report.</p><script>
const cases=PAYLOAD;const mode=document.getElementById('mode'),step=document.getElementById('step'),stats=document.getElementById('stats');
function draw(){const c=cases[mode.value];step.max=c.states.length-1;const n=Math.min(+step.value,c.states.length-1);step.value=n;const s=c.states[n],t=c.trace[n];stats.textContent=`Step ${n} | token position ${t.token_position} | state RMS ${t.state_rms.toFixed(5)} | hidden delta RMS ${(100*t.delta_relative_rms).toFixed(3)}% | simulation time ${t.simulation_time.toFixed(2)}`;const ctx=document.getElementById('rates').getContext('2d');ctx.clearRect(0,0,1024,180);ctx.fillStyle='#526a7c';ctx.fillRect(0,90,1024,1);s.forEach((v,i)=>{ctx.fillStyle=v>=0?'#4ed4b4':'#da8499';ctx.fillRect(i*2,90-Math.max(v,0)*85,1.7,Math.abs(v)*85)});}
mode.addEventListener('change',draw);step.addEventListener('input',draw);draw();</script></html>'''
    (results / "dashboard.html").write_text(document.replace("PAYLOAD", payload))

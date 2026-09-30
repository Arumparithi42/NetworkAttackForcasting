"""SIH26153 - Network Attack Forecasting dashboard (Streamlit, fully offline).

    streamlit run app.py
"""
from __future__ import annotations

import json
import shutil
import sqlite3
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard.data import ROOT, discover, host_flows, load_day
from netwm.attack_mapping.mapper import STAGE_LABEL, TACTICS, tactic_refs
from netwm.features.feature_spec import DESCRIPTIONS
from netwm.forecasting.rollout import feature_intervention
from netwm.inference.pipeline import Forecaster

st.set_page_config(page_title="Network Attack Forecasting", page_icon="🛡️", layout="wide")

BLUE, ORANGE, INK2, GRID = "#2a78d6", "#eb6834", "#52514e", "#e4e3df"
STAGE_COLOURS = {"BENIGN": "#c9c8c2", "RECON_DISCOVERY": "#eda100", "INITIAL_ACCESS_ATTEMPT": "#e87ba4",
                 "COMMAND_AND_CONTROL": "#008300", "IMPACT": "#e34948"}
STATUS = {"LOW": ("#1f7a3a", "●"), "GUARDED": ("#8a6d00", "▲"), "ELEVATED": ("#b35900", "▲"),
          "HIGH": ("#b3261e", "■")}

st.markdown("""<style>
.badge{display:inline-block;padding:1px 8px;border-radius:10px;font-size:0.72rem;font-weight:600;
       letter-spacing:.03em;margin-right:6px}
.obs{background:#ecebe7;color:#3b3a37}.inf{background:#fdf0d5;color:#7a5200}.pred{background:#dde9fa;color:#1b4f91}
.card{border:1px solid #e4e3df;border-radius:10px;padding:12px 14px;background:#fff}
.card h4{margin:0;font-size:0.78rem;color:#52514e;font-weight:600;text-transform:uppercase;letter-spacing:.04em}
.card .v{font-size:1.6rem;font-weight:700;margin-top:4px}
.card .s{font-size:0.8rem;color:#52514e}
</style>""", unsafe_allow_html=True)

OBS = '<span class="badge obs">OBSERVED</span>'
INF = '<span class="badge inf">INFERRED</span>'
PRED = '<span class="badge pred">PREDICTED</span>'


def card(col, title, value, sub="", colour=None):
    style = f"color:{colour}" if colour else ""
    col.markdown(f'<div class="card"><h4>{title}</h4><div class="v" style="{style}">{value}</div>'
                 f'<div class="s">{sub}</div></div>', unsafe_allow_html=True)


def fig_style(fig, h=320):
    fig.update_layout(height=h, margin=dict(l=10, r=10, t=30, b=10), plot_bgcolor="#fcfcfb",
                      paper_bgcolor="#fcfcfb", font=dict(size=12, color="#0b0b0b"),
                      legend=dict(orientation="h", y=1.12, x=0), hovermode="x unified")
    fig.update_xaxes(gridcolor=GRID, zeroline=False)
    fig.update_yaxes(gridcolor=GRID, zeroline=False)
    return fig


# ------------------------------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_forecaster(path: str) -> Forecaster:
    return Forecaster(path)


@st.cache_data(show_spinner=False)
def get_scenarios():
    return {k: {"artifact": str(v["artifact"]), "days": {d: {kk: (str(vv) if vv else None) for kk, vv in x.items()}
                                                          for d, x in v["days"].items()},
                "flows_dir": str(v["flows_dir"]) if v["flows_dir"] else None}
            for k, v in discover().items()}


def entry_of(sc):
    e = dict(sc)
    e["flows_dir"] = Path(sc["flows_dir"]) if sc["flows_dir"] else None
    return e


@st.cache_data(show_spinner="Loading replay day…")
def get_day(scn_name: str, day: str):
    sc = entry_of(get_scenarios()[scn_name])
    return load_day(sc, day)


@st.cache_data(show_spinner="Running the world model over the whole day…")
def get_timeline(scn_name: str, day: str):
    sc = get_scenarios()[scn_name]
    fc = get_forecaster(sc["artifact"])
    states, labels, flows = get_day(scn_name, day)
    ctx = fc.context_from_states(states)
    return fc.timeline(ctx)


def get_ctx(scn_name, day, host):
    sc = get_scenarios()[scn_name]
    fc = get_forecaster(sc["artifact"])
    states, labels, flows = get_day(scn_name, day)
    if fc.entity == "host":
        flows = host_flows(entry_of(sc), day, host)
    return fc, fc.context_from_states(states[states["host"] == host], flows)


@st.cache_data(show_spinner="Simulating futures and computing explanations…")
def get_record(scn_name, day, host, t, samples):
    fc, ctx = get_ctx(scn_name, day, host)
    return fc.record(ctx, host, int(t), n_samples=samples)


# ------------------------------------------------------------------------------------------------
scenarios = get_scenarios()
st.sidebar.title("🛡️ Attack Forecasting")
st.sidebar.caption("SIH26153 · World-model forecasting · runs fully offline")
if not scenarios:
    st.error("No trained model found. Train one (`python train.py --config configs/cic2018_network.yaml`) "
             "or keep the committed `demo/` folder.")
    st.stop()
scn = st.sidebar.selectbox("Scenario (model + replay data)", list(scenarios))
fc = get_forecaster(scenarios[scn]["artifact"])
day = st.sidebar.selectbox("Replay day (held-out test data)", list(scenarios[scn]["days"]))
states, labels, _ = get_day(scn, day)
tl = get_timeline(scn, day)
if tl.empty:
    st.warning("Not enough windows in this replay day.")
    st.stop()
windows = sorted(tl["window"].unique())
ws = fc.cfg["window_s"]
fmt_w = lambda w: pd.Timestamp(int(w) * ws, unit="s", tz="UTC").strftime("%H:%M UTC")  # noqa: E731
default_w = windows[len(windows) // 3]
now_w = st.sidebar.select_slider("Now (replay clock)", options=windows, value=default_w, format_func=fmt_w)
show_truth = st.sidebar.toggle("Show dataset label (not available in deployment)", value=False)
samples = st.sidebar.select_slider("Monte-Carlo futures", options=[8, 16, 32, 64], value=32)
st.sidebar.markdown(f"**Horizon:** K = {fc.K} windows × {ws}s · **History:** L = {fc.L}")
st.sidebar.markdown(f"**Alert threshold:** {fc.threshold:.2f} (validation FPR {fc.bundle['target_fpr']:.0%})")

now = tl[tl["window"] == now_w].sort_values("p_within", ascending=False)
entity_opts = list(now["host"])
host = st.sidebar.selectbox("Entity", entity_opts, index=0,
                            help="Hosts are sorted by forecast risk at the current time.")
t_now = int(now.loc[now["host"] == host, "t"].iloc[0])
rec = get_record(scn, day, host, t_now, samples)
pred, inf = rec["predicted"], rec["inferred"]

st.markdown(f"### {host if fc.entity == 'host' else 'Monitored network'} · {fmt_w(now_w)} · {day}")
st.caption(f"{OBS} computed directly from traffic  {INF} interpretation of the current state  "
           f"{PRED} world-model forecast of future windows", unsafe_allow_html=True)

tabs = st.tabs(["Overview", "Timeline", "Network", "Explain", "Evidence", "Forecast & what-if",
                "Audit ledger", "Model & results"])

# ------------------------------------------------------------------ overview
with tabs[0]:
    c = st.columns(5)
    colour, icon = STATUS[pred["risk_level"]]
    card(c[0], "Risk level", f"{icon} {pred['risk_level']}", f"confidence {pred['confidence']}", colour)
    card(c[1], f"P(attack ≤ {fc.K} windows)", f"{pred['p_attack_within_K']:.0%}",
         f"MC band {pred['band_5_95'][0]:.0%}–{pred['band_5_95'][1]:.0%}")
    stage = pred["most_likely_future_stage"]
    card(c[2], "Predicted stage", STAGE_LABEL.get(stage, "—") if stage else "—",
         ", ".join(f"{t['id']} {t['name']}" for t in tactic_refs(stage)) or "below alert threshold")
    hs = pred["first_crossing_step"]
    card(c[3], "Forecast horizon", f"t+{hs}" if hs else "none", f"{hs * ws // 60} min ahead" if hs else f"within {fc.K} windows")
    sp = pred.get("surprise_percentile")
    card(c[4], "Surprise (unexpected behaviour)", f"{sp:.0%}" if sp is not None else "—",
         "percentile vs benign validation")
    st.markdown(f"{PRED} *{pred['statement']}*", unsafe_allow_html=True)
    st.markdown(f"{INF} current state looks like **{STAGE_LABEL[inf['current_stage']]}** "
                f"({inf['current_stage_probs'][inf['current_stage']]:.0%})", unsafe_allow_html=True)

    st.markdown("#### Entities ranked by forecast risk at this time")
    recent = tl[(tl["window"] <= now_w) & (tl["window"] > now_w - 10)]
    alerts10 = recent.groupby("host")["alert"].sum()
    top = now.head(15).copy()
    stage_cols = [c for c in top.columns if c.startswith("now_")]
    top["current (inferred)"] = top[stage_cols].idxmax(axis=1).str.replace("now_", "").map(STAGE_LABEL)
    top["alerts, last 10 windows"] = top["host"].map(alerts10).fillna(0).astype(int)
    st.dataframe(top[["host", "p_within", "current (inferred)", "alerts, last 10 windows", "surprise_pct"]]
                 .rename(columns={"p_within": f"P(attack ≤ {fc.K})", "surprise_pct": "surprise pct"}),
                 hide_index=True, width="stretch",
                 column_config={f"P(attack ≤ {fc.K})": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.2f"),
                                "surprise pct": st.column_config.NumberColumn(format="%.2f")})

# ------------------------------------------------------------------ timeline
with tabs[1]:
    g = tl[tl["host"] == host].sort_values("window")
    past = g[g["window"] <= now_w]
    times = pd.to_datetime(past["window"] * ws, unit="s", utc=True)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=times, y=past["p_within"], name="forecast made at each past window",
                             line=dict(color=BLUE, width=2)))
    steps = np.array([[pred["stage_probs"][f"t+{k + 1}"][c] for c in fc.classes] for k in range(fc.K)])
    fut_t = pd.to_datetime([(now_w + k + 1) * ws for k in range(fc.K)], unit="s", utc=True)
    fig.add_trace(go.Bar(x=fut_t, y=pred["cumulative_by_step"], name="P(attack by t+k), calibrated",
                         marker_color="#9dbfeb", width=ws * 800))
    fig.add_hline(y=fc.threshold, line_dash="dash", line_color=INK2, annotation_text="alert threshold",
                  annotation_position="top left")
    fig.add_vline(x=pd.Timestamp(int(now_w) * ws, unit="s", tz="UTC"), line_color="#0b0b0b", line_width=1)
    if show_truth:
        lab = labels[labels["host"] == host].set_index("window")["stage"]
        lt = lab.reindex(g["window"]).fillna(0)
        att = lt[lt > 0]
        fig.add_trace(go.Scatter(x=pd.to_datetime(att.index * ws, unit="s", utc=True), y=[-0.05] * len(att),
                                 mode="markers", marker=dict(symbol="square", size=6, color="#e34948"),
                                 name="dataset label: attack (ground truth)"))
    fig.update_yaxes(range=[-0.1, 1.05], title="probability")
    st.markdown(f"{PRED} past forecasts, current time (black line) and the next {fc.K} windows", unsafe_allow_html=True)
    st.plotly_chart(fig_style(fig, 360), width="stretch")

    st.markdown(f"#### Attack-stage progression {INF} past → {PRED} future", unsafe_allow_html=True)
    now_cols = [f"now_{c}" for c in fc.classes]
    seq = past.tail(20)
    inferred = seq[now_cols].to_numpy().argmax(1)
    chips = []
    for w, i in zip(seq["window"], inferred):
        cname = fc.classes[i]
        chips.append(f'<span title="{fmt_w(w)}" style="display:inline-block;width:22px;height:22px;border-radius:4px;'
                     f'margin:1px;background:{STAGE_COLOURS[cname]}"></span>')
    chips.append('<span style="display:inline-block;width:3px;height:26px;background:#0b0b0b;margin:0 6px"></span>')
    for k in range(fc.K):
        cname = fc.classes[int(steps[k].argmax())]
        chips.append(f'<span title="t+{k + 1}: {cname} {steps[k].max():.0%}" style="display:inline-block;width:22px;height:22px;'
                     f'border-radius:4px;margin:1px;background:{STAGE_COLOURS[cname]};opacity:{0.35 + 0.65 * steps[k].max():.2f};'
                     f'outline:2px dashed #2a78d6"></span>')
    st.markdown("".join(chips), unsafe_allow_html=True)
    st.caption(" · ".join(f'<span style="color:{STAGE_COLOURS[c]}">■</span> {STAGE_LABEL[c]}' for c in fc.classes)
               + " — dashed = predicted (opacity = probability)", unsafe_allow_html=True)

# ------------------------------------------------------------------ network
with tabs[2]:
    if fc.entity != "host":
        st.info("This dataset's flow CSVs contain no IP addresses, so the model works on the network as a "
                "whole (one state per window). The host view is available in the host-level scenario.")
        wf = rec["observed"]["top_flows"]
        if wf:
            df = pd.DataFrame(wf)
            st.markdown(f"{OBS} busiest destination services in the last 3 windows", unsafe_allow_html=True)
            fig = go.Figure(go.Bar(x=df["flows"], y=df["dst_port"].astype(int).astype(str) + "/" +
                                   df["proto"].astype(int).astype(str), orientation="h", marker_color=BLUE))
            fig.update_yaxes(autorange="reversed", title="dst port / protocol")
            fig.update_xaxes(title="flows")
            st.plotly_chart(fig_style(fig, 360), width="stretch")
    else:
        _, ctx = get_ctx(scn, day, host)
        f = ctx.flows
        f = f[(f["window"] > now_w - 3) & (f["window"] <= now_w)] if len(f) else f
        if f is None or not len(f):
            st.info("No flow records available for this host in the last windows.")
        else:
            import networkx as nx
            e = f.groupby(["src_ip", "dst_ip"]).size().reset_index(name="flows").sort_values("flows", ascending=False).head(60)
            G = nx.DiGraph()
            for r in e.itertuples():
                G.add_edge(r.src_ip, r.dst_ip, w=r.flows)
            pos = nx.spring_layout(G, seed=1, k=0.9)
            risk = now.set_index("host")["p_within"].to_dict()
            ex, ey = [], []
            for a, b in G.edges():
                ex += [pos[a][0], pos[b][0], None]
                ey += [pos[a][1], pos[b][1], None]
            fig = go.Figure(go.Scatter(x=ex, y=ey, mode="lines", line=dict(color="#c9c8c2", width=1), hoverinfo="skip"))
            nodes = list(G.nodes())
            fig.add_trace(go.Scatter(
                x=[pos[n][0] for n in nodes], y=[pos[n][1] for n in nodes], mode="markers+text",
                text=[n if (n == host or risk.get(n, 0) >= fc.threshold) else "" for n in nodes], textposition="top center",
                marker=dict(size=[18 if n == host else 11 for n in nodes],
                            color=[risk.get(n, np.nan) for n in nodes], colorscale=[[0, "#dde9fa"], [1, "#b3261e"]],
                            cmin=0, cmax=1, line=dict(color="#fcfcfb", width=2), showscale=True,
                            colorbar=dict(title="P(attack)")),
                hovertext=[f"{n}<br>P(attack≤K)={risk[n]:.2f}" if n in risk else f"{n}<br>(external / not modelled)" for n in nodes],
                hoverinfo="text"))
            fig.update_xaxes(visible=False)
            fig.update_yaxes(visible=False)
            fig.update_layout(showlegend=False)
            st.markdown(f"{OBS} communication graph of **{host}**, last 3 windows (grey nodes = external or unmodelled hosts)",
                        unsafe_allow_html=True)
            st.plotly_chart(fig_style(fig, 480), width="stretch")
            st.dataframe(e.rename(columns={"src_ip": "source", "dst_ip": "destination"}), hide_index=True,
                         width="stretch")

# ------------------------------------------------------------------ explain
with tabs[3]:
    ex = rec["explanation"]
    st.markdown(f"{PRED} **Why the model estimates P(attack ≤ {fc.K}) = {pred['p_attack_within_K']:.0%}** "
                f"— Integrated Gradients vs. the median benign state", unsafe_allow_html=True)
    if ex["evidence"]:
        for i, ev in enumerate(ex["evidence"], 1):
            st.markdown(f"**{i}.** {ev['text']}  \n<small>feature `{ev['feature']}` · attribution {ev['attribution']:+.3f}</small>",
                        unsafe_allow_html=True)
    else:
        st.info("No feature pushes the risk above the benign reference at this time.")
    if ex["features_against"]:
        st.markdown("**Pointing the other way (lowering risk):** " + ", ".join(
            f"{DESCRIPTIONS.get(n, n)} ({v:+.3f})" for n, v in ex["features_against"]))
    heat = np.array(ex["heatmap"])
    order = np.argsort(-np.abs(heat).sum(0))[:12]
    lim = float(np.abs(heat[:, order]).max()) or 1.0
    fig = go.Figure(go.Heatmap(z=heat[:, order].T, x=[f"t-{fc.L - 1 - i}" if i < fc.L - 1 else "t" for i in range(fc.L)],
                               y=[fc.features[i] for i in order], colorscale="RdBu", reversescale=True, zmin=-lim, zmax=lim,
                               colorbar=dict(title="attribution")))
    fig.update_yaxes(autorange="reversed")
    st.markdown("Temporal attribution (which windows and features mattered)")
    st.plotly_chart(fig_style(fig, 420), width="stretch")
    st.caption(f"IG convergence delta {ex['convergence_delta']:.2e} (should be ≈0). "
               "Explanations describe what the model relied on — not proof of attacker intent.")

# ------------------------------------------------------------------ evidence
with tabs[4]:
    st.markdown(f"#### {INF} Candidate ATT&CK techniques (behaviour consistent with…)", unsafe_allow_html=True)
    ct = inf["candidate_techniques"]
    if ct:
        for tq in ct:
            st.markdown(f"- **{tq['id']} {tq['name']}** · {', '.join(x['name'] for x in tq['tactics'])} · "
                        f"strength *{tq['strength']}* — {tq['note']}  \n  <small>evidence: {tq['evidence']}</small>",
                        unsafe_allow_html=True)
    else:
        st.write("No indicator rule fired for the current window.")
    st.markdown(f"#### {OBS} Top flows (last 3 windows)", unsafe_allow_html=True)
    tf = rec["observed"]["top_flows"]
    st.dataframe(pd.DataFrame(tf), hide_index=True, width="stretch") if tf else st.write("—")
    st.markdown(f"#### {OBS} Key features, last 3 windows", unsafe_allow_html=True)
    obs = rec["observed"]["window_features_last3"]
    keys = [e["feature"] for e in rec["explanation"]["evidence"]] + [k for k in obs if k not in
                                                                     [e["feature"] for e in rec["explanation"]["evidence"]]][:10]
    st.dataframe(pd.DataFrame({"feature": keys, "meaning": [DESCRIPTIONS.get(k, "") for k in keys],
                               "t-2": [obs[k][0] for k in keys], "t-1": [obs[k][1] for k in keys],
                               "t": [obs[k][2] for k in keys]}), hide_index=True, width="stretch")
    st.caption(f"Stages without ground truth in the training data (never predicted): {', '.join(rec['not_supported'])}.")

# ------------------------------------------------------------------ forecast & what-if
with tabs[5]:
    band = np.array([[[pred["stage_probs"][f"t+{k + 1}"][c]] for c in fc.classes] for k in range(fc.K)])[..., 0]
    fig = go.Figure()
    xs = [f"t+{k + 1}" for k in range(fc.K)]
    for i, c in enumerate(fc.classes):
        if c == "BENIGN":
            continue
        fig.add_trace(go.Bar(x=xs, y=band[:, i], name=STAGE_LABEL[c], marker_color=STAGE_COLOURS[c]))
    fig.update_layout(barmode="stack")
    fig.update_yaxes(title="P(stage) per future window", range=[0, 1])
    st.markdown(f"{PRED} attack-stage probabilities for each future window", unsafe_allow_html=True)
    st.plotly_chart(fig_style(fig, 320), width="stretch")

    st.markdown(f"#### {PRED} What-if simulation", unsafe_allow_html=True)
    st.caption("Clamp selected features of every *imagined* future window to their benign level and re-run "
               "the world model. This is a model-based what-if, not a causal guarantee.")
    choices = st.multiselect("Features to hold at benign level in the future",
                             fc.features, default=[f for f in ["uniq_dst_ports_out", "new_peers_out", "svc_smb",
                                                               "uniq_dst_ports", "n_flows"] if f in fc.features][:2],
                             format_func=lambda f: f"{f} — {DESCRIPTIONS.get(f, '')}")
    if choices and st.button("Run what-if simulation"):
        idx = [fc.features.index(f) for f in choices]
        _, ctx = get_ctx(scn, day, host)
        wi = fc.record(ctx, host, t_now, n_samples=samples, with_explanation=False,
                       intervention=feature_intervention(idx, [float(fc.baseline[i]) for i in idx]))
        a, b = pred["p_attack_within_K"], wi["predicted"]["p_attack_within_K"]
        cc = st.columns(2)
        card(cc[0], "Forecast as observed", f"{a:.0%}")
        card(cc[1], "With intervention", f"{b:.0%}", f"change {b - a:+.0%}")

# ------------------------------------------------------------------ ledger
with tabs[6]:
    from netwm.ledger.hashchain import EvidenceLedger
    st.markdown("Tamper-evident log of forecasts: SHA-256 hash chain + Ed25519 signatures. "
                "Only hashes and model outputs are recorded — never raw traffic. The forecasting system "
                "works identically without it.")
    if "ledger_dir" not in st.session_state:
        st.session_state.ledger_dir = tempfile.mkdtemp(prefix="netwm_ledger_")
    lpath = Path(st.session_state.ledger_dir) / "ledger.sqlite"
    led = EvidenceLedger(lpath)
    c1, c2, c3 = st.columns(3)
    if c1.button("Record this forecast"):
        r = led.append(rec)
        st.success(f"record #{r['seq']} · {r['record_hash'][:20]}…")
    if c2.button("Verify chain"):
        v = led.verify()
        (st.success if v["ok"] else st.error)(
            f"Chain OK — {v['n']} records verified" if v["ok"] else
            f"TAMPERING DETECTED at record #{v['first_bad_seq']} ({v['reason']})")
    if c3.button("Tamper demo: edit a stored record"):
        con = sqlite3.connect(lpath)
        row = con.execute("SELECT seq, payload FROM ledger ORDER BY seq LIMIT 1").fetchone()
        if row:
            p = json.loads(row[1])
            p["predicted"]["p_attack_within_K"] = 0.0
            con.execute("UPDATE ledger SET payload=? WHERE seq=?", (json.dumps(p), row[0]))
            con.commit()
            st.warning(f"Record #{row[0]} silently changed to P=0. Now press 'Verify chain'.")
        else:
            st.info("Record a forecast first.")
    st.dataframe(pd.DataFrame(led.records(50)), hide_index=True, width="stretch")
    st.caption(f"Merkle root of all records: `{led.merkle()}` — can be anchored on a local EVM chain "
               "(optional, `netwm/ledger/anchor_evm.py`).")

# ------------------------------------------------------------------ model & results
with tabs[7]:
    art = Path(scenarios[scn]["artifact"])
    mp = art / "metrics.json"
    st.markdown(f"**Model:** host-centric recurrent world model (GRU) · {len(fc.features)} features · "
                f"trained on `{fc.bundle['dataset']}` · sha256 `{fc.bundle['model_sha256'][:16]}` · "
                f"ATT&CK v{rec['attack_version']}")
    if mp.exists():
        m = json.load(open(mp))
        rows = []
        for name, r in m.items():
            if isinstance(r, dict) and "early_warning" in r:
                rows.append({"model": name, "PR-AUC": r["pr_auc"], "PR-AUC onset": r["pr_auc_onset"],
                             "precision": r["precision"], "recall": r["recall"], "F1": r["f1"], "FPR": r["fpr"],
                             "recall T-1": r["early_warning"].get("1", {}).get("recall"),
                             "recall T-2": r["early_warning"].get("2", {}).get("recall")})
        st.dataframe(pd.DataFrame(rows).round(3), hide_index=True, width="stretch")
        st.caption("Held-out test days; thresholds chosen on validation. B0 persistence uses the true current label "
                   "(oracle) and is a reference only.")
    rep = ROOT / "reports" / fc.bundle["dataset"]
    imgs = sorted(rep.glob("*.png")) if rep.exists() else sorted((art / "figures").glob("*.png"))
    cols = st.columns(2)
    for i, p in enumerate(imgs[:8]):
        cols[i % 2].image(str(p), caption=p.stem, width="stretch")

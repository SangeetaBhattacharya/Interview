import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import scipy.stats as stats

# ---------------------------
# NHS theme
# ---------------------------
NHS_BLUE = "#005EB8"
NHS_DARK = "#003087"
NHS_GREY = "#6b7280"
NHS_BG = "#f6f7fb"
GOOD = "#007f3b"
BAD = "#d5281b"
AMBER = "#FFB81C"

NHS_LOGO_URL = "https://www.england.nhs.uk/wp-content/themes/nhsengland/static/img/nhs-logo-blue.svg"

st.set_page_config(page_title="SPC charts", layout="wide")

# ---------------------------
# CSS
# ---------------------------
st.markdown(
    f"""
    <style>
      .stApp {{
        background: {NHS_BG};
      }}
      .nhs-header {{
        display:flex; align-items:flex-start; justify-content:space-between;
        gap:16px; margin-top: 4px; margin-bottom: 8px;
      }}
      .nhs-title {{
        font-size: 30px; font-weight: 800; margin: 0; color: #111827;
      }}
      .nhs-desc {{
        background: white; border: 1px solid #e5e7eb; border-radius: 12px;
        padding: 10px 12px; font-size: 14px; color: #111827; line-height: 1.5;
        margin-bottom: 10px;
      }}
      .nhs-desc a {{
        color: {NHS_BLUE}; font-weight: 700; text-decoration: none;
      }}
      .nhs-desc a:hover {{ text-decoration: underline; }}

      .tabbar {{
        background: white; border: 1px solid #e5e7eb; border-radius: 10px;
        padding: 10px 12px; margin-bottom: 10px; display:flex; flex-wrap:wrap; gap:10px;
      }}
      .tab {{
        padding: 8px 12px; border-radius: 8px; font-weight: 700; font-size: 13px;
        color: {NHS_BLUE}; border: 1px solid transparent;
      }}
      .tab.active {{
        color: #111827; background: #eef6ff; border: 1px solid #cfe6ff;
      }}

      .toolbar {{
        display:flex; justify-content:space-between; align-items:center; gap:10px;
        background:white; border:1px solid #e5e7eb; border-radius:12px;
        padding: 8px 12px; margin-bottom: 10px;
      }}
      .pill {{
        display:inline-flex; align-items:center; gap:8px;
        padding: 8px 12px; border-radius: 999px;
        border: 1px solid #e5e7eb; background: #f9fafb;
        font-weight: 800; font-size: 13px;
      }}
      .help a {{
        display:inline-flex; align-items:center; gap:8px;
        padding: 8px 10px; border-radius: 999px;
        border: 1px solid {NHS_BLUE}; color: {NHS_BLUE};
        font-weight: 800; font-size: 13px; text-decoration:none;
        background: white; margin-left: 8px;
      }}
      .help a:hover {{ background:#eef6ff; }}

      .kpis {{
        display:grid; grid-template-columns: repeat(5, 1fr);
        gap: 10px; margin-bottom: 10px;
      }}
      .kpi {{
        background:white; border:1px solid #e5e7eb; border-radius:14px; padding: 10px 12px;
      }}
      .kpi .lab {{ color: {NHS_GREY}; font-size: 12px; margin:0; }}
      .kpi .val {{ font-size: 20px; font-weight: 900; margin:4px 0 0; }}
      .kpi .sm  {{ color: {NHS_GREY}; font-size: 12px; margin:4px 0 0; }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------
# Mock SPC data
# ---------------------------
np.random.seed(12)

def generate_series(
    view="National",
    indicator="Babies who were born preterm (Rate per 1,000)",
    provider="Airedale NHS Foundation Trust",
):
    months = pd.date_range("2024-01-01", "2025-10-01", freq="MS")
    n = len(months)

    base = 62.0 if view == "National" else 60.5
    if view != "National":
        base += (hash(provider) % 7 - 3) * 0.4

    noise = np.random.normal(0, 1.6, n)
    values = np.zeros(n)
    values[0] = base + noise[0]
    for i in range(1, n):
        values[i] = 0.65 * values[i - 1] + 0.35 * (base + noise[i])

    # Inject special-cause patterns
    if n > 11:
        values[11] += 6.0  # ~Dec 2024 spike
    if view != "National" and n > 18:
        values[18] -= 5.0  # provider dip later

    return pd.DataFrame({"Month": months, "Rate": values})


def spc_limits(df):
    mean = df["Rate"].mean()
    sd = df["Rate"].std(ddof=1)
    ucl = mean + 3 * sd
    lcl = mean - 3 * sd
    return mean, ucl, lcl


def rule_flags(df, mean, ucl, lcl):
    df = df.copy()
    df["Outside"] = (df["Rate"] > ucl) | (df["Rate"] < lcl)

    above = df["Rate"] > mean
    below = df["Rate"] < mean

    def mark_runs(bool_series, run_len=8):
        marks = np.zeros(len(bool_series), dtype=bool)
        count = 0
        for i, b in enumerate(bool_series):
            count = count + 1 if b else 0
            if count >= run_len:
                marks[i - run_len + 1 : i + 1] = True
        return marks

    df["Run8"] = mark_runs(above) | mark_runs(below)
    df["SpecialCause"] = df["Outside"] | df["Run8"]
    return df


def build_spc_figure(df, mean, ucl, lcl, show_ann=True, show_run=True):
    df = rule_flags(df, mean, ucl, lcl)
    in_control = df[~df["SpecialCause"]]
    special = df[df["SpecialCause"]]
    run_only = df[df["Run8"] & ~df["Outside"]]

    fig = go.Figure()

    sigma = (ucl - mean) / 3.0 if np.isfinite(ucl) and np.isfinite(mean) else 0.0
    z1_low, z1_high = mean - 1 * sigma, mean + 1 * sigma
    z2_low, z2_high = mean - 2 * sigma, mean + 2 * sigma
    z3_low, z3_high = mean - 3 * sigma, mean + 3 * sigma

    fig.add_hrect(y0=z3_low, y1=z2_low, fillcolor=NHS_BLUE, opacity=0.04, line_width=0)
    fig.add_hrect(y0=z2_low, y1=z1_low, fillcolor=NHS_BLUE, opacity=0.06, line_width=0)
    fig.add_hrect(y0=z1_low, y1=z1_high, fillcolor=NHS_BLUE, opacity=0.08, line_width=0)
    fig.add_hrect(y0=z1_high, y1=z2_high, fillcolor=NHS_BLUE, opacity=0.06, line_width=0)
    fig.add_hrect(y0=z2_high, y1=z3_high, fillcolor=NHS_BLUE, opacity=0.04, line_width=0)

    fig.add_trace(
        go.Scatter(
            x=in_control["Month"],
            y=in_control["Rate"],
            mode="lines+markers",
            name="In control",
            line=dict(color=NHS_BLUE, width=2),
            marker=dict(size=7),
            hovertemplate="%{x|%b %Y}<br>Rate: %{y:.2f}<extra></extra>",
        )
    )

    if not special.empty:
        fig.add_trace(
            go.Scatter(
                x=special["Month"],
                y=special["Rate"],
                mode="markers",
                name="Special cause",
                marker=dict(size=12, color=BAD, line=dict(width=1, color="#111827")),
                hovertemplate="%{x|%b %Y}<br><b>Special cause</b><br>Rate: %{y:.2f}<extra></extra>",
            )
        )

    if show_run and (not run_only.empty):
        fig.add_trace(
            go.Scatter(
                x=run_only["Month"],
                y=run_only["Rate"],
                mode="markers",
                name="Run-of-8 signal",
                marker=dict(size=10, symbol="diamond", color=AMBER),
                hovertemplate="%{x|%b %Y}<br><b>Run-of-8</b><br>Rate: %{y:.2f}<extra></extra>",
            )
        )

    fig.add_hline(
        y=mean,
        line_color="#111827",
        line_width=2,
        annotation_text="Mean" if show_ann else None,
        annotation_position="top left",
    )
    fig.add_hline(
        y=ucl,
        line_dash="dash",
        line_color=NHS_GREY,
        line_width=2,
        annotation_text="UCL" if show_ann else None,
        annotation_position="top left",
    )
    fig.add_hline(
        y=lcl,
        line_dash="dash",
        line_color=NHS_GREY,
        line_width=2,
        annotation_text="LCL" if show_ann else None,
        annotation_position="bottom left",
    )

    fig.update_layout(
        height=520,
        margin=dict(l=20, r=20, t=10, b=10),
        xaxis_title=None,
        yaxis_title=None,
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        font=dict(family="Inter, Arial"),
    )
    return fig, df


# ---------------------------
# CUSUM aligned with your guidance (demo implementation)
# ---------------------------
def _rate_to_prob(indicator: str, rate_value: float) -> float:
    name = indicator.lower()
    if "percent" in name:
        return max(0.0, min(1.0, rate_value / 100.0))
    if "rate per 1,000" in name or "rate per 1000" in name:
        return max(0.0, min(1.0, rate_value / 1000.0))
    if "rate per 100" in name:
        return max(0.0, min(1.0, rate_value / 100.0))
    return max(0.0, min(1.0, rate_value / 1000.0))


def _make_births_series(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = int(rng.integers(350, 900))
    season = 1.0 + 0.05 * np.sin(np.linspace(0, 2 * np.pi, n))
    noise = rng.normal(0, 25, n)
    births = np.clip(base * season + noise, 200, None)
    return births.astype(int)


def build_cusum_vlad_and_signals(
    df: pd.DataFrame,
    indicator: str,
    national_reference_rate: float,
    arl_level1: int = 20,    # ~95%
    arl_level2: int = 100,   # ~99%
    seed: int = 42,
):
    """
    - CUSUM chart: cumulative variation between observed events and expected events (national reference rate)
    - Excess events (VLAD): cumulative (Observed - Expected) vs national reference

    Thresholds are approximated for a prototype (production would replicate CUSUMdesign getH() Markov chain thresholds).
    """
    dfx = df.copy().reset_index(drop=True)
    n = len(dfx)

    births = _make_births_series(n, seed=seed)

    p_nat = _rate_to_prob(indicator, float(national_reference_rate))
    p_loc = np.array([_rate_to_prob(indicator, float(r)) for r in dfx["Rate"].values])

    expected = births * p_nat
    observed = births * p_loc

    # Excess events (VLAD)
    excess = observed - expected
    cum_excess = np.cumsum(excess)

    # k based on μ0 and μ1 = 2*μ0 (doubling)
    mu0 = np.maximum(expected, 1e-6)
    mu1 = 2.0 * mu0
    k = (mu1 - mu0) / (np.log(mu1) - np.log(mu0))

    cusum = np.zeros(n)
    for i in range(n):
        prev = cusum[i - 1] if i > 0 else 0.0
        cusum[i] = prev + observed[i] - k[i]

    # Prototype thresholds from ARL idea (simple scaling)
    inc = observed - k
    inc_sd = float(np.std(inc, ddof=1)) if n > 1 else 1.0
    inc_sd = inc_sd if np.isfinite(inc_sd) and inc_sd > 0 else 1.0
    H1 = inc_sd * np.log(max(arl_level1, 2))
    H2 = inc_sd * np.log(max(arl_level2, 2))

    # Indices above thresholds (ALL points)
    sig1_idx = np.where(cusum >= H1)[0]
    sig2_idx = np.where(cusum >= H2)[0]
    sig2_set = set(sig2_idx.tolist())
    sig1_only_idx = np.array([i for i in sig1_idx if i not in sig2_set], dtype=int)

    # --- CUSUM figure
    fig_cusum = go.Figure()
    fig_cusum.add_trace(
        go.Scatter(
            x=dfx["Month"],
            y=cusum,
            mode="lines+markers",
            name="CUSUM statistic",
            line=dict(color=NHS_BLUE, width=2),
            marker=dict(size=6),
            hovertemplate="%{x|%b %Y}<br>CUSUM: %{y:.2f}<extra></extra>",
        )
    )
    fig_cusum.add_hline(
        y=H1, line_dash="dot", line_color=AMBER, line_width=2,
        annotation_text="Level 1 (95%)", annotation_position="top left"
    )
    fig_cusum.add_hline(
        y=H2, line_dash="dot", line_color=BAD, line_width=2,
        annotation_text="Level 2 (99%)", annotation_position="top left"
    )
    fig_cusum.add_hline(y=0, line_color="#111827", line_width=1)

    # Signal dots: ALL points (amber for >=H1, red for >=H2; red takes priority)
    if len(sig1_only_idx) > 0:
        fig_cusum.add_trace(
            go.Scatter(
                x=dfx["Month"].iloc[sig1_only_idx],
                y=cusum[sig1_only_idx],
                mode="markers",
                name="Signal level 1 (amber)",
                marker=dict(size=11, color=AMBER),
                hovertemplate="%{x|%b %Y}<br><b>Level 1 signal (95%)</b><extra></extra>",
            )
        )
    if len(sig2_idx) > 0:
        fig_cusum.add_trace(
            go.Scatter(
                x=dfx["Month"].iloc[sig2_idx],
                y=cusum[sig2_idx],
                mode="markers",
                name="Signal level 2 (red)",
                marker=dict(size=12, color=BAD),
                hovertemplate="%{x|%b %Y}<br><b>Level 2 signal (99%)</b><extra></extra>",
            )
        )

    fig_cusum.update_layout(
        height=420,
        margin=dict(l=20, r=20, t=10, b=10),
        xaxis_title="Month",
        yaxis_title="CUSUM statistic (Observed vs Expected events)",
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        font=dict(family="Inter, Arial"),
    )

    # --- Excess events (VLAD) figure + matching signal markers
    fig_excess = go.Figure()
    fig_excess.add_trace(
        go.Scatter(
            x=dfx["Month"],
            y=cum_excess,
            mode="lines+markers",
            name="Cumulative excess events",
            line=dict(color=NHS_BLUE, width=2),
            marker=dict(size=6),
            hovertemplate="%{x|%b %Y}<br>Cumulative excess: %{y:.2f}<extra></extra>",
        )
    )
    fig_excess.add_hline(y=0, line_color="#111827", line_width=1)

    # Mark the same signal months on VLAD, so signals correspond visually
    if len(sig1_only_idx) > 0:
        fig_excess.add_trace(
            go.Scatter(
                x=dfx["Month"].iloc[sig1_only_idx],
                y=cum_excess[sig1_only_idx],
                mode="markers",
                name="Signal level 1 (amber)",
                marker=dict(size=11, color=AMBER),
                hovertemplate="%{x|%b %Y}<br><b>Level 1 signal month</b><extra></extra>",
            )
        )
    if len(sig2_idx) > 0:
        fig_excess.add_trace(
            go.Scatter(
                x=dfx["Month"].iloc[sig2_idx],
                y=cum_excess[sig2_idx],
                mode="markers",
                name="Signal level 2 (red)",
                marker=dict(size=12, color=BAD),
                hovertemplate="%{x|%b %Y}<br><b>Level 2 signal month</b><extra></extra>",
            )
        )

    fig_excess.update_layout(
        height=420,
        margin=dict(l=20, r=20, t=10, b=10),
        xaxis_title="Month",
        yaxis_title="Cumulative excess events vs national reference",
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        font=dict(family="Inter, Arial"),
    )

    out = dfx.copy()
    out["Births"] = births
    out["ExpectedEvents"] = expected
    out["ObservedEvents"] = observed
    out["ExcessEvents"] = excess
    out["CumExcessEvents"] = cum_excess
    out["CUSUM"] = cusum

    return fig_cusum, fig_excess, out, H1, H2, len(sig1_only_idx), len(sig2_idx)


# ---------------------------
# Diagnostics
# ---------------------------
def build_distribution_diagnostics(df):
    series = df["Rate"].dropna().astype(float)
    stats_dict = {
        "n": int(series.shape[0]),
        "mean": float(series.mean()),
        "sd": float(series.std(ddof=1)) if series.shape[0] > 1 else 0.0,
        "min": float(series.min()),
        "max": float(series.max()),
        "skew": float(stats.skew(series)) if series.shape[0] > 2 else 0.0,
    }

    hist_fig = px.histogram(df, x="Rate", nbins=12, opacity=0.85, title="Distribution (Histogram)")
    hist_fig.update_layout(
        height=320,
        margin=dict(l=10, r=10, t=40, b=10),
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis_title=None,
        yaxis_title=None,
        showlegend=False,
    )

    if stats_dict["sd"] > 0:
        xs = np.linspace(stats_dict["min"], stats_dict["max"], 200)
        ys = stats.norm.pdf(xs, loc=stats_dict["mean"], scale=stats_dict["sd"])
        ys_scaled = ys / ys.max() * (series.shape[0] / 4)
        hist_fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys_scaled,
                mode="lines",
                name="Normal curve (reference)",
                line=dict(color="#111827", width=2),
                hoverinfo="skip",
            )
        )

    box_fig = px.box(df, y="Rate", points="outliers", title="Spread (Box plot)")
    box_fig.update_layout(
        height=320,
        margin=dict(l=10, r=10, t=40, b=10),
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis_title=None,
        yaxis_title=None,
        showlegend=False,
    )

    return hist_fig, box_fig, stats_dict


# ---------------------------
# Header + context
# ---------------------------
st.markdown(
    f"""
    <div class="nhs-header">
      <div><p class="nhs-title">Statistical Process Control (SPC) charts</p></div>
      <div><img src="{NHS_LOGO_URL}" alt="NHS logo" style="height:42px;"></div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="tabbar">
      <div class="tab">Homepage</div>
      <div class="tab">Overview</div>
      <div class="tab">Org Profile</div>
      <div class="tab">CQIM</div>
      <div class="tab">CQIM+</div>
      <div class="tab active">CQIM SPC</div>
      <div class="tab">Comparison</div>
      <div class="tab">MCoC</div>
      <div class="tab">NMI</div>
      <div class="tab">NMI+</div>
      <div class="tab">Guidance</div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="nhs-desc">
      This page contains <b>Statistical Process Control (SPC)</b> charts showing variation over time for
      <b>Clinical Quality Improvement Metrics (CQIMs)</b>, and indicating where variation is statistically significant.
      SPC helps distinguish <b>common-cause</b> variation from <b>special-cause</b> signals that may require investigation or action.
      Further details on interpreting SPC charts can be found on the <a href="#" onclick="return false;">Guidance</a> tab.
    </div>
    """,
    unsafe_allow_html=True,
)

st.info(
    "Run-of-8 signal: If 8 consecutive data points fall above or below the mean, this indicates a sustained shift unlikely due to random variation (special cause)."
)

# ---------------------------
# Controls
# ---------------------------
c1, c2, c3, c4 = st.columns([1.1, 2, 2, 1.4])

with c1:
    view = st.radio("", ["National SPC charts view", "Provider SPC charts view"], index=0)

with c2:
    indicator = st.selectbox(
        "Select indicator",
        [
            "Babies readmitted to hospital who were under 30 days old (Percent)",
            "Babies that were fully or partially breastfed at 6 to 8 weeks old (Percent)",
            "Babies who were born preterm (Rate per 1,000)",
            "Babies with a first feed of breast milk (Percent)",
            "Babies with an APGAR score between 0 and 6 (Rate per 1,000)",
            "Caesarean section rate for Robson Group 1 women (Percent)",
            "Caesarean section rate for Robson Group 2 women (Percent)",
            "Caesarean section rate for Robson Group 5 women (Percent)",
            "Women who had a 3rd or 4th degree tear at delivery (Rate per 1,000)",
            "Women who had a PPH of 1,500ml or more (Rate per 1,000)",
        ],
        index=2,
    )

with c3:
    provider = st.selectbox(
        "Select provider",
        [
            "Airedale NHS Foundation Trust (RCF)",
            "Ashford and St Peter's Hospitals NHS Foundation Trust (RTK)",
            "Barking, Havering and Redbridge University Hospitals NHS Trust (RF4)",
            "Barnsley Hospital NHS Foundation Trust (RFF)",
            "Barts Health NHS Trust (R1H)",
            "Basildon and Thurrock University Hospitals NHS Foundation Trust (RDD)",
            "Bedford Hospital NHS Trust (RC1)",
            "Bedfordshire Hospitals NHS Foundation Trust (RC9)",
            "Birmingham Women's and Children's NHS Foundation Trust (RQ3)",
        ],
        index=0,
        disabled=view.startswith("National"),
    )

with c4:
    show_ann = st.checkbox("Show Mean/UCL/LCL", value=True)
    show_run = st.checkbox("Highlight run-of-8", value=True)

# Generate full data (for bounds + filtering)
v = "National" if view.startswith("National") else "Provider"
df_full = generate_series(view=v, indicator=indicator, provider=provider)

min_d = df_full["Month"].min().date()
max_d = df_full["Month"].max().date()

d1, d2 = st.columns(2)
with d1:
    start_date = st.date_input("Start date", value=min_d, min_value=min_d, max_value=max_d, key="start_date")
with d2:
    end_date = st.date_input("End date", value=max_d, min_value=min_d, max_value=max_d, key="end_date")

# Toolbar (help links)
st.markdown(
    f"""
    <div class="toolbar">
      <div class="pill">Showing <b>{"National" if view.startswith("National") else "Provider"}</b> SPC charts view</div>
      <div class="help">
        <a href="#" onclick="return false;">ⓘ About SPC charts</a>
        <a href="#" onclick="return false;">ⓘ Smoking CQIMs &amp; SATOD</a>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------
# Filter by selected date range (month/year behaviour)
# ---------------------------
start_m = pd.Timestamp(start_date).to_period("M").to_timestamp()
end_m = pd.Timestamp(end_date).to_period("M").to_timestamp()

if start_m > end_m:
    st.error("Start date must be earlier than or equal to End date.")
    st.stop()

series_label = f"{start_m.strftime('%b %Y')} – {end_m.strftime('%b %Y')}"

df = df_full[(df_full["Month"] >= start_m) & (df_full["Month"] <= end_m)].copy()
if df.empty:
    st.warning("No data available for the selected date range.")
    st.stop()

# Build SPC
mean, ucl, lcl = spc_limits(df)
fig_spc, df_flagged = build_spc_figure(df, mean, ucl, lcl, show_ann=show_ann, show_run=show_run)

current_rate = float(df_flagged["Rate"].iloc[-1])
any_special = bool(df_flagged["SpecialCause"].any())
current_special = bool(df_flagged["SpecialCause"].iloc[-1])

kpi_html = f"""
<div class="kpis">
  <div class="kpi"><p class="lab">Current series</p><p class="val">{series_label}</p><p class="sm">Selected date range</p></div>
  <div class="kpi"><p class="lab">Current rate</p><p class="val">{current_rate:.2f}</p><p class="sm">Latest data point</p></div>
  <div class="kpi"><p class="lab">Mean</p><p class="val">{mean:.2f}</p><p class="sm">Centre line</p></div>
  <div class="kpi"><p class="lab">UCL / LCL</p><p class="val">{ucl:.2f} / {lcl:.2f}</p><p class="sm">±3σ limits</p></div>
  <div class="kpi"><p class="lab">Special cause (current)</p>
    <p class="val" style="color:{BAD if current_special else GOOD};">{'YES' if current_special else 'NO'}</p>
    <p class="sm">{'Signals present' if any_special else 'No signals detected'}</p>
  </div>
</div>
"""
st.markdown(kpi_html, unsafe_allow_html=True)

# ---------------------------
# Tabs: SPC / CUSUM / Diagnostics
# ---------------------------
tab_spc, tab_cusum, tab_diag = st.tabs(["SPC Chart", "CUSUM & Excess Events", "Diagnostics"])

with tab_spc:
    st.plotly_chart(fig_spc, use_container_width=True)

with tab_cusum:
    # National reference rate (aligned to your text): use National series mean over the selected period
    df_nat_full = generate_series(view="National", indicator=indicator, provider=provider)
    df_nat = df_nat_full[(df_nat_full["Month"] >= start_m) & (df_nat_full["Month"] <= end_m)].copy()
    nat_ref = float(df_nat["Rate"].mean()) if not df_nat.empty else float(df["Rate"].mean())

    fig_cusum, fig_excess, df_cu, H1, H2, n_sig1, n_sig2 = build_cusum_vlad_and_signals(
        df=df,
        indicator=indicator,
        national_reference_rate=nat_ref,
        arl_level1=20,
        arl_level2=100,
        seed=abs(hash(provider + indicator)) % (2**32),
    )

    st.markdown(f"**National reference rate (used for expected events):** {nat_ref:.2f}")
    st.caption("Prototype note: thresholds are approximated for this demo; production implementation would replicate CUSUMdesign getH() Markov chain thresholds.")

    # --- CUSUM Signal chart
    st.subheader("Maternity Outcomes Signal (CUSUM) chart")
    st.plotly_chart(fig_cusum, use_container_width=True)

    if n_sig2 > 0:
        st.info(f"CUSUM: {n_sig2} Level 2 signal point(s) (red) — 99% confidence not due to chance.")
    elif n_sig1 > 0:
        st.info(f"CUSUM: {n_sig1} Level 1 signal point(s) (amber) — 95% confidence not due to chance.")
    else:
        st.info("CUSUM: No Level 1 or Level 2 signal points in the selected period.")

    st.caption("X-axis: Month (time period).  •  Y-axis: CUSUM statistic based on cumulative variation between observed and expected events (national reference rate).")

    # --- Excess events (VLAD)
    st.subheader("Excess events (VLAD)")
    st.plotly_chart(fig_excess, use_container_width=True)

    last_excess = float(df_cu["CumExcessEvents"].iloc[-1])
    if last_excess > 0:
        st.info(f"Excess events: >0 — more events than expected vs national reference (latest cumulative excess = {last_excess:.1f}).")
    elif last_excess < 0:
        st.info(f"Excess events: <0 — fewer events than expected vs national reference (latest cumulative excess = {last_excess:.1f}).")
    else:
        st.info("Excess events: 0 — observed events match expected vs national reference.")

    st.caption("X-axis: Month (time period).  •  Y-axis: Cumulative excess events (Observed − Expected) vs national reference rate.")

with tab_diag:
    st.markdown("### Distribution diagnostics")
    hist_fig, box_fig, s = build_distribution_diagnostics(df)

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(hist_fig, use_container_width=True)
    with col2:
        st.plotly_chart(box_fig, use_container_width=True)

    st.markdown("### Summary statistics (selected date range)")
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("n", f"{s['n']}")
    m2.metric("Mean", f"{s['mean']:.2f}")
    m3.metric("SD", f"{s['sd']:.2f}")
    m4.metric("Min", f"{s['min']:.2f}")
    m5.metric("Max", f"{s['max']:.2f}")
    m6.metric("Skew", f"{s['skew']:.2f}")

    st.caption("Tip: Strong skew or extreme outliers may inflate SD and widen control limits.")

# ---------------------------
# Export (SPC chart JPEG)
# ---------------------------
st.subheader("Export")
if st.button("Create SPC JPEG export"):
    try:
        img_bytes = fig_spc.to_image(format="jpeg", scale=2)  # requires kaleido
        st.session_state["jpeg_spc"] = img_bytes
        st.success("SPC JPEG created. Use the download button.")
    except Exception as e:
        st.error("JPEG export needs Kaleido. Install it with: pip install -U kaleido")
        st.exception(e)

jpeg = st.session_state.get("jpeg_spc", None)
if jpeg:
    st.download_button(
        label="Download SPC chart JPEG",
        data=jpeg,
        file_name="spc_chart.jpeg",
        mime="image/jpeg",
    )
import io
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

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

# Public NHS logo
NHS_LOGO_URL = "https://www.england.nhs.uk/wp-content/themes/nhsengland/static/img/nhs-logo-blue.svg"

st.set_page_config(page_title="SPC charts", layout="wide")

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
        padding: 8px 10px; border-radius: 999px;# Controls (National vs Provider view)
# Controls (National vs Provider view) + date range
c1, c2, c3, c4, c5 = st.columns([1.1, 2, 2, 1.4, 2.2])
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

st.info(
    "Run-of-8 signal: If 8 consecutive data points fall above or below the mean, "
    "this indicates a sustained shift unlikely due to random variation (special cause)."
)

# ---------------------------
# Mock SPC data + rules
# ---------------------------
np.random.seed(12)

def generate_series(view="National", indicator="Babies who were born preterm (Rate per 1,000)", provider="Airedale NHS Foundation Trust"):
    months = pd.date_range("2024-01-01", "2025-10-01", freq="MS")
    n = len(months)

    base = 62.0 if view == "National" else 60.5
    if view != "National":
        base += (hash(provider) % 7 - 3) * 0.4

    noise = np.random.normal(0, 1.6, n)
    values = np.zeros(n)
    values[0] = base + noise[0]
    for i in range(1, n):
        values[i] = 0.65*values[i-1] + 0.35*(base + noise[i])

    # Inject special-cause patterns
    values[11] += 6.0  # ~Dec 2024 spike
    if view != "National":
        values[18] -= 5.0

    return pd.DataFrame({"Month": months, "Rate": values})

def spc_limits(df):
    mean = df["Rate"].mean()
    sd = df["Rate"].std(ddof=1)
    ucl = mean + 3*sd
    lcl = mean - 3*sd
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
                marks[i-run_len+1:i+1] = True
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

    fig.add_trace(go.Scatter(
        x=in_control["Month"], y=in_control["Rate"],
        mode="lines+markers",
        name="In control",
        line=dict(color=NHS_BLUE, width=2),
        marker=dict(size=7),
        hovertemplate="%{x|%b %Y}<br>Rate: %{y:.2f}<extra></extra>"
    ))

    if len(special) > 0:
        fig.add_trace(go.Scatter(
            x=special["Month"], y=special["Rate"],
            mode="markers",
            name="Special cause",
            marker=dict(size=12, color=BAD, line=dict(width=1, color="#111827")),
            hovertemplate="%{x|%b %Y}<br><b>Special cause</b><br>Rate: %{y:.2f}<extra></extra>"
        ))

    if show_run and len(run_only) > 0:
        fig.add_trace(go.Scatter(
            x=run_only["Month"], y=run_only["Rate"],
            mode="markers",
            name="Run-of-8 signal",
            marker=dict(size=10, symbol="diamond", color=AMBER),
            hovertemplate="%{x|%b %Y}<br><b>Run-of-8</b><br>Rate: %{y:.2f}<extra></extra>"
        ))

    fig.add_hline(y=mean, line_color="#111827", line_width=2,
                  annotation_text="Mean" if show_ann else None,
                  annotation_position="top left")
    fig.add_hline(y=ucl, line_dash="dash", line_color=NHS_GREY, line_width=2,
                  annotation_text="UCL" if show_ann else None,
                  annotation_position="top left")
    fig.add_hline(y=lcl, line_dash="dash", line_color=NHS_GREY, line_width=2,
                  annotation_text="LCL" if show_ann else None,
                  annotation_position="bottom left")

    fig.update_layout(
        height=520,
        margin=dict(l=20, r=20, t=10, b=10),
        xaxis_title=None, yaxis_title=None,
        plot_bgcolor="white", paper_bgcolor="white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        font=dict(family="Inter, Arial")
    )
    return fig, df

# ---------------------------
# Header with NHS logo
# ---------------------------
st.markdown(
    f"""
    <div class="nhs-header">
      <div>
        <p class="nhs-title">Statistical Process Control (SPC) charts</p>
      </div>
      <div>
        <img src="{NHS_LOGO_URL}" alt="NHS logo" style="height:42px;">
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------
# Tab context (overall dashboard)
# ---------------------------
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

# ---------------------------
# Page description ("why used") + Guidance link placeholder
# ---------------------------
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

# ---------------------------
# Controls (National vs Provider view)
# ---------------------------
c1, c2, c3, c4 = st.columns([1.1, 2, 2, 1.4])

with c1:
    view = st.radio("", ["National SPC charts view", "Provider SPC charts view"], index=0)

with c2:
    indicator = st.selectbox(
        "Select indicator",
        [
            "Babies who were born preterm (Rate per 1,000)",
            "Smoking at time of delivery (Rate per 100)",
            "Apgar <7 at 5 minutes (Rate per 1,000)"
        ],
        index=0,
    )

with c3:
    provider = st.selectbox(
        "Select provider",
        [
            "Airedale NHS Foundation Trust",
            "Bradford Teaching Hospitals NHS Foundation Trust",
            "Leeds Teaching Hospitals NHS Trust",
            "York and Scarborough Teaching Hospitals NHS Foundation Trust"
        ],
        index=0,
        disabled=view.startswith("National")
    )

with c4:
    show_ann = st.checkbox("Show Mean/UCL/LCL", value=True)
    show_run = st.checkbox("Highlight run-of-8", value=True)
    st.caption("Run-of-8 signal: 8 consecutive points above or below the mean suggests a non-random shift (special cause).")

with c5:
    # Calendar pickers (we'll treat selected dates as Month/Year and normalise to month start)
    start_date = st.date_input("Start date", value=pd.Timestamp("2024-01-01").date())
    end_date   = st.date_input("End date", value=pd.Timestamp("2025-10-01").date())

# Help links toolbar
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
# Build chart + KPIs
# ---------------------------
# Build chart + KPIs
v = "National" if view.startswith("National") else "Provider"
df = generate_series(view=v, indicator=indicator, provider=provider)

# Normalise selected dates to the first day of their month (month/year behaviour)
start_m = pd.Timestamp(start_date).to_period("M").to_timestamp()
end_m   = pd.Timestamp(end_date).to_period("M").to_timestamp()

# Guard: ensure start <= end
if start_m > end_m:
    st.error("Start date must be earlier than or equal to End date.")
    st.stop()

# Filter to selected month range
df = df[(df["Month"] >= start_m) & (df["Month"] <= end_m)].copy()

# Guard: avoid empty selection
if df.empty:
    st.warning("No data available for the selected date range.")
    st.stop()

mean, ucl, lcl = spc_limits(df)
fig, df_flagged = build_spc_figure(df, mean, ucl, lcl, show_ann=show_ann, show_run=show_run)

series_label = f"{df_flagged['Month'].iloc[0].strftime('%b %Y')} – {df_flagged['Month'].iloc[-1].strftime('%b %Y')}"
current_rate = float(df_flagged["Rate"].iloc[-1])
any_special = bool(df_flagged["SpecialCause"].any())
current_special = bool(df_flagged["SpecialCause"].iloc[-1])

# KPI strip
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

# Chart
st.plotly_chart(fig, use_container_width=True)

# ---------------------------
# Export chart as JPEG (download)
# ---------------------------
st.subheader("Export")
colA, colB = st.columns([1, 2])

with colA:
    if st.button("Create JPEG export"):
        img_bytes = fig.to_image(format="jpeg", scale=2)  # requires kaleido
        st.session_state["jpeg"] = img_bytes
        st.success("JPEG created. Use the download button.")

with colB:
    jpeg = st.session_state.get("jpeg", None)
    if jpeg:
        st.download_button(
            label="Download chart JPEG",
            data=jpeg,
            file_name="spc_chart.jpeg",
            mime="image/jpeg"
        )

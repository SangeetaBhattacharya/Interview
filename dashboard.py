import numpy as np
import pandas as pd
from datetime import datetime
from dash import Dash, dcc, html, Input, Output
import plotly.graph_objs as go

# ------------------------------------------------------------
# Sample data (replace with your real dataset)
# ------------------------------------------------------------

dates = pd.date_range("2024-01-01", periods=24, freq="MS")
values = np.array([
    62.0,63.5,62.8,61.9,64.2,60.5,62.1,61.0,60.9,58.2,61.1,61.0,
    67.2,62.0,61.9,62.6,62.7,62.5,62.4,61.3,61.6,61.8,60.2,63.0
])

df = pd.DataFrame({"Date": dates, "Rate": values})

# ------------------------------------------------------------
# SPC calculation helper
# ------------------------------------------------------------

def compute_spc(df):
    df = df.copy()
    df["MR"] = df["Rate"].diff().abs()
    MR_bar = df["MR"][1:].mean()
    sigma = MR_bar / 1.128  # XmR sigma estimate

    CL = df["Rate"].mean()
    UCL = CL + 3*sigma
    LCL = CL - 3*sigma

    # Rule 1: outside 3 sigma
    df["Rule1"] = (df["Rate"] > UCL) | (df["Rate"] < LCL)

    # Rule 2: 8 consecutive points on same side of CL
    signs = np.sign(df["Rate"] - CL)
    run_flags = np.zeros(len(df), dtype=bool)
    run_len = 8
    for i in range(len(df)):
        if i >= run_len-1:
            window = signs[i-run_len+1:i+1]
            if all(window > 0) or all(window < 0):
                run_flags[i-run_len+1:i+1] = True

    df["Rule_Run8"] = run_flags
    df["Special"] = df[["Rule1","Rule_Run8"]].any(axis=1)

    return df, CL, UCL, LCL, sigma

# ------------------------------------------------------------
# Build dashboard
# ------------------------------------------------------------

app = Dash(__name__)

app.layout = html.Div([
    html.H1("Statistical Process Control (SPC) Dashboard",
            style={"color":"#212B32", "font-family":"Segoe UI"}),

    html.Div([
        html.Label("Select Indicator:",
                   style={"font-weight":"bold","color":"#212B32"}),
        dcc.Dropdown(
            id="indicator",
            value="Babies born preterm",
            options=[{"label":"Babies born preterm (per 1,000)", 
                      "value":"Babies born preterm"}],
            style={"width":"50%"}
        )
    ], style={"padding":"10px 0"}),

    html.Div(id="narrative", style={"padding":"10px 0", "font-size":"16px"}),

    dcc.Graph(id="spc_chart")
])

# ------------------------------------------------------------
# Callbacks
# ------------------------------------------------------------

@app.callback(
    [Output("spc_chart","figure"),
     Output("narrative","children")],
    [Input("indicator","value")]
)
def update_chart(ind):

    df_spc, CL, UCL, LCL, sigma = compute_spc(df)

    # NHS palette
    NHS_BLUE = "#005EB8"
    ERROR = "#D5281B"
    MUTED = "#8DA3B0"
    TEXT = "#212B32"

    # Narrative
    special_count = df_spc["Special"].sum()
    if special_count == 0:
        narrative = "No special cause variation detected; all points within expected limits."
    else:
        first_special = df_spc[df_spc["Special"]].iloc[0]["Date"]
        narrative = f"{special_count} special-cause point(s) detected. First occurred in {first_special:%b %Y}."

    # Build figure
    fig = go.Figure()

    # Common-cause points
    fig.add_trace(go.Scatter(
        x=df_spc["Date"],
        y=df_spc["Rate"],
        mode="lines+markers",
        name="Rate",
        marker=dict(size=7, color=MUTED),
        line=dict(color=MUTED, width=2)
    ))

    # Special-cause overlay
    fig.add_trace(go.Scatter(
        x=df_spc[df_spc["Special"]]["Date"],
        y=df_spc[df_spc["Special"]]["Rate"],
        mode="markers",
        name="Special Cause",
        marker=dict(size=10, color=ERROR)
    ))

    # CL, UCL, LCL
    fig.add_hline(y=CL, line_color=NHS_BLUE, line_width=2, annotation_text="CL")
    fig.add_hline(y=UCL, line_color=NHS_BLUE, line_width=1.5, line_dash="dash",
                  annotation_text="UCL")
    fig.add_hline(y=LCL, line_color=NHS_BLUE, line_width=1.5, line_dash="dash",
                  annotation_text="LCL")

    fig.update_layout(
        template="simple_white",
        height=500,
        showlegend=True,
        margin=dict(l=30,r=30,t=30,b=30),
        xaxis_title="Date",
        yaxis_title="Rate per 1,000",
        font=dict(color=TEXT, family="Segoe UI"),
        hovermode="x"
    )

    return fig, narrative

# ------------------------------------------------------------
# Run dashboard
# ------------------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True)

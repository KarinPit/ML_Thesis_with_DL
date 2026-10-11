"""
explore_data.py
---------------
Interactive dashboard for exploring jones_israel_all_years.nc.

Features:
  - Variable selector dropdown
  - Timestep slider
  - Map of the selected variable at the selected timestep
  - Stats panel: mean, std, median, NaN count, min, max
  - For ltg: also shows % of grid cells with lightning

Usage (on EC2 or locally):
    pip install plotly dash
    python explore_data.py
    Then open http://localhost:8050 in your browser
"""

import xarray as xr
import numpy as np
import pandas as pd
import os

import dash
from dash import dcc, html, Input, Output, callback
import plotly.graph_objects as go
import plotly.express as px

# ── Country borders (via cartopy Natural Earth data) ───────────────────────────
def _load_country_borders():
    try:
        import cartopy.io.shapereader as shpreader
        shpfile = shpreader.natural_earth(resolution='110m', category='cultural',
                                          name='admin_0_countries')
        reader  = shpreader.Reader(shpfile)
        lons, lats = [], []
        for record in reader.records():
            geom = record.geometry
            polys = [geom] if geom.geom_type == 'Polygon' else list(geom.geoms)
            for poly in polys:
                xs, ys = poly.exterior.xy
                lons.extend(list(xs) + [None])
                lats.extend(list(ys) + [None])
        print("Country borders loaded from cartopy.", flush=True)
        return lons, lats
    except Exception as e:
        print(f"cartopy borders unavailable ({e}) — install with: pip install cartopy", flush=True)
        return [], []

BORDER_LONS, BORDER_LATS = _load_country_borders()

# ── Load data ──────────────────────────────────────────────────────────────────
DATA_PATH = '/home/ec2-user/ML_Thesis_with_DL/data/jones_israel_all_years.nc'
if not os.path.exists(DATA_PATH):
    DATA_PATH = 'data/jones_israel_all_years.nc'

print("Loading dataset...", flush=True)
ds = xr.open_dataset(DATA_PATH)

# Resample to 12-hourly to keep things manageable
feature_vars = ['cape', 'precipitation', 'lsm', 'rh', 'shear', 't2m', 'wcd', 'vertical_velocity_500hPa']
ds_feat = ds[feature_vars].resample(time='12h').mean()
ds_ltg  = ds[['ltg']].resample(time='12h').sum()
ds_12h  = xr.merge([ds_feat, ds_ltg])

# resample() fills gaps between years with NaN rows — drop them by keeping only Sep-Dec
season_mask = ds_12h.time.dt.month.isin([9, 10, 11, 12])
ds_12h = ds_12h.isel(time=season_mask.values)

all_vars  = list(ds_12h.data_vars)
all_times = pd.DatetimeIndex(ds_12h.time.values)
lats = ds_12h.lat.values
lons = ds_12h.lon.values

print(f"Ready: {len(all_times)} timesteps, {len(all_vars)} variables", flush=True)
print(f"Months in dataset: {sorted(all_times.month.unique().tolist())}", flush=True)

# ── App ────────────────────────────────────────────────────────────────────────
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H2("Lightning Domain Explorer", style={'textAlign': 'center', 'fontFamily': 'Arial'}),

    # Controls row
    html.Div([
        html.Div([
            html.Label("Variable", style={'fontWeight': 'bold'}),
            dcc.Dropdown(
                id='var-dropdown',
                options=[{'label': v, 'value': v} for v in all_vars],
                value='ltg',
                clearable=False,
                style={'width': '250px'}
            ),
        ], style={'marginRight': '40px'}),

        html.Div([
            html.Label("Timestep", style={'fontWeight': 'bold'}),
            html.Div([
                dcc.Slider(
                    id='time-slider',
                    min=0,
                    max=len(all_times) - 1,
                    step=1,
                    value=0,
                    marks={i: str(all_times[i].date()) for i in range(0, len(all_times), max(1, len(all_times)//10))},
                    tooltip={"placement": "bottom", "always_visible": True},
                ),
            ], style={'width': '700px'}),
        ]),
    ], style={'display': 'flex', 'alignItems': 'center', 'padding': '10px 20px',
              'flexWrap': 'wrap', 'fontFamily': 'Arial'}),

    # Timestamp display
    html.Div(id='time-display', style={'textAlign': 'center', 'fontSize': '16px',
                                        'fontFamily': 'Arial', 'marginBottom': '5px'}),

    # Map
    dcc.Graph(id='map-plot', style={'height': '520px'}),

    # Stats panel
    html.Div(id='stats-panel', style={
        'display': 'flex', 'flexWrap': 'wrap', 'justifyContent': 'center',
        'gap': '20px', 'padding': '10px 20px', 'fontFamily': 'Arial',
        'backgroundColor': '#f4f6f9', 'borderTop': '1px solid #ddd'
    }),
], style={'maxWidth': '1200px', 'margin': '0 auto'})


@app.callback(
    Output('map-plot',    'figure'),
    Output('stats-panel', 'children'),
    Output('time-display','children'),
    Input('var-dropdown', 'value'),
    Input('time-slider',  'value'),
)
def update(var, t_idx):
    ts  = all_times[t_idx]
    arr = ds_12h[var].isel(time=t_idx).values.copy()   # (lat, lon)
    arr = np.nan_to_num(arr, nan=0.0)   # replace NaNs with 0 for display

    # ── Stats ──────────────────────────────────────────────────────────────────
    flat     = arr.ravel()
    nan_mask = np.isnan(flat)
    valid    = flat[~nan_mask]

    mean_v   = float(np.nanmean(arr))   if valid.size > 0 else float('nan')
    std_v    = float(np.nanstd(arr))    if valid.size > 0 else float('nan')
    median_v = float(np.nanmedian(arr)) if valid.size > 0 else float('nan')
    min_v    = float(np.nanmin(arr))    if valid.size > 0 else float('nan')
    max_v    = float(np.nanmax(arr))    if valid.size > 0 else float('nan')
    nan_count = int(nan_mask.sum())

    stat_items = [
        ('Mean',   f'{mean_v:.4f}'),
        ('Std',    f'{std_v:.4f}'),
        ('Median', f'{median_v:.4f}'),
        ('Min',    f'{min_v:.4f}'),
        ('Max',    f'{max_v:.4f}'),
        ('NaNs',   str(nan_count)),
    ]

    # Lightning-specific: % of grid cells with lightning
    if var == 'ltg':
        pct_ltg = 100.0 * float((arr > 0).sum()) / arr.size
        stat_items.append(('% cells with lightning', f'{pct_ltg:.2f}%'))

    stat_boxes = []
    for label, val in stat_items:
        stat_boxes.append(html.Div([
            html.Div(label, style={'fontSize': '12px', 'color': '#666'}),
            html.Div(val,   style={'fontSize': '18px', 'fontWeight': 'bold', 'color': '#2c3e50'}),
        ], style={
            'backgroundColor': 'white', 'borderRadius': '8px',
            'padding': '10px 20px', 'boxShadow': '0 1px 4px rgba(0,0,0,0.1)',
            'minWidth': '130px', 'textAlign': 'center'
        }))

    # ── Map ────────────────────────────────────────────────────────────────────
    colorscale = 'YlOrRd' if var == 'ltg' else 'RdBu_r'

    fig = go.Figure(go.Heatmap(
        z=arr,
        x=lons,
        y=lats,
        colorscale=colorscale,
        colorbar=dict(title=var),
        hoverongaps=False,
        hovertemplate='lon: %{x:.2f}°<br>lat: %{y:.2f}°<br>value: %{z:.4f}<extra></extra>',
    ))

    # Overlay country borders
    if BORDER_LONS:
        fig.add_trace(go.Scatter(
            x=BORDER_LONS, y=BORDER_LATS,
            mode='lines',
            line=dict(color='black', width=0.8),
            hoverinfo='skip',
            showlegend=False,
        ))

    fig.update_layout(
        title=f'{var}  —  {ts.strftime("%Y-%m-%d %H:%M UTC")}',
        xaxis_title='Longitude',
        yaxis_title='Latitude',
        margin=dict(l=50, r=20, t=50, b=40),
        plot_bgcolor='#cce5ff',
        paper_bgcolor='white',
        font=dict(family='Arial'),
        xaxis=dict(range=[float(lons.min()), float(lons.max())], constrain='domain'),
        yaxis=dict(range=[float(lats.min()), float(lats.max())], scaleanchor='x', scaleratio=1),
    )

    time_label = f"Timestep {t_idx + 1} / {len(all_times)}  —  {ts.strftime('%A, %d %B %Y  %H:%M UTC')}"

    return fig, stat_boxes, time_label


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=8050)

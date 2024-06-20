import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import io
import base64
from datetime import timedelta
from sklearn.metrics import mean_squared_error
from dash import dcc, html, Input, Output, State
import dash
import dash_bootstrap_components as dbc
import plotly.express as px

# Load the merged dataset
merged_dataset_path = 'data/merged_final_dataset.csv'
merged_df = pd.read_csv(merged_dataset_path)

# Load the datasets for other features
file_path = 'data/final_dataset.csv'
df = pd.read_csv(file_path)

# Convert date columns to datetime
df['order_purchase_timestamp'] = pd.to_datetime(df['order_purchase_timestamp'])
df['order_delivered_customer_date'] = pd.to_datetime(df['order_delivered_customer_date'], errors='coerce')

# Calculate delivery time in days
df['delivery_time'] = (df['order_delivered_customer_date'] - df['order_purchase_timestamp']).dt.days

# Filter out invalid data
df = df[df['delivery_time'].notna() & df['review_score'].notna()]

# Calculate average delivery time and rating by state
state_summary = df.groupby('customer_state').agg({'delivery_time': 'mean', 'review_score': 'mean'}).reset_index()

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Sidebar layout
sidebar = html.Div(
    [
        html.H2("Olist Consulting", className="display-6"),
        html.Hr(),
        dbc.Nav(
            [
                dbc.NavLink("Demand Forecast", href="/", active="exact"),
                dbc.NavLink("Rating and Delivery Time", href="/rating-delivery", active="exact"),
                dbc.NavLink("Seller Analysis", href="/seller-analysis", active="exact"),
            ],
            vertical=True,
            pills=True,
        ),
        html.Img(src="/assets/images/olist_logo.png", style={"width": "100%", "height": "auto", "padding-top": "12.5rem"}),  # Nueva imagen
    ],
    style={
        "position": "fixed",
        "top": 0,
        "left": 0,
        "bottom": 0,
        "width": "16rem",
        "padding": "2rem 1rem",
        "background-color": "#e0f2f1",
    },
)

top_bar = html.Div(
    [
        html.Div(
            [
                html.Img(
                    src="/assets/images/olist_logo.png",
                    style={"width": "8%", "height": "auto"},
                )
            ],
            style={
                "display": "flex",
                "justify-content": "flex-end",
                "align-items": "flex-start",
                "width": "98%",
                "padding-top": "1rem",
            }
        ),
    ],
    style={
        "height": "4rem",
        "background-color": "#e0f2f1",
        "margin-left": "0rem",
        "position": "fixed",
        "top": 0,
        "right": 0,
        "left": "16rem",
        "z-index": 100,
    }
)

# Demand Forecast Analysis Layout
state_dropdown = dcc.Dropdown(
    id='state-dropdown',
    options=[{'label': state, 'value': state} for state in df['customer_state'].unique()],
    placeholder="Select a customer state",
)

category_dropdown = dcc.Dropdown(
    id='category-dropdown',
    options=[{'label': category, 'value': category} for category in df['product_category_name_english'].unique()],
    placeholder="Select a product category",
)

forecast_option = dcc.RadioItems(
    id='forecast-option',
    options=[
        {'label': 'Only by Category', 'value': 'category'},
        {'label': 'Only by State', 'value': 'state'},
        {'label': 'By Both State and Category', 'value': 'both'}
    ],
    value='state'
)

go_button = html.Button('Go', id='go-button', n_clicks=0)

analysis_content = html.Div(id='analysis-content')

# Layout for Demand Forecast Analysis
analysis_content_layout = html.Div(
    [
        html.H1("Demand Forecast Analysis"),
        state_dropdown,
        category_dropdown,
        forecast_option,
        go_button,
        analysis_content
    ],
    id="analysis-content-layout",
    style={
        "margin-left": "8rem",
        "margin-top": "2rem",
        "padding": "2rem",
    },
)

# Rating and Delivery Time Analysis Layout
rating_delivery_layout = html.Div(
    [
        html.H1("Rating and Delivery Time Analysis"),
        dcc.Dropdown(
            id='metric-dropdown',
            options=[
                {'label': 'Delivery Time', 'value': 'delivery_time'},
                {'label': 'Rating', 'value': 'review_score'}
            ],
            value='delivery_time',
            clearable=False
        ),
        dcc.Graph(id='choropleth-map', style={'height': '800px', 'width': '100%'})
    ],
    id="rating-delivery-layout",
    style={
        "margin-left": "8rem",
        "margin-top": "1rem",
        "padding": "2rem",
    },
)

# Seller Analysis Layout
seller_analysis_layout = html.Div(
    [
        html.H1("Seller Analysis"),
        dcc.Dropdown(
            id='seller-metric-dropdown',
            options=[
                {'label': 'Revenue', 'value': 'revenue_final'},
                {'label': 'Delivery Time', 'value': 'delivery_time_final'},
                {'label': 'Rating', 'value': 'avg_rating'}
            ],
            placeholder="Select a metric to filter",
            clearable=True
        ),
        dcc.Dropdown(
            id='state-filter-dropdown',
            options=[{'label': state, 'value': state} for state in merged_df['customer_state_summary'].unique()],
            placeholder="Select a customer state",
            clearable=True
        ),
        dcc.Dropdown(
            id='category-filter-dropdown',
            options=[{'label': category, 'value': category} for category in merged_df['product_category_name_english_summary'].unique()],
            placeholder="Select a product category",
            clearable=True
        ),
        html.Button('Go', id='seller-go-button', n_clicks=0),
        dcc.Graph(id='seller-scatter-plot', style={'height': '800px', 'width': '100%'}),
        html.H2("Top 5 Sellers Ranking"),
        html.Div(id='top-sellers-ranking')
    ],
    id="seller-analysis-layout",
    style={
        "margin-left": "8rem",
        "margin-top": "1rem",
        "padding": "2rem",
    },
)

app.layout = html.Div(
    id="main-content", 
    style={
        "background-image": "url(https://blogger.googleusercontent.com/img/b/R29vZ2xl/AVvXsEiPgEUDujOorzousPfNQLmysmvM9gujOCdlN4z3r8N1Ylr837t7X1jvTUbDDOSvVIVD2hypxxx-R86qaRD7bGUOVAFlKYOjHzX9kMnOiIAkZc_9v76pJQcKEN6vEr1akh44PvkEFiFiIxMJ8fgBXVoychqJyFfNkc44jjgpMEO1lihWpRJ3un_jlHMJDpvd/s1000/background.jpg)",
        "background-size": "contain",  # La imagen se ajustará al contenedor manteniendo su relación de aspecto
        "background-repeat": "no-repeat",
        "background-position": "right",
        "opacity": 0.9,
    },
    children=[
        dcc.Location(id="url"),
        sidebar,
        top_bar,
        html.Div(id="page-content", style={
            "margin-left": "1rem",
            "padding": "9rem",
        })
    ]
)

# Callbacks for URL Routing and Scatter Plot
@app.callback(
    Output("page-content", "children"),
    [Input("url", "pathname")]
)
def display_page(pathname):
    if pathname == "/rating-delivery":
        return rating_delivery_layout
    elif pathname == "/seller-analysis":
        return seller_analysis_layout
    else:
        return analysis_content_layout

# Callback for Forecast Analysis
@app.callback(
    Output('analysis-content', 'children'),
    Input('go-button', 'n_clicks'),
    State('state-dropdown', 'value'),
    State('category-dropdown', 'value'),
    State('forecast-option', 'value')
)
def update_analysis_content(n_clicks, state_value, category_value, forecast_option_value):
    if n_clicks > 0:
        # Implement your forecast logic here
        if forecast_option_value == 'state':
            return html.Div(f"Forecasting demand for {state_value}")
        elif forecast_option_value == 'category':
            return html.Div(f"Forecasting demand for category: {category_value}")
        elif forecast_option_value == 'both':
            return html.Div(f"Forecasting demand for {state_value} and category: {category_value}")
    return html.Div()

# Callback for Rating and Delivery Time Analysis
@app.callback(
    Output('choropleth-map', 'figure'),
    Input('metric-dropdown', 'value')
)
def update_choropleth(metric_value):
    if metric_value == 'delivery_time':
        fig = px.choropleth(state_summary, locations='customer_state', locationmode='country names',
                            color='delivery_time', hover_name='customer_state',
                            title='Average Delivery Time by State')
    elif metric_value == 'review_score':
        fig = px.choropleth(state_summary, locations='customer_state', locationmode='country names',
                            color='review_score', hover_name='customer_state',
                            title='Average Rating by State')

    fig.update_geos(showcoastlines=True, coastlinecolor="DarkBlue", showland=True, landcolor="LightGreen",
                    showocean=True, oceancolor="LightBlue", showlakes=True, lakecolor="Blue")
    return fig

# Callback for Seller Analysis Scatter Plot
@app.callback(
    Output('seller-scatter-plot', 'figure'),
    Output('top-sellers-ranking', 'children'),
    Input('seller-go-button', 'n_clicks'),
    State('seller-metric-dropdown', 'value'),
    State('state-filter-dropdown', 'value'),
    State('category-filter-dropdown', 'value')
)
def update_seller_scatter_plot(n_clicks, metric_value, state_value, category_value):
    if n_clicks > 0:
        # Implement your seller analysis logic here
        if metric_value == 'revenue_final':
            # Example logic, replace with actual logic for seller revenue analysis
            fig = px.scatter(merged_df, x='seller_id', y='revenue_final', color='revenue_final',
                             title='Seller Revenue Analysis')
            top_sellers = merged_df.sort_values(by='revenue_final', ascending=False).head(5)[['seller_id', 'revenue_final']]
            top_sellers_ranking = html.Table(
                [
                    html.Thead(html.Tr([html.Th("Rank"), html.Th("Seller ID"), html.Th("Revenue")]))
                ] +
                [
                    html.Tr([html.Td(i + 1), html.Td(row['seller_id']), html.Td(row['revenue_final'])])
                    for i, row in top_sellers.iterrows()
                ]
            )
        elif metric_value == 'delivery_time_final':
            # Example logic, replace with actual logic for seller delivery time analysis
            fig = px.scatter(merged_df, x='seller_id', y='delivery_time_final', color='delivery_time_final',
                             title='Seller Delivery Time Analysis')
            top_sellers = merged_df.sort_values(by='delivery_time_final', ascending=True).head(5)[['seller_id', 'delivery_time_final']]
            top_sellers_ranking = html.Table(
                [
                    html.Thead(html.Tr([html.Th("Rank"), html.Th("Seller ID"), html.Th("Delivery Time")]))
                ] +
                [
                    html.Tr([html.Td(i + 1), html.Td(row['seller_id']), html.Td(row['delivery_time_final'])])
                    for i, row in top_sellers.iterrows()
                ]
            )
        elif metric_value == 'avg_rating':
            # Example logic, replace with actual logic for seller rating analysis
            fig = px.scatter(merged_df, x='seller_id', y='avg_rating', color='avg_rating',
                             title='Seller Rating Analysis')
            top_sellers = merged_df.sort_values(by='avg_rating', ascending=False).head(5)[['seller_id', 'avg_rating']]
            top_sellers_ranking = html.Table(
                [
                    html.Thead(html.Tr([html.Th("Rank"), html.Th("Seller ID"), html.Th("Avg Rating")]))
                ] +
                [
                    html.Tr([html.Td(i + 1), html.Td(row['seller_id']), html.Td(row['avg_rating'])])
                    for i, row in top_sellers.iterrows()
                ]
            )
        else:
            fig = px.scatter()
            top_sellers_ranking = html.Div()

        return fig, top_sellers_ranking
    return px.scatter(), html.Div()

if __name__ == '__main__':
    app.run_server(debug=True)

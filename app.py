import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import io
import base64
from datetime import timedelta
from sklearn.metrics import mean_squared_error
import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output, State
import plotly.express as px

# Load the merged dataset
merged_dataset_path = 'data/merged_final_dataset_cleaned.csv'
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

# Load additional datasets for the new feature
closed_deals_path = 'data/olist_closed_deals_dataset.csv'
qualified_leads_path = 'data/olist_marketing_qualified_leads_dataset.csv'
closed_deals = pd.read_csv(closed_deals_path)
qualified_leads = pd.read_csv(qualified_leads_path)

# Calculate the number of unique sellers per product category
unique_sellers_per_category = df.groupby('product_category_name_english')['seller_id'].nunique().reset_index()
unique_sellers_per_category.rename(columns={'seller_id': 'unique_sellers_count'}, inplace=True)

# Calculate the total number of sales per product category
total_sales_per_category = df.groupby('product_category_name_english')['order_id'].count().reset_index()
total_sales_per_category.rename(columns={'order_id': 'total_sales'}, inplace=True)

# Merge the two dataframes
category_analysis = pd.merge(unique_sellers_per_category, total_sales_per_category, on='product_category_name_english')

# Calculate the average sales per seller per product category
category_analysis['avg_sales_per_seller'] = category_analysis['total_sales'] / category_analysis['unique_sellers_count']

# Identify categories with high average sales per seller
median_avg_sales_per_seller = category_analysis['avg_sales_per_seller'].median()
high_power_threshold = median_avg_sales_per_seller * 1.5

high_power_categories = category_analysis[category_analysis['avg_sales_per_seller'] > high_power_threshold]

# Merge the datasets on 'mql_id' to identify which qualified leads became closed deals
merged_data = pd.merge(closed_deals, qualified_leads, on='mql_id', how='inner')

# Group by 'origin' and 'business_segment' and count the number of closed deals for each combination
closed_deals_by_origin_segment = merged_data.groupby(['origin', 'business_segment'])['mql_id'].count().reset_index()
closed_deals_by_origin_segment.rename(columns={'mql_id': 'closed_deals_count'}, inplace=True)

# Count the total number of qualified leads for each origin
qualified_leads_by_origin = qualified_leads.groupby('origin')['mql_id'].count().reset_index()
qualified_leads_by_origin.rename(columns={'mql_id': 'qualified_leads_count'}, inplace=True)

# Merge the two dataframes to get both closed deals count and qualified leads count
conversion_data = pd.merge(closed_deals_by_origin_segment, qualified_leads_by_origin, on='origin', how='inner')

# Calculate the conversion rate for each origin and business segment
conversion_data['conversion_rate'] = conversion_data['closed_deals_count'] / conversion_data['qualified_leads_count']

# Initialize the Dash app
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
                dbc.NavLink("Seller Power and Conversion Rates", href="/seller-power-conversion", active="exact"),
            ],
            vertical=True,
            pills=True,
        ),
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

# Top bar layout
top_bar = html.Div(
    [
        html.Div(
            [
                html.Img(
                    src="assets/logo.png",
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
        "margin-top": "1rem",
        "padding": "1rem",
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
        "padding": "1rem",
    },
)

# Seller Analysis Layout
seller_analysis_layout = html.Div(
    [
        html.H1("Seller Analysis"),
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
        dcc.Dropdown(
            id='ranking-filter-dropdown',
            options=[
                {'label': 'Top 10 Best Sellers', 'value': 'best'},
                {'label': 'Top 10 Worst Sellers', 'value': 'worst'}
            ],
            placeholder="Select ranking filter",
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
        "padding": "1rem",
    },
)

# Seller Power and Conversion Rates Layout
seller_power_conversion_layout = html.Div(
    [
        html.H1("Seller Power and Conversion Rates"),
        dcc.Dropdown(
            id='num-top-categories-dropdown',
            options=[
                {'label': 'Top 5', 'value': 5},
                {'label': 'Top 10', 'value': 10},
                {'label': 'Top 15', 'value': 15},
                {'label': 'Top 20', 'value': 20}
            ],
            value=10,
            clearable=False
        ),
        dcc.Dropdown(
            id='segment-dropdown',
            options=[{'label': segment, 'value': segment} for segment in conversion_data['business_segment'].unique()],
            placeholder="Select a business segment",
            clearable=True
        ),
        html.Button('Analyze', id='analyze-button', n_clicks=0),
        dcc.Graph(id='top-categories-plot', style={'height': '800px', 'width': '100%'}),
        dcc.Graph(id='conversion-rates-plot', style={'height': '800px', 'width': '100%'})
    ],
    id="seller-power-conversion-layout",
    style={
        "margin-left": "8rem",
        "margin-top": "1rem",
        "padding": "1rem",
    },
)

# App layout
app.layout = html.Div(
    [
        dcc.Location(id="url"),
        sidebar,
        html.Div(id="page-content", style={"margin-left": "8rem", "padding": "1rem"})
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
    elif pathname == "/seller-power-conversion":
        return seller_power_conversion_layout
    return analysis_content_layout

# Callback for Demand Forecast Analysis
def prepare_data(data, selection_type, customer_state=None, product_category=None):
    if selection_type == 'state':
        df = data[data['customer_state'] == customer_state].copy()
    elif selection_type == 'category':
        df = data[data['product_category_name_english'] == product_category].copy()
    elif selection_type == 'both':
        df = data[(data['customer_state'] == customer_state) & (data['product_category_name_english'] == product_category)].copy()
    else:
        raise ValueError("Invalid selection_type. Choose from 'state', 'category', or 'both'.")
    
    df = df.set_index('order_purchase_timestamp').resample('D').size().reset_index(name='demand')
    return df

def analyze_orders(selection_type, state=None, category=None):
    df['order_purchase_timestamp'] = pd.to_datetime(df['order_purchase_timestamp'])
    cutoff_date = pd.to_datetime('2018-07-31')
    df_filtered = df[df['order_purchase_timestamp'] <= cutoff_date]

    prepared_df = prepare_data(df_filtered, selection_type, state, category)
    prepared_df = prepared_df.sort_values('order_purchase_timestamp')

    train = prepared_df.iloc[:-21].copy()
    test = prepared_df.iloc[-21:].copy()

    def create_features(df):
        df = df.copy()
        df['day_of_week'] = df['order_purchase_timestamp'].dt.dayofweek
        df['day_of_month'] = df['order_purchase_timestamp'].dt.day
        df['week_of_year'] = df['order_purchase_timestamp'].dt.isocalendar().week
        df['month'] = df['order_purchase_timestamp'].dt.month
        return df

    train = create_features(train)
    test = create_features(test)

    X_train = train.drop(['order_purchase_timestamp', 'demand'], axis=1)
    y_train = train['demand']
    X_test = test.drop(['order_purchase_timestamp', 'demand'], axis=1)
    y_test = test['demand']

    model = xgb.XGBRegressor(objective='reg:squarederror')
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    def calculate_intervals(predictions, alpha=0.05):
        errors = y_train - model.predict(X_train)
        error_std = np.std(errors)
        interval_range = error_std * 1.96
        lower_bounds = predictions - interval_range
        upper_bounds = predictions + interval_range
        return lower_bounds, upper_bounds

    lower_bounds, upper_bounds = calculate_intervals(preds)

    rmse = np.sqrt(mean_squared_error(y_test, preds))
    print(f'Root Mean Square Error (RMSE): {rmse}')

    start_date = test['order_purchase_timestamp'].min() - timedelta(days=30)
    plot_data = prepared_df[(prepared_df['order_purchase_timestamp'] >= start_date) & (prepared_df['order_purchase_timestamp'] <= test['order_purchase_timestamp'].max())]

    plt.figure(figsize=(14, 7))
    plt.plot(plot_data['order_purchase_timestamp'], plot_data['demand'], label='Historical')
    plt.plot(test['order_purchase_timestamp'], y_test, label='Test')
    plt.plot(test['order_purchase_timestamp'], preds, label='Forecast')
    plt.fill_between(test['order_purchase_timestamp'], lower_bounds, upper_bounds, color='gray', alpha=0.2, label='95% Prediction Interval')
    plt.legend()

    # Save plot to a BytesIO object
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plot_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()

    results = test[['order_purchase_timestamp']].copy()
    results['forecast'] = preds
    results['lower_bound'] = lower_bounds
    results['upper_bound'] = upper_bounds
    results_filtered = results[results['order_purchase_timestamp'] >= start_date]

    return plot_base64, rmse, results_filtered

@app.callback(
    Output('analysis-content', 'children'),
    [Input('go-button', 'n_clicks')],
    [State('state-dropdown', 'value'), State('category-dropdown', 'value'), State('forecast-option', 'value')]
)
def update_analysis(n_clicks, state, category, forecast_option):
    if n_clicks > 0:
        plot_base64, rmse, forecast_comparison = analyze_orders(forecast_option, state, category)

        plot_img = html.Img(src=f'data:image/png;base64,{plot_base64}', style={'width': '100%'})
        rmse_text = html.P(f"Root Mean Square Error (RMSE): {rmse}")
        forecast_table = dbc.Table.from_dataframe(forecast_comparison, striped=True, bordered=True, hover=True)

        return [plot_img, rmse_text, forecast_table]
    return html.P("Select options and click 'Go' to run the analysis.")

# Callback for Rating and Delivery Time Analysis
@app.callback(
    Output('choropleth-map', 'figure'),
    [Input('metric-dropdown', 'value')]
)
def update_choropleth(selected_metric):
    if selected_metric == 'delivery_time':
        color_scale = 'Reds'
        color_label = 'Avg Delivery Time (days)'
    else:
        color_scale = 'Blues'
        color_label = 'Avg Rating'
    
    fig = px.choropleth(
        state_summary,
        geojson="https://raw.githubusercontent.com/codeforamerica/click_that_hood/master/public/data/brazil-states.geojson",
        locations='customer_state',
        featureidkey="properties.sigla",
        hover_name='customer_state',
        color=selected_metric,
        color_continuous_scale=color_scale,
        labels={selected_metric: color_label},
        hover_data={
            'delivery_time': True,
            'review_score': True,
            'customer_state': False
        },
        title=f'Average {color_label} by State'
    )
    
    fig.update_geos(fitbounds="locations", visible=False)
    fig.update_layout(
        margin={"r":0,"t":50,"l":0,"b":0},
        clickmode='event+select',
        autosize=True,
        width=1000,
        height=600,
        coloraxis_colorbar=dict(
            title=color_label,
            thicknessmode="pixels", thickness=15,
            lenmode="pixels", len=200,
            yanchor="middle", y=0.5,
            xanchor="left", x=-0.1
        )
    )

    return fig

# Callback for Seller Analysis
@app.callback(
    [Output('seller-scatter-plot', 'figure'),
     Output('top-sellers-ranking', 'children')],
    [Input('seller-go-button', 'n_clicks')],
    [State('state-filter-dropdown', 'value'),
     State('category-filter-dropdown', 'value'),
     State('ranking-filter-dropdown', 'value')]
)
def update_seller_analysis(n_clicks, selected_state, selected_category, ranking_filter):
    filtered_data = merged_df

    if selected_state:
        filtered_data = filtered_data[filtered_data['customer_state_summary'] == selected_state]
    if selected_category:
        filtered_data = filtered_data[filtered_data['product_category_name_english_summary'] == selected_category]

        # Apply the ranking filter
    if ranking_filter == 'best':
        filtered_data = filtered_data.nlargest(10, 'revenue_final')
    elif ranking_filter == 'worst':
        filtered_data = filtered_data.nsmallest(10, 'revenue_final')

    fig = px.scatter(
        filtered_data,
        x='delivery_time_summary',
        y='revenue_final',
        size='avg_rating',
        hover_name='seller_id',
        title='Seller Analysis: Delivery Time vs. Revenue with Rating as Size',
        labels={'delivery_time_summary': 'Avg Delivery Time', 'revenue_final': 'Revenue', 'avg_rating': 'Avg Rating'},
        size_max=60
    )
    
    fig.update_traces(marker=dict(color=filtered_data['avg_rating'], colorscale='Plasma'))
    
    fig.update_layout(
        margin={"r":0,"t":50,"l":0,"b":0},
        height=800,
        width=1000
    )

    fig.add_annotation(
        xref="paper", yref="paper",
        x=1.05, y=1,
        showarrow=False,
        text="Dot size represents average rating",
        font=dict(
            size=12,
            color="black"
        ),
        align="left"
    )

    def get_top_n_unique(data, column, n=5):
        return data.drop_duplicates(subset=['seller_id']).nlargest(n, column)[['seller_id', column]]

    top_sellers_revenue = get_top_n_unique(filtered_data, 'revenue_final')
    top_sellers_delivery_time = get_top_n_unique(filtered_data, 'delivery_time_summary')
    top_sellers_rating = get_top_n_unique(filtered_data, 'avg_rating')
    
    filtered_data['overall_score'] = (
        (filtered_data['revenue_final'].rank(ascending=False) +
         filtered_data['delivery_time_summary'].rank(ascending=True) +
         filtered_data['avg_rating'].rank(ascending=False)) / 3
    )
    top_sellers_overall = get_top_n_unique(filtered_data, 'overall_score')

    top_sellers_content = html.Div([
        html.H3("Top 5 by Revenue"),
        dbc.Table.from_dataframe(top_sellers_revenue, striped=True, bordered=True, hover=True),
        
        html.H3("Top 5 by Delivery Time"),
        dbc.Table.from_dataframe(top_sellers_delivery_time, striped=True, bordered=True, hover=True),
        
        html.H3("Top 5 by Rating"),
        dbc.Table.from_dataframe(top_sellers_rating, striped=True, bordered=True, hover=True),
        
        html.H3("Top 5 Overall"),
        dbc.Table.from_dataframe(top_sellers_overall, striped=True, bordered=True, hover=True)
    ])

    return fig, top_sellers_content

# Callback for Seller Power and Conversion Rates Analysis
@app.callback(
    [Output('top-categories-plot', 'figure'),
     Output('conversion-rates-plot', 'figure')],
    [Input('analyze-button', 'n_clicks')],
    [State('num-top-categories-dropdown', 'value'),
     State('segment-dropdown', 'value')]
)
def update_seller_power_conversion(n_clicks, num_top_categories, selected_segment):
    # Plot High Power Categories
    top_categories = category_analysis.sort_values(by='avg_sales_per_seller', ascending=False).head(num_top_categories)
    fig1 = px.bar(
        top_categories,
        x='product_category_name_english',
        y='avg_sales_per_seller',
        title=f'Top {num_top_categories} Categories with Highest Seller Power',
        labels={'avg_sales_per_seller': 'Average Sales per Seller'},
        color='avg_sales_per_seller', # Adding color to the bars
        color_continuous_scale='Viridis'
    )
    fig1.add_hline(y=high_power_threshold, line_dash="dash", line_color="red", annotation_text="High Power Threshold")

    # Plot Conversion Rates for Segment
    if selected_segment:
        filtered_data = conversion_data[conversion_data['business_segment'] == selected_segment]
        fig2 = px.bar(
            filtered_data,
            x='origin',
            y='conversion_rate',
            title=f'Conversion Rates for Business Segment: {selected_segment}',
            labels={'conversion_rate': 'Conversion Rate'},
            color='conversion_rate', # Adding color to the bars
            color_continuous_scale='Blues'
        )
    else:
        fig2 = {}

    return fig1, fig2

if __name__ == "__main__":
    app.run_server(port=8099)

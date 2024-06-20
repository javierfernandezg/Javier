import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import io
import base64
from datetime import timedelta
from sklearn.metrics import mean_squared_error
import streamlit as st
import plotly.express as px
import zipfile


with zipfile.ZipFile('merged_final_dataset.csv.zip', 'r') as zipf:
    with zipf.open('merged_final_dataset.csv') as f:
        merged_df = pd.read_csv(f)

with zipfile.ZipFile('final_dataset.csv.zip', 'r') as zipf:
    with zipf.open('final_dataset.csv') as f:
        df = pd.read_csv(f)

# Convert date columns to datetime
df['order_purchase_timestamp'] = pd.to_datetime(df['order_purchase_timestamp'])
df['order_delivered_customer_date'] = pd.to_datetime(df['order_delivered_customer_date'], errors='coerce')

# Calculate delivery time in days
df['delivery_time'] = (df['order_delivered_customer_date'] - df['order_purchase_timestamp']).dt.days

# Filter out invalid data
df = df[df['delivery_time'].notna() & df['review_score'].notna()]

# Calculate average delivery time and rating by state
state_summary = df.groupby('customer_state').agg({'delivery_time': 'mean', 'review_score': 'mean'}).reset_index()

st.sidebar.title("Olist Consulting")
page = st.sidebar.radio("Navigate", ["Demand Forecast", "Rating and Delivery Time", "Seller Analysis"])

st.sidebar.image("https://autonomobrasil.com/wp-content/uploads/2020/08/94330eda4548620db8263200465ecbc4_content_img_856036196532-2.png", use_column_width=True)

st.image("https://meta.com.br/wp-content/uploads/2022/05/Logo-Olist.png", width=200)

# Demand Forecast Analysis Layout
if page == "Demand Forecast":
    st.title("Demand Forecast Analysis")

    state = st.selectbox('Select a customer state', df['customer_state'].unique(), key='state')
    category = st.selectbox('Select a product category', df['product_category_name_english'].unique(), key='category')
    forecast_option = st.radio(
        'Forecast Option',
        ('Only by Category', 'Only by State', 'By Both State and Category'),
        index=1,
        key='forecast_option'
    )

    go_button = st.button('Go')

    def prepare_data(data, selection_type, customer_state=None, product_category=None):
        if selection_type == 'Only by State':
            df = data[data['customer_state'] == customer_state].copy()
        elif selection_type == 'Only by Category':
            df = data[data['product_category_name_english'] == product_category].copy()
        elif selection_type == 'By Both State and Category':
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

        start_date = test['order_purchase_timestamp'].min() - timedelta(days=30)
        plot_data = prepared_df[(prepared_df['order_purchase_timestamp'] >= start_date) & (prepared_df['order_purchase_timestamp'] <= test['order_purchase_timestamp'].max())]

        fig, ax = plt.subplots(figsize=(14, 7))
        ax.plot(plot_data['order_purchase_timestamp'], plot_data['demand'], label='Historical')
        ax.plot(test['order_purchase_timestamp'], y_test, label='Test')
        ax.plot(test['order_purchase_timestamp'], preds, label='Forecast')
        ax.fill_between(test['order_purchase_timestamp'], lower_bounds, upper_bounds, color='gray', alpha=0.2, label='95% Prediction Interval')
        ax.legend()

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

    if go_button:
        plot_base64, rmse, forecast_comparison = analyze_orders(forecast_option, state, category)

        st.image(f'data:image/png;base64,{plot_base64}', use_column_width=True)
        st.write(f"Root Mean Square Error (RMSE): {rmse}")
        st.dataframe(forecast_comparison)

# Rating and Delivery Time Analysis Layout
elif page == "Rating and Delivery Time":
    st.title("Rating and Delivery Time Analysis")
    
    metric = st.selectbox(
        'Select a metric to visualize',
        ['Delivery Time', 'Rating'],
        key='metric'
    )

    if metric == 'Delivery Time':
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
        color=metric.lower().replace(' ', '_'),
        color_continuous_scale=color_scale,
        labels={metric.lower().replace(' ', '_'): color_label},
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

    st.plotly_chart(fig, use_container_width=True)

# Seller Analysis Layout
elif page == "Seller Analysis":
    st.title("Seller Analysis")
    
    selected_metric = st.selectbox(
        'Select a metric to filter',
        ['Revenue', 'Delivery Time', 'Rating'],
        key='seller_metric'
    )
    
    selected_state = st.selectbox(
        'Select a customer state',
        ['All'] + list(merged_df['customer_state_summary'].unique()),
        key='state_filter'
    )
    
    selected_category = st.selectbox(
        'Select a product category',
        ['All'] + list(merged_df['product_category_name_english_summary'].unique()),
        key='category_filter'
    )

    go_button_seller = st.button('Go', key='seller_go')

    if go_button_seller:
        filtered_data = merged_df

        if selected_state != 'All':
            filtered_data = filtered_data[filtered_data['customer_state_summary'] == selected_state]
        if selected_category != 'All':
            filtered_data = filtered_data[filtered_data['product_category_name_english_summary'] == selected_category]

        fig = px.scatter(
            filtered_data,
            x='delivery_time_summary',
            y='revenue_final',
            size='review_score_summary',
            color='seller_id',
            labels={'delivery_time_summary': 'Avg Delivery Time (days)', 'revenue_final': 'Revenue', 'review_score_summary': 'Avg Rating'},
            title=f'Seller Performance: {selected_metric} Analysis',
            hover_name='seller_id'
        )
        
        fig.update_layout(
            margin={"r":0,"t":50,"l":0,"b":0},
            clickmode='event+select',
            autosize=True,
            width=1000,
            height=600
        )
        
        st.plotly_chart(fig, use_container_width=True)

if name == "main":
app.run_server(port=8054)

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

# Cargar los datos
df = pd.read_csv('data/final_dataset.csv')
merged_df = pd.read_csv('data/merged_final_dataset.csv')
closed_deals = pd.read_csv('data/olist_closed_deals_dataset.csv')
qualified_leads = pd.read_csv('data/olist_marketing_qualified_leads_dataset.csv')

# Convertir columnas de fecha a datetime
df['order_purchase_timestamp'] = pd.to_datetime(df['order_purchase_timestamp'])
df['order_delivered_customer_date'] = pd.to_datetime(df['order_delivered_customer_date'], errors='coerce')

# Calcular tiempo de entrega en días
df['delivery_time'] = (df['order_delivered_customer_date'] - df['order_purchase_timestamp']).dt.days

# Filtrar datos inválidos
df = df[df['delivery_time'].notna() & df['review_score'].notna()]

# Calcular tiempo de entrega promedio y calificación por estado
state_summary = df.groupby('customer_state').agg({'delivery_time': 'mean', 'review_score': 'mean'}).reset_index()

# Sidebar
st.sidebar.title("Olist Consulting")
page = st.sidebar.selectbox(
    "Choose a page",
    ["Demand Forecast", "Rating and Delivery Time", "Seller Analysis", "Seller Power and Conversion Rates"]
)

# Demand Forecast Analysis
if page == "Demand Forecast":
    st.title("Demand Forecast Analysis")

    state = st.selectbox("Select a customer state", df['customer_state'].unique())
    category = st.selectbox("Select a product category", df['product_category_name_english'].unique())
    forecast_option = st.radio(
        "Forecast Option",
        ['Only by Category', 'Only by State', 'By Both State and Category'],
        index=1
    )

    if st.button('Go'):
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
            st.write(f'Root Mean Square Error (RMSE): {rmse}')

            start_date = test['order_purchase_timestamp'].min() - timedelta(days=30)
            plot_data = prepared_df[(prepared_df['order_purchase_timestamp'] >= start_date) & (prepared_df['order_purchase_timestamp'] <= test['order_purchase_timestamp'].max())]

            plt.figure(figsize=(14, 7))
            plt.plot(plot_data['order_purchase_timestamp'], plot_data['demand'], label='Historical')
            plt.plot(test['order_purchase_timestamp'], y_test, label='Test')
            plt.plot(test['order_purchase_timestamp'], preds, label='Forecast')
            plt.fill_between(test['order_purchase_timestamp'], lower_bounds, upper_bounds, color='gray', alpha=0.2, label='95% Prediction Interval')
            plt.legend()

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

        forecast_option_mapping = {
            'Only by Category': 'category',
            'Only by State': 'state',
            'By Both State and Category': 'both'
        }

        selection_type = forecast_option_mapping[forecast_option]
        
        plot_base64, rmse, forecast_comparison = analyze_orders(selection_type, state, category)
        st.image(f'data:image/png;base64,{plot_base64}', use_column_width=True)
        st.write(f"Root Mean Square Error (RMSE): {rmse}")
        st.dataframe(forecast_comparison)

# Rating and Delivery Time Analysis
elif page == "Rating and Delivery Time":
    st.title("Rating and Delivery Time Analysis")
    metric = st.selectbox(
        "Select metric",
        ['delivery_time', 'review_score'],
        index=0
    )

    if metric == 'delivery_time':
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
        color=metric,
        color_continuous_scale=color_scale,
        labels={metric: color_label},
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

# Seller Analysis
elif page == "Seller Analysis":
    st.title("Seller Analysis")

    selected_state = st.selectbox("Select a customer state", [None] + list(merged_df['customer_state_summary'].unique()), index=0)
    selected_category = st.selectbox("Select a product category", [None] + list(merged_df['product_category_name_english_summary'].unique()), index=0)
    ranking_filter = st.selectbox(
        "Select ranking filter",
        ['Top 10 Best Sellers', 'Top 10 Worst Sellers'],
        index=0
    )

    if st.button('Go'):
        filtered_data = merged_df

        if selected_state:
            filtered_data = filtered_data[filtered_data['customer_state_summary'] == selected_state]
        if selected_category:
            filtered_data = filtered_data[filtered_data['product_category_name_english_summary'] == selected_category]

        if ranking_filter == 'Top 10 Best Sellers':
            filtered_data = filtered_data.nlargest(10, 'revenue_final')
        elif ranking_filter == 'Top 10 Worst Sellers':
            filtered_data = filtered_data.nsmallest(10, 'revenue_final')

        fig = px.scatter(
            filtered_data,
            x='delivery_time_summary',
            y='revenue_final',
            size='avg_rating',
            hover_name='seller_id',
            title='Seller Analysis: Delivery Time vs. Revenue with Rating as Size',
            labels={'delivery_time_summary': 'Avg Delivery Time', 'revenue_final': 'Revenue', 'avg_rating': 'Avg Rating'},
            trendline="ols",
            trendline_scope="overall",
            color='avg_rating',
            color_continuous_scale='Bluered_r',
            range_color=[filtered_data['avg_rating'].min(), filtered_data['avg_rating'].max()]
        )

        fig.update_layout(
            xaxis=dict(title='Avg Delivery Time'),
            yaxis=dict(title='Revenue'),
            margin=dict(l=40, r=40, b=40, t=40),
            width=800,
            height=600,
        )
        
        st.plotly_chart(fig, use_container_width=True)

# Seller Power and Conversion Rates
elif page == "Seller Power and Conversion Rates":
    st.title("Seller Power and Conversion Rates")
    
    num_top_categories = st.selectbox(
        'Select number of top categories',
        [5, 10, 15, 20],
        index=1
    )
    
    segment = st.selectbox("Select a business segment", [None] + list(conversion_data['business_segment'].unique()), index=0)

    if st.button('Analyze'):
        top_categories = category_analysis.sort_values(by='avg_sales_per_seller', ascending=False).head(num_top_categories)
        fig1 = px.bar(
            top_categories,
            x='product_category_name_english',
            y='avg_sales_per_seller',
            title=f'Top {num_top_categories} Categories with Highest Seller Power',
            labels={'avg_sales_per_seller': 'Average Sales per Seller'},
            color='avg_sales_per_seller',
            color_continuous_scale='Viridis'
        )
        fig1.add_hline(y=high_power_threshold, line_dash="dash", line_color="red", annotation_text="High Power Threshold")

        if segment:
            filtered_data = conversion_data[conversion_data['business_segment'] == segment]
            fig2 = px.bar(
                filtered_data,
                x='origin',
                y='conversion_rate',
                title=f'Conversion Rates for Business Segment: {segment}',
                labels={'conversion_rate': 'Conversion Rate'},
                color='conversion_rate',
                color_continuous_scale='Blues'
            )
        else:
            fig2 = {}

        st.plotly_chart(fig1, use_container_width=True)
        if fig2:
            st.plotly_chart(fig2, use_container_width=True)

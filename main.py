import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import plotly.express as px
import joblib

# Load and preprocess data
@st.cache_data
def load_data():
    data = pd.read_csv('simulerad_smaltdata_C.csv')
    
    # Calculate expected carbon from input materials
    data['Beräknad_kol'] = (
        (data['SkrotA_andel_pct']/100 * data['SkrotA_kol_pct']/100) +
        (data['SkrotB_andel_pct']/100 * data['SkrotB_kol_pct']/100) +
        (data['SkrotC_andel_pct']/100 * data['SkrotC_kol_pct']/100) +
        (data['Tillsatt_kol_kg']/data['Total_vikt_kg'])
    ) * 100
    
    # Calculate carbon deviation
    data['Kol_avvikelse'] = data['Faktisk_kol_pct'] - data['Förväntad_kol_pct']
    
    return data

# Main app
def main():
    st.set_page_config(page_title="Steel Melt Carbon Predictor", layout="wide")
    st.title("🏭 AI-Powered Steel Melt Carbon Predictor")
    st.markdown("""
    **Problem:** In foundries, predicting final carbon content is critical for steel quality. 
    Traditional methods rely on static calculations that ignore process dynamics (temperature, 
    scrap mix, melting time). This leads to **off-spec melts and costly rework**.
    """)
    st.write("""NOTE: This is a tech demo showing a conceptiual application of modern
    machine learning application in an industrial setting.""")
    # Load data
    df = load_data()

    with st.sidebar:
        st.markdown("""
        ### How This Works
        This AI model predicts carbon content more accurately by learning from:
        - **Process Parameters** (Temp, Melt Time)
        - **Scrap Composition** (A/B/C grades)
        - **Added Carbon** (Real-world adjustments)
        - **Historical Deviations** (Expected vs Actual Carbon)
        """)
    
    # Sidebar for user input
    st.sidebar.header('Model Configuration')
    
    # Normalization options
    st.sidebar.subheader('Data Normalization')
    normalization_option = st.sidebar.selectbox(
        'Select Normalization Method',
        ['None', 'Standard Scaler (Z-score)', 'Min-Max Scaler'],
        index=1
    )
    
    test_size = st.sidebar.slider('Test Size Ratio', 0.1, 0.5, 0.2, step=0.05)
    n_estimators = st.sidebar.slider('Number of Trees', 10, 300, 150)
    max_depth = st.sidebar.slider('Max Tree Depth', 3, 30, 15)
    
    # Feature selection
    st.sidebar.header('Feature Selection')
    features = [
        'Temp_C', 'Tid_sen_senast_smälta_s', 'Smälttid_s',
        'SkrotA_andel_pct', 'SkrotB_andel_pct', 'SkrotC_andel_pct',
        'Tillsatt_kol_kg', 'Beräknad_kol', 'Förväntad_kol_pct'
    ]
    selected_features = st.sidebar.multiselect(
        'Select Features', 
        features, 
        default=features
    )
    
    # Prepare data
    X = df[selected_features]
    y = df['Faktisk_kol_pct']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    
    # Apply normalization
    if normalization_option == 'Standard Scaler (Z-score)':
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        st.session_state.scaler = scaler  # Save scaler for predictions
    elif normalization_option == 'Min-Max Scaler':
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        st.session_state.scaler = scaler
    else:
        st.session_state.scaler = None
    
    # Train model
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Show results
    col1, col2 = st.columns(2)
    with col1:
        st.subheader('Model Performance')
        st.metric("Normalization Method", normalization_option)
        st.metric("Mean Absolute Error", f"{mae:.4f}")
        st.metric("R² Score", f"{r2:.4f}")
        
        # Actual vs Predicted plot
        st.subheader('Actual vs Predicted')
        results_df = pd.DataFrame({
            'Actual': y_test,
            'Predicted': y_pred,
            'Error': y_pred - y_test
        })
        fig = px.scatter(
            results_df, x='Actual', y='Predicted', 
            color='Error', color_continuous_scale='RdYlGn',
            title='Actual vs Predicted Carbon Content',
            trendline='ols'
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Feature importance
        st.subheader('Feature Importance')
        importance_df = pd.DataFrame({
            'Feature': selected_features,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=True)
        
        fig = px.bar(
            importance_df, 
            x='Importance', 
            y='Feature', 
            orientation='h',
            title='Feature Importance Scores',
            color='Importance',
            color_continuous_scale='Blues'
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("""
        ### Why Feature Importance Matters
        The model reveals which factors most impact carbon content:
        - **High Importance**: Strong effect on results (prioritize control)  
        - **Low Importance**: Minimal effect (can potentially ignore)  
        *Example:* If "Time Since Last Melt" is important, it suggests furnace degradation affects chemistry.
        """)
        
        # Show normalized data stats
        if normalization_option != 'None' and 'scaler' in st.session_state:
            st.subheader('Normalization Statistics')
            scaler = st.session_state.scaler
            
            if isinstance(scaler, StandardScaler):
                stats_df = pd.DataFrame({
                    'Feature': selected_features,
                    'Mean': scaler.mean_,
                    'Std Dev': scaler.scale_
                })
                st.dataframe(stats_df.style.format({
                    'Mean': '{:.4f}',
                    'Std Dev': '{:.4f}'
                }))
            elif isinstance(scaler, MinMaxScaler):
                stats_df = pd.DataFrame({
                    'Feature': selected_features,
                    'Min': scaler.data_min_,
                    'Max': scaler.data_max_
                })
                st.dataframe(stats_df.style.format({
                    'Min': '{:.4f}',
                    'Max': '{:.4f}'
                }))
    
    # Prediction interface
    st.subheader('🔮 Predict New Melt')
    with st.form('prediction_form'):
        col1, col2, col3 = st.columns(3)
        temp = col1.number_input('Temperature (°C)', 1400, 1500, 1430)
        time_since = col2.number_input('Time Since Last Melt (s)', 0, 200000, 100000)
        melt_time = col3.number_input('Melt Time (s)', 2000, 8000, 5000)
        
        col1, col2, col3 = st.columns(3)
        scrap_a = col1.number_input('Scrap A %', 0.0, 100.0, 30.0)
        scrap_b = col2.number_input('Scrap B %', 0.0, 100.0, 40.0)
        scrap_c = col3.number_input('Scrap C %', 0.0, 100.0, 30.0)
        
        col1, col2, col3 = st.columns(3)
        scrap_a_carbon = col1.number_input('Scrap A Carbon %', 0.0, 2.0, 0.5, step=0.01)
        scrap_b_carbon = col2.number_input('Scrap B Carbon %', 0.0, 2.0, 1.0, step=0.01)
        scrap_c_carbon = col3.number_input('Scrap C Carbon %', 0.0, 2.0, 1.5, step=0.01)
        
        col1, col2 = st.columns(2)
        added_carbon = col1.number_input('Added Carbon (kg)', 0.0, 500.0, 200.0)
        total_weight = col2.number_input('Total Weight (kg)', 5000, 15000, 10000)
        
        expected_carbon = st.number_input('Expected Carbon %', 2.5, 4.0, 3.3, step=0.01)
        
        # Calculate derived features
        calculated_carbon = (
            (scrap_a/100 * scrap_a_carbon) +
            (scrap_b/100 * scrap_b_carbon) +
            (scrap_c/100 * scrap_c_carbon) +
            (added_carbon / total_weight)
        ) * 100
        
        submit = st.form_submit_button('Predict Carbon Content')
        
        if submit:
            # Create input array
            input_data = {
                'Temp_C': temp,
                'Tid_sen_senast_smälta_s': time_since,
                'Smälttid_s': melt_time,
                'SkrotA_andel_pct': scrap_a,
                'SkrotB_andel_pct': scrap_b,
                'SkrotC_andel_pct': scrap_c,
                'Tillsatt_kol_kg': added_carbon,
                'Beräknad_kol': calculated_carbon,
                'Förväntad_kol_pct': expected_carbon
            }
            
            # Convert to dataframe
            input_df = pd.DataFrame([input_data])[selected_features]
            
            # Apply normalization if used in training
            if 'scaler' in st.session_state and st.session_state.scaler is not None:
                input_df = st.session_state.scaler.transform(input_df)
            
            # Make prediction
            prediction = model.predict(input_df)[0]
            
            # Display results
            col1, col2, col3 = st.columns(3)
            st.success(f'### Predicted Carbon Content: **{prediction:.3f}%**')

            col1, col2, col3 = st.columns(3)
            col1.metric(
                "Expected Carbon", 
                f"{expected_carbon:.3f}%",
                help="Traditional calculation based on scrap inputs and added carbon alone"
            )
            col2.metric(
                "Predicted Carbon", 
                f"{prediction:.3f}%", 
                f"{prediction - expected_carbon:.3f}%",
                delta_color="inverse",
                help="AI prediction accounting for process dynamics and historical patterns"
            )
            col3.metric(
                "Deviation", 
                f"{prediction - expected_carbon:.3f}%",
                help="How much the AI expects reality to differ from traditional calculations"
            )

            st.markdown("""
            🔍 **Key Insight:**  
            The difference between *Expected* and *Predicted* carbon shows where traditional methods 
            underestimate/overestimate results. This model captures hidden factors like:
            - Temperature effects on carbon absorption  
            - Scrap mix interactions  
            - Furnace condition (via time since last melt)  
            """)
            
            # Show input summary
            with st.expander("Input Summary"):
                input_summary = pd.DataFrame({
                    'Parameter': [
                        'Temperature', 'Time Since Last Melt', 'Melt Time',
                        'Scrap A %', 'Scrap B %', 'Scrap C %',
                        'Scrap A Carbon', 'Scrap B Carbon', 'Scrap C Carbon',
                        'Added Carbon', 'Total Weight', 'Expected Carbon',
                        'Calculated Carbon'
                    ],
                    'Value': [
                        f"{temp} °C", f"{time_since} s", f"{melt_time} s",
                        f"{scrap_a}%", f"{scrap_b}%", f"{scrap_c}%",
                        f"{scrap_a_carbon}%", f"{scrap_b_carbon}%", f"{scrap_c_carbon}%",
                        f"{added_carbon} kg", f"{total_weight} kg", f"{expected_carbon}%",
                        f"{calculated_carbon:.3f}%"
                    ]
                })
                st.table(input_summary)

if __name__ == '__main__':
    main()
import streamlit as st
import pandas as pd
import joblib
import numpy as np
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="🏠 House Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': None
    }
)
# Custom CSS for better styling
st.markdown("""
<style>
    /* Force dark theme */
    .main {
        background-color: #0e1117 !important;
    }
    
    .stApp {
        background-color: #0e1117 !important;
    }
    
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .sub-header {
        font-size: 2.5rem;
        color: #ecf0f1;
        margin-bottom: 1rem;
        font-weight: bold;
    }
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 2rem 0;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
    .input-section {
        padding: 1rem 0;
        margin: 1rem 0;
        color: #ecf0f1;
        min-height: 0px;
    }
    
    /* Fix input field visibility */
    .stSelectbox > div > div {
        background-color: #2c3e50 !important;
        color: white !important;
        border: 1px solid #34495e !important;
    }
    .stSelectbox > div > div:hover {
        background-color: #34495e !important;
        border-color: #3498db !important;
    }
    
    .stNumberInput > div > div > input {
        background-color: #2c3e50 !important;
        color: white !important;
        border: 1px solid #34495e !important;
    }
    .stNumberInput > div > div > input:focus {
        background-color: #34495e !important;
        border-color: #3498db !important;
        box-shadow: 0 0 0 2px rgba(52, 152, 219, 0.2) !important;
    }
    
    .stCheckbox > div > div {
        background-color: #2c3e50 !important;
        border: 1px solid #34495e !important;
    }
    
    /* Style text inputs */
    .stTextInput > div > div > input {
        background-color: #2c3e50 !important;
        color: white !important;
        border: 1px solid #34495e !important;
    }
    .stTextInput > div > div > input:focus {
        background-color: #34495e !important;
        border-color: #3498db !important;
        box-shadow: 0 0 0 2px rgba(52, 152, 219, 0.2) !important;
    }
    
    /* Style selectbox options */
    .stSelectbox > div > div > div > div {
        background-color: #2c3e50 !important;
        color: white !important;
    }
    .stSelectbox > div > div > div > div:hover {
        background-color: #34495e !important;
    }
    
    /* Style number input buttons */
    .stNumberInput > div > div > button {
        background-color: #34495e !important;
        color: white !important;
        border: 1px solid #34495e !important;
    }
    .stNumberInput > div > div > button:hover {
        background-color: #3498db !important;
        border-color: #3498db !important;
    }
    
    /* Style checkbox */
    .stCheckbox > div > div > div {
        background-color: #2c3e50 !important;
        border: 1px solid #34495e !important;
    }
    .stCheckbox > div > div > div:checked {
        background-color: #3498db !important;
        border-color: #3498db !important;
    }
    
    /* Improve overall contrast */
    .stMarkdown {
        color: #ecf0f1 !important;
    }
    
    /* Style labels */
    .stMarkdown p {
        color: #ecf0f1 !important;
    }
    
    /* Style sidebar */
    .css-1d391kg {
        background-color: #0e1117 !important;
    }
    
    /* Style sidebar content */
    .css-1lcbmhc {
        background-color: #1a1a1a !important;
        border-radius: 10px !important;
        padding: 1rem !important;
        margin: 0.5rem !important;
    }
    
    /* Style sidebar headers */
    .css-1lcbmhc h3 {
        color: #3498db !important;
        font-weight: bold !important;
        margin-bottom: 0.5rem !important;
    }
    
    /* Style sidebar text */
    .css-1lcbmhc p {
        color: #ecf0f1 !important;
        font-size: 0.9rem !important;
        line-height: 1.4 !important;
    }
    
    /* Style sidebar lists */
    .css-1lcbmhc ul {
        color: #ecf0f1 !important;
        font-size: 0.9rem !important;
    }
    
    /* Style sidebar dividers */
    .css-1lcbmhc hr {
        border-color: #34495e !important;
        margin: 1rem 0 !important;
    }
    
    /* Style sidebar columns */
    .css-1lcbmhc .row-widget {
        background-color: transparent !important;
    }
    
    /* Style sidebar strong text */
    .css-1lcbmhc strong {
        color: #3498db !important;
    }
    
    /* Style button */
    .stButton > button {
        background-color: #3498db !important;
        color: white !important;
        border: none !important;
        border-radius: 5px !important;
        padding: 10px 20px !important;
        font-weight: bold !important;
    }
    .stButton > button:hover {
        background-color: #2980b9 !important;
    }
    
    /* Style sidebar buttons */
    .stButton > button {
        background-color: #2c3e50 !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 12px 8px !important;
        font-size: 0.8rem !important;
        font-weight: bold !important;
        transition: all 0.3s ease !important;
        margin: 2px 0 !important;
        height: 50px !important;
        width: 100% !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        text-align: center !important;
        white-space: normal !important;
        word-wrap: break-word !important;
        line-height: 1.2 !important;
    }
    .stButton > button:hover {
        background-color: #34495e !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 12px rgba(52, 73, 94, 0.4) !important;
    }
    
    /* Style sidebar columns for better alignment */
    .css-1r6slb0 {
        gap: 8px !important;
    }
    
    /* Ensure consistent button sizing */
    .stButton {
        width: 100% !important;
        height: 50px !important;
    }
    
    /* Style primary button (Back to Predictor) */
    .stButton > button[data-baseweb="button"] {
        background-color: #3498db !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 12px 16px !important;
        font-weight: bold !important;
        font-size: 0.9rem !important;
        margin: 8px 0 !important;
        height: 50px !important;
    }
    .stButton > button[data-baseweb="button"]:hover {
        background-color: #2980b9 !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 12px rgba(52, 152, 219, 0.4) !important;
    }
    
    /* Style page content */
    .main .block-container {
        padding-top: 0rem !important;
        padding-bottom: 2rem !important;
        min-height: 100vh !important;
    }
    
    /* Style page headers */
    .main-header {
        font-size: 2.5rem !important;
        font-weight: bold !important;
        text-align: center !important;
        color: #3498db !important;
        margin-bottom: 2rem !important;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1) !important;
    }
    
    /* Add more spacing between form elements */
    .stMarkdown {
        margin-bottom: 1rem !important;
    }
    
    .stSelectbox, .stNumberInput, .stCheckbox, .stTextInput {
        margin-bottom: 1rem !important;
    }
    
    /* Style prediction box with more height */
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 2rem 0;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        min-height: 150px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load the trained model with joblib"""
    try:
        model = joblib.load('linear_regression_model.pkl')
        return model
    except FileNotFoundError:
        st.error("Model file 'linear_regression_model.pkl' not found!")
        return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def clean_city_name(city_name):
    """Clean city name by removing commas and extra information"""
    # Remove the CITY_ prefix
    if city_name.startswith('CITY_'):
        city_name = city_name.replace('CITY_', '')
    
    # Remove leading comma and space
    if city_name.startswith(', '):
        city_name = city_name[2:]
    
    # Split by comma and take the first part (main city name)
    if ',' in city_name:
        city_name = city_name.split(',')[0].strip()
    
    # Remove extra spaces
    city_name = ' '.join(city_name.split())
    
    return city_name

def get_available_locations():
    """Get available cities and localities from the model"""
    try:
        model = joblib.load('linear_regression_model.pkl')
        
        # Try to get feature names from the model
        if hasattr(model, 'feature_names_in_'):
            expected_features = model.feature_names_in_
        else:
            st.warning("Model doesn't have feature names. Cannot load available locations.")
            return [], []
        
        cities = []
        localities = []
        
        for feature in expected_features:
            if feature.startswith('CITY_'):
                city = clean_city_name(feature)
                if city:  # Skip empty city names
                    cities.append(city)
            elif feature.startswith('LOCALITY_'):
                locality = feature.replace('LOCALITY_', '')
                if locality:  # Skip empty locality names
                    localities.append(locality)
        
        return sorted(list(set(cities))), sorted(list(set(localities)))
    except Exception as e:
        st.warning(f"Could not load available locations: {e}")
        return [], []

def create_input_features():
    """Create input features for the model"""
    st.markdown('<div class="input-section">', unsafe_allow_html=True)
    st.markdown('<h3 class="sub-header">📋 Property Details</h3>', unsafe_allow_html=True)
    
    # Get available cities and localities
    available_cities, available_localities = get_available_locations()
    
    col1, col2 = st.columns(2)
    
    with col1:
        # POSTED_BY
        posted_by = st.selectbox(
            "Posted By",
            ["Owner", "Dealer", "Builder"],
            help="Who posted the property listing"
        )
        posted_by_mapping = {"Owner": 1, "Dealer": 2, "Builder": 3}
        
        # UNDER_CONSTRUCTION
        under_construction = st.checkbox(
            "Under Construction",
            help="Is the property currently under construction?"
        )
        
        # RERA
        rera = st.checkbox(
            "RERA Registered",
            help="Is the property RERA registered?"
        )
        
        # BHK_NO
        bhk_no = st.number_input(
            "Number of Bedrooms (BHK)",
            min_value=1,
            max_value=10,
            value=2,
            help="Number of bedrooms in the property"
        )
        
        # SQUARE_FT
        square_ft = st.number_input(
            "Area (Square Feet)",
            min_value=100,
            max_value=10000,
            value=1000,
            step=50,
            help="Total area of the property in square feet"
        )
    
    with col2:
        # BHK_OR_RK (Note: Both become 1 as per your specification)
        bhk_or_rk = 1  # Fixed value as per your note
        
        # READY_TO_MOVE
        ready_to_move = st.checkbox(
            "Ready to Move",
            help="Is the property ready to move in?"
        )
        
        # RESALE
        resale = st.checkbox(
            "Resale Property",
            help="Is this a resale property?"
        )
        
        # LOCALITY
        if available_localities:
            locality = st.selectbox(
                "Locality",
                options=[""] + available_localities,
                help="Select the locality/area of the property"
            )
        else:
            locality = st.text_input(
                "Locality",
                placeholder="Enter locality name",
                help="Enter the locality/area of the property"
            )
        
        # CITY
        if available_cities:
            city = st.selectbox(
                "City",
                options=[""] + available_cities,
                help="Select the city where the property is located"
            )
        else:
            city = st.text_input(
                "City",
                placeholder="Enter city name",
                help="Enter the city where the property is located"
            )
    
    # Show available options if user wants to see them
    # with st.expander("📍 Available Cities and Localities"):
    #     if available_cities:
    #         st.write("**Available Cities:**")
    #         st.write(", ".join(available_cities[:20]) + ("..." if len(available_cities) > 20 else ""))
    #     if available_localities:
    #         st.write("**Available Localities:**")
    #         st.write(", ".join(available_localities[:20]) + ("..." if len(available_localities) > 20 else ""))
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    return {
        'POSTED_BY': posted_by_mapping[posted_by],
        'UNDER_CONSTRUCTION': int(under_construction),
        'RERA': int(rera),
        'BHK_NO.': bhk_no,
        'BHK_OR_RK': bhk_or_rk,
        'SQUARE_FT': square_ft,
        'READY_TO_MOVE': int(ready_to_move),
        'RESALE': int(resale),
        'LOCALITY': locality,
        'CITY': city
    }

def prepare_features(input_data):
    """Prepare features for prediction including one-hot encoding with proper column matching"""
    # Create base dataframe
    df = pd.DataFrame([input_data])
    
    # Clean city name if it has commas
    if 'CITY' in df.columns:
        df['CITY'] = df['CITY'].apply(clean_city_name)
    
    # One-hot encode LOCALITY and CITY
    locality_dummies = pd.get_dummies(df['LOCALITY'], prefix='LOCALITY')
    city_dummies = pd.get_dummies(df['CITY'], prefix='CITY')
    
    # Drop original categorical columns and add dummies
    df = df.drop(['LOCALITY', 'CITY'], axis=1)
    df = pd.concat([df, locality_dummies, city_dummies], axis=1)
    
    # Load the model to get expected feature names
    try:
        model = joblib.load('linear_regression_model.pkl')
        
        # Try to get feature names from the model
        if hasattr(model, 'feature_names_in_'):
            expected_features = model.feature_names_in_
        elif hasattr(model, 'coef_') and hasattr(model, 'intercept_'):
            
            num_features = len(model.coef_)
            expected_features = [f'feature_{i}' for i in range(num_features)]
        else:
            st.error("Model doesn't have expected attributes. Please retrain the model with feature names.")
            return None
        
        # Create a dataframe with all expected features, filled with zeros
        feature_df = pd.DataFrame(0, index=[0], columns=expected_features)
        
        # Fill in the values we have
        for col in df.columns:
            if col in expected_features:
                feature_df[col] = df[col].values
            # Handle potential column name variations
            elif col == 'BHK_NO.' and 'BHK_NO' in expected_features:
                feature_df['BHK_NO'] = df[col].values
            elif col == 'BHK_NO.' and 'BHK_NO.' in expected_features:
                feature_df['BHK_NO.'] = df[col].values
        
        return feature_df
        
    except Exception as e:
        st.error(f"Error preparing features: {e}")
        return None

def format_price(price):
    """Format price in Indian currency format"""
    # The model predicts in lakhs, so we need to convert to rupees first
    price_in_rupees = price * 100000
    
    if price_in_rupees < 100000:
        return f"₹{price_in_rupees:,.0f}"
    elif price_in_rupees < 10000000:
        return f"₹{price:.1f} Lakhs"
    else:
        return f"₹{price/100:.1f} Crores"

def show_about_page():
    st.markdown('<h1 class="main-header">About Us</h1>', unsafe_allow_html=True)
    
    # Main content with same layout as main page
    col1, col2, col3 = st.columns([0.5, 3, 0.5])
    
    with col2:
        st.markdown("""
        This application is a machine learning model designed to predict the market value of residential properties in India.
        The model uses a variety of features including property type, location, area, and construction status to make accurate predictions.
        """)
        st.markdown("---")
        st.markdown("### 💻 Made By:")
        st.markdown("""
        - Urvi Agarwal (1024190058)
        - Cheshta Garg (1024190008)
        """)
        st.markdown("---")
        if st.button("🏠 Back to Predictor", use_container_width=True, type="primary"):
            st.session_state.page = "main"

def show_model_info_page():
    st.markdown('<h1 class="main-header">Model Information</h1>', unsafe_allow_html=True)
    
    # Main content with same layout as main page
    col1, col2, col3 = st.columns([0.5, 3, 0.5])
    
    with col2:
        st.markdown("""
        This application uses a Linear Regression model trained on a large dataset of Indian real estate properties.
        The model has been optimized for accuracy and can predict property prices with a high degree of confidence.
        """)
        st.markdown("---")
        st.markdown("### 📊 Model Details")
        st.markdown("""
        - **Model Type:** Linear Regression
        - **Training Data:** Indian Real Estate Dataset (https://www.kaggle.com/code/sonialikhan/house-price-prediction-challenge-2024/notebook)
        - **Features Used:** 9
        - **Accuracy on Training Dataset:** 87.141% 
        - **Accuracy on Validation Dataset:** 74.315% 
        """)
        st.markdown("---")
        if st.button("🏠 Back to Predictor", use_container_width=True, type="primary"):
            st.session_state.page = "main"

def show_how_to_use_page():
    st.markdown('<h1 class="main-header">How to Use This Streamlit Interface</h1>', unsafe_allow_html=True)
    
    # Main content with same layout as main page
    col1, col2, col3 = st.columns([0.5, 3, 0.5])
    
    with col2:
        st.markdown("""
        To use this application, follow these simple steps:
        1. **Enter Property Details:**
           - Select "Posted By" (Owner, Dealer, Builder)
           - Check "Under Construction" if applicable
           - Select "RERA Registered" if the property is RERA registered
           - Enter the number of bedrooms (BHK)
           - Enter the total area in square feet
        2. **Location Details:**
           - Select the locality of the property
           - Enter the city where the property is located
        3. **Property Type:**
           - Check "Ready to Move" if the property is ready for occupancy
           - Select "Resale Property" if it's a resale
        4. **Predict:**
           - Click the "Predict Price" button to get an estimated market value.
        """)
        st.markdown("---")
        
        if st.button("🏠 Back to Predictor", use_container_width=True, type="primary"):
            st.session_state.page = "main"

def show_tips_page():
    st.markdown('<h1 class="main-header">Tips for Accurate Predictions</h1>', unsafe_allow_html=True)
    
    # Main content with same layout as main page
    col1, col2, col3 = st.columns([0.5, 3, 0.5])
    
    with col2:
        st.markdown("""
        To get the most accurate predictions, consider the following:
        1. **Location:** Properties in prime locations tend to have higher values.
        2. **Area:** Larger properties generally command higher prices.
        3. **Construction Status:** Newly constructed properties often have higher values.
        4. **RERA Registration:** RERA registered properties are generally more reliable.
        5. **BHK:** The number of bedrooms (BHK) is a significant factor.
        """)
        st.markdown("---")
        st.markdown("### 💡 Tips")
        st.markdown("""
        - Ensure all fields are filled correctly.
        - Try different locations to get a better understanding of market trends.
        - The prediction is an estimate and may vary based on local market conditions.
        """)
        st.markdown("---")
        if st.button("🏠 Back to Predictor", use_container_width=True, type="primary"):
            st.session_state.page = "main"

def show_price_ranges_page():
    st.markdown('<h1 class="main-header">Price Ranges in Major Indian Cities</h1>', unsafe_allow_html=True)
    
    # Main content with same layout as main page
    col1, col2, col3 = st.columns([0.5, 3, 0.5])
    
    with col2:
        st.markdown("""
        This section provides an overview of typical price ranges for different property types across major Indian cities.
        """)
        st.markdown("---")
        st.markdown("### 📈 Price Ranges Based On Dataset")
        
        # Create a two-column layout for price ranges and graph
        col1, col2 = st.columns([0.9, 1.05])
        
        with col1:
            st.markdown("""
            - **1 BHK:** ₹0.10 - 9.50 Crores  
            - **2 BHK:** ₹0.12 - 10.00 Crores  
            - **3 BHK:** ₹0.13 - 10.00 Crores      
            - **4 BHK:** ₹0.22 - 10.50 Crores  
            - **5 BHK:** ₹0.32 - 10.00 Crores  
            - **6 BHK:** ₹0.35 - 9.50 Crores  
            - **7 BHK:** ₹0.26 - 1.80 Crores  
            - **8 BHK:** ₹0.80 - 4.50 Crores  
            - **9 BHK:** ₹1.00 - 2.00 Crores  
            - **10 BHK:** ₹3.50 - 4.00 Crores  
            - **11 BHK:** ₹0.50 - 0.60 Crores  
            - **15 BHK:** ₹2.50 - 3.00 Crores  
            """)
        with col2:
            st.markdown("<div style='margin-top: -3.5rem;'>", unsafe_allow_html=True)
            st.image("graph.png", use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("---")
        st.markdown("### 💡 Tips")
        st.markdown("""
        - Prices can vary significantly based on locality, condition, and amenities.
        - This is a general guide and should be used as a reference.
        """)
        st.markdown("---")
        if st.button("🏠 Back to Predictor", use_container_width=True, type="primary"):
            st.session_state.page = "main"

def show_features_page():
    st.markdown('<h1 class="main-header">Features Used in Prediction</h1>', unsafe_allow_html=True)
    
    # Main content with same layout as main page
    col1, col2, col3 = st.columns([0.5, 3, 0.5])
    
    with col2:
        st.markdown("""
        The machine learning model considers several key features to predict property prices.
        """)
        st.markdown("---")
        st.markdown("### 🎯 Features")
        
        # Create a two-column layout for features and heatmap
        feat_col1, feat_col2 = st.columns([1.7, 1])
        
        with feat_col1:
            st.markdown("""
            - **POSTED_BY:** Who posted the property (Owner/Dealer/Builder)
            - **UNDER_CONSTRUCTION:** Whether property is under construction
            - **RERA:** RERA registration status (Yes/No)
            - **BHK_NO.:** Number of Bedrooms, Hall, and Kitchen
            - **SQUARE_FT:** Total area in square feet
            - **READY_TO_MOVE:** Whether property is ready to move in
            - **RESALE:** Whether it's a resale property
            - **LOCALITY:** Specific locality/area within the city
            - **CITY:** City where the property is located
            """)
        
        with feat_col2:
            st.markdown("<div style='margin-top: -3.5rem;'>", unsafe_allow_html=True)
            st.image("heatmap.jpg", use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("---")
        st.markdown("### 💡 Tips")
        st.markdown("""
        - The more detailed your input, the more accurate the prediction.
        - Try to provide as many relevant features as possible.
        """)
        st.markdown("---")
        if st.button("🏠 Back to Predictor", use_container_width=True, type="primary"):
            st.session_state.page = "main"

def show_main_page():
    # Header
    st.markdown('<h1 class="main-header">🏠 House Price Predictor</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Predict the market value of your property with our advanced ML model</p>', unsafe_allow_html=True)
    
    # Load model
    model = load_model()
    if model is None:
        st.error("❌ Model could not be loaded. Please check if the model file exists.")
        return
    
    # Main content
    col1, col2, col3 = st.columns([0.5, 3, 0.5])
    
    with col2:
        # Get user inputs
        input_data = create_input_features()
        
        # Prediction button
        if st.button("🔮 Predict Price", type="primary", use_container_width=True):
            try:
                # Prepare features
                features_df = prepare_features(input_data)
                
                if features_df is None:
                    st.error("❌ Could not prepare features for prediction.")
                    return
                
                # Make prediction
                prediction = model.predict(features_df)[0]
                
                # Display result
                st.markdown(f"<h2 style='color: #3498db; margin-top: 4rem;'>Predicted Price</h2>", unsafe_allow_html=True)
                st.markdown(f"<h1 style='font-size: 3rem; margin: 0rem 0;'>{format_price(prediction)}</h1>", unsafe_allow_html=True)
                st.markdown(f"<p style='font-size: 1.1rem; opacity: 0.9;'>Estimated market value</p>", unsafe_allow_html=True)
                
                # Add padding below predict button
                st.markdown("<div style='margin-top: 3rem;'></div>", unsafe_allow_html=True)
                
                # Additional insights
                st.markdown("### 📈 Price Insights")
                col_a, col_b, col_c = st.columns(3)
                
                with col_a:
                    # Convert prediction from lakhs to rupees for per sq ft calculation
                    prediction_in_rupees = prediction * 100000
                    price_per_sqft = prediction_in_rupees / input_data['SQUARE_FT']
                    st.metric("Price per Sq Ft", f"₹{price_per_sqft:,.0f}")
                
                with col_b:
                    if input_data['SQUARE_FT'] > 0:
                        # Convert prediction from lakhs to rupees for average price calculation
                        prediction_in_rupees = prediction * 100000
                        avg_price = prediction_in_rupees / input_data['SQUARE_FT']
                        st.metric("Average Price", f"₹{avg_price:,.0f}/sq ft")
                
                with col_c:
                    if input_data['BHK_NO.'] > 0:
                        # Price per BHK in lakhs
                        price_per_bhk = prediction / input_data['BHK_NO.']
                        st.metric("Price per BHK", f"₹{price_per_bhk:.1f} Lakhs")
                
            except Exception as e:
                st.error(f"❌ Error making prediction: {e}")
                st.info("💡 Tip: Make sure all fields are filled correctly")
                # Show more detailed error info in debug mode
                if st.checkbox("Show detailed error"):
                    st.exception(e)

if __name__ == "__main__":
    # Initialize session state
    if 'page' not in st.session_state:
        st.session_state.page = "main"
    
    # Sidebar with information (always visible)
    with st.sidebar:
        st.markdown("## 🏠 House Price Predictor")
        st.markdown("---")
        
        # Interactive buttons for different pages
        if st.button("ℹ️ About", use_container_width=True):
            st.session_state.page = "about"
        
        if st.button("📊 Model Info", use_container_width=True):
            st.session_state.page = "model_info"
        
        if st.button("🔧 How to Use", use_container_width=True):
            st.session_state.page = "how_to_use"
        
        if st.button("💡 Tips", use_container_width=True):
            st.session_state.page = "tips"
        
        if st.button("📈 Price Ranges", use_container_width=True):
            st.session_state.page = "price_ranges"
        
        if st.button("🎯 Features", use_container_width=True):
            st.session_state.page = "features"
        
        st.markdown("---")
        
        # Back to main button
        if st.button("🏠 Back to Predictor", use_container_width=True, type="primary"):
            st.session_state.page = "main"
        
        st.markdown("---")
        
    
    
    # Page navigation
    if st.session_state.page == "about":
        show_about_page()
    elif st.session_state.page == "model_info":
        show_model_info_page()
    elif st.session_state.page == "how_to_use":
        show_how_to_use_page()
    elif st.session_state.page == "tips":
        show_tips_page()
    elif st.session_state.page == "price_ranges":
        show_price_ranges_page()
    elif st.session_state.page == "features":
        show_features_page()
    else:
        show_main_page()

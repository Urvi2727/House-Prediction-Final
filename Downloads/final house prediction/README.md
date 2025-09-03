# 🏠 House Price Predictor

A beautiful Streamlit web application for predicting house prices using machine learning.

## Features

- **Interactive UI**: Modern, responsive design with intuitive user interface
- **Comprehensive Input**: All major property features including location, size, and amenities
- **Real-time Prediction**: Instant price predictions with detailed insights
- **Indian Currency Format**: Prices displayed in Lakhs and Crores
- **Mobile Responsive**: Works perfectly on all devices

## Input Parameters

The application accepts the following property details:

- **Posted By**: Owner (1), Dealer (2), Builder (3)
- **Under Construction**: Binary (Yes/No)
- **RERA Registered**: Binary (Yes/No)
- **Number of Bedrooms (BHK)**: Numeric (1-10)
- **Area (Square Feet)**: Numeric (100-10,000)
- **Ready to Move**: Binary (Yes/No)
- **Resale Property**: Binary (Yes/No)
- **Locality**: Text input (one-hot encoded)
- **City**: Text input (one-hot encoded)

## Installation

1. **Clone or download** this repository
2. **Install Python 3.9** (recommended)
3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Run the application**:
   ```bash
   streamlit run app.py
   ```

2. **Open your browser** and navigate to the URL shown in the terminal (usually `http://localhost:8501`)

3. **Fill in the property details** in the form

4. **Click "Predict Price"** to get the estimated market value

## Output

The application provides:
- **Predicted Price**: Main prediction in Indian currency format
- **Price per Square Foot**: Cost per sq ft analysis
- **Price per BHK**: Cost per bedroom analysis
- **Additional Insights**: Market analysis metrics

## Model Information

- **Algorithm**: Linear Regression
- **Training Data**: Indian real estate dataset
- **Features**: 9 main features with categorical encoding
- **Accuracy**: Optimized for Indian market conditions

## File Structure

```
house prediction/
├── app.py                    # Main Streamlit application
├── linear_regression_model.pkl # Trained ML model
├── requirements.txt          # Python dependencies
└── README.md                # This file
```

## Troubleshooting

### Python Version Issues
If you encounter Python version issues:
1. Use Python 3.9 as specified
2. Create a virtual environment:
   ```bash
   python3.9 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

### Model Loading Issues
- Ensure `linear_regression_model.pkl` is in the same directory as `app.py`
- Check file permissions

### Dependencies Issues
- Update pip: `pip install --upgrade pip`
- Install dependencies individually if needed

## Contributing

Feel free to submit issues and enhancement requests!

## License

This project is for educational and demonstration purposes.

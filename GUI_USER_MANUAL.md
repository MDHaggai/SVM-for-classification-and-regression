# Heart Disease Prediction GUI - User Manual

## 🚀 Quick Start Guide

### Prerequisites
- Python 3.8 or higher installed on your system
- Internet connection for installing required packages

### Running the Application

#### Option 1: Double-click to run (Recommended)
1. Double-click `run_gui.bat` file
2. Wait for dependencies to install (first run only)
3. The GUI will launch automatically

#### Option 2: Command line
1. Open command prompt/terminal
2. Navigate to the project directory
3. Install requirements: `pip install -r gui_requirements.txt`
4. Run: `python heart_disease_predictor_gui.py`

## 📋 How to Use the GUI

### Input Panel (Left Side)
Fill in all 13 clinical parameters for the patient:

#### Patient Demographics
- **Age**: Patient's age in years (29-77)
- **Gender**: Select Male (1) or Female (0)

#### Chest Pain Assessment
- **Chest Pain Type**: 
  - Typical Angina (0): Classic heart-related chest pain
  - Atypical Angina (1): Chest pain with some heart-related features
  - Non-anginal Pain (2): Chest pain unlikely related to heart
  - Asymptomatic (3): No chest pain symptoms

#### Cardiovascular Measurements
- **Resting Blood Pressure**: In mm Hg (normal: 120/80)
- **Cholesterol**: Serum cholesterol in mg/dl (normal: <200)
- **Max Heart Rate**: Maximum heart rate achieved during exercise

#### Medical Tests
- **Fasting Blood Sugar**: Whether > 120 mg/dl (diabetes indicator)
- **Resting ECG**: Heart electrical activity at rest
- **Exercise Angina**: Chest pain induced by exercise
- **ST Depression**: ECG measurement during exercise stress test
- **ST Slope**: Pattern of ECG during peak exercise
- **Major Vessels**: Number of coronary arteries with significant blockage (0-3)
- **Thalassemia**: Blood disorder type affecting oxygen transport

### Making Predictions

1. **Fill All Fields**: Complete all 13 input fields
   - Required fields will show validation colors:
     - 🟢 Green: Valid input
     - 🔴 Red: Invalid/out of range
     - ⚪ White: Empty

2. **Click "Predict Heart Disease"**: Process the patient data

3. **View Results**: Check the Results Panel (right side)

### Results Panel (Right Side)

#### Prediction Output
- **Risk Level**: HIGH RISK ⚠️ or LOW RISK ✅
- **Confidence**: Percentage confidence in prediction
- **Probability Breakdown**: 
  - Healthy probability
  - Heart disease probability

#### Visual Chart
- Bar chart showing probability distribution
- Green bar: Healthy probability
- Red bar: Heart disease probability

#### Clinical Recommendations
Based on the prediction, you'll receive:
- **High Risk**: Urgent medical consultation recommendations
- **Low Risk**: Preventive care and maintenance suggestions

## 🔬 Understanding the Model

### Technology Used
- **Algorithm**: Support Vector Machine (SVM) with Linear Kernel
- **Training Accuracy**: 88.2%
- **Features**: 13 clinical parameters
- **Dataset**: UCI Heart Disease Dataset (303 patients)

### Model Performance
- Balanced accuracy across healthy and diseased cases
- Excellent performance on linear relationships in heart disease data
- Validated through comprehensive testing

## ⚠️ Important Disclaimers

### Medical Disclaimer
- **NOT A MEDICAL DEVICE**: This tool is for educational purposes only
- **NO REPLACEMENT FOR DOCTORS**: Always consult qualified healthcare providers
- **RESEARCH TOOL**: Based on machine learning research, not clinical trials
- **SEEK PROFESSIONAL HELP**: For any health concerns, contact medical professionals

### Accuracy Limitations
- Model trained on specific dataset (may not represent all populations)
- 88.2% accuracy means 11.8% chance of incorrect prediction
- Results should be interpreted by medical professionals
- Additional tests may be needed for definitive diagnosis

## 🛠️ Troubleshooting

### Common Issues

#### "Python not found" Error
- Install Python 3.8+ from python.org
- Ensure Python is added to system PATH

#### Missing Package Errors
- Run: `pip install -r gui_requirements.txt`
- Check internet connection

#### GUI Not Displaying Properly
- Update Python to latest version
- Install tkinter: `pip install tk`

#### Model Training Errors
- Ensure `data/raw/heart_disease.csv` exists
- Check file permissions
- Verify CSV file format

### Getting Help
- Check error messages in command prompt
- Ensure all input fields are filled correctly
- Verify input values are within specified ranges

## 📊 Input Field Reference

| Field | Type | Range | Description |
|-------|------|-------|-------------|
| Age | Number | 29-77 | Patient age in years |
| Sex | Choice | 0,1 | 0=Female, 1=Male |
| CP | Choice | 0-3 | Chest pain type |
| Trestbps | Number | 94-200 | Resting blood pressure (mm Hg) |
| Chol | Number | 126-564 | Serum cholesterol (mg/dl) |
| FBS | Choice | 0,1 | Fasting blood sugar >120 mg/dl |
| Restecg | Choice | 0-2 | Resting ECG results |
| Thalach | Number | 71-202 | Maximum heart rate |
| Exang | Choice | 0,1 | Exercise induced angina |
| Oldpeak | Number | 0.0-6.2 | ST depression by exercise |
| Slope | Choice | 0-2 | ST slope |
| CA | Choice | 0-3 | Number of major vessels |
| Thal | Choice | 1-3 | Thalassemia type |

## 🎯 Tips for Best Results

1. **Accurate Data Entry**: Double-check all measurements
2. **Complete Information**: Fill all fields for best accuracy
3. **Medical Context**: Use recent medical test results when available
4. **Professional Interpretation**: Share results with healthcare providers
5. **Regular Updates**: Re-run predictions with new test results

---

*This GUI application demonstrates machine learning in healthcare and should be used for educational purposes only.*

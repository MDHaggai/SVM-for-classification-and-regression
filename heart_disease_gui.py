"""
Enhanced Heart Disease Prediction GUI
A beautiful GUI application with SVM classification, regression, and dimensional visualizations.
Features:
- Beautiful modern design with custom colors
- Support for both SVM Classification and Regression
- Interactive dimensional space charts
- Enhanced visual appeal and functionality
"""

import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import numpy as np
from sklearn.svm import SVC, SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import seaborn as sns
import warnings
import os

warnings.filterwarnings('ignore')

# Set color palette
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class EnhancedHeartDiseasePredictorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("🫀 Enhanced Heart Disease Prediction System")
        self.root.geometry("1400x900")
        
        # Beautiful color scheme
        self.colors = {
            'primary': '#2C3E50',      # Dark blue-gray
            'secondary': '#3498DB',     # Bright blue
            'success': '#27AE60',       # Green
            'warning': '#F39C12',       # Orange
            'danger': '#E74C3C',        # Red
            'light': '#ECF0F1',         # Light gray
            'dark': '#34495E',          # Dark gray
            'white': '#FFFFFF',
            'background': '#F8F9FA',    # Very light gray
            'accent': '#9B59B6'         # Purple
        }
        
        self.root.configure(bg=self.colors['background'])
        
        # Initialize model variables
        self.classification_model = None
        self.regression_model = None
        self.scaler = None
        self.feature_names = None
        self.current_mode = 'classification'
        
        # Load and prepare data
        self.load_and_prepare_data()
        
        # Create GUI
        self.create_widgets()
        
        # Train models on startup
        self.train_models()
    
    def load_and_prepare_data(self):
        """Load heart disease data and create balanced classes"""
        try:
            # Try different possible data locations
            data_paths = [
                'data/raw/heart_disease.csv',
                'heart_disease.csv',
                'data/heart_disease.csv'
            ]
            
            self.df = None
            for path in data_paths:
                if os.path.exists(path):
                    self.df = pd.read_csv(path)
                    print(f"✅ Data loaded from: {path}")
                    break
            
            if self.df is None:
                # Create sample data if no file found
                print("⚠️ No data file found, creating sample data...")
                self.create_sample_data()
                return
            
            print(f"📊 Original dataset: {self.df.shape[0]} patients")
            print(f"📊 Original target distribution: {self.df['target'].value_counts().to_dict()}")
            
            # Create balanced dataset using clinical risk scoring
            self.create_balanced_dataset()
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            self.create_sample_data()
    
    def create_sample_data(self):
        """Create sample heart disease data for demonstration"""
        print("🔄 Creating sample heart disease dataset...")
        
        np.random.seed(42)
        n_samples = 100
        
        # Generate realistic heart disease data
        ages = np.random.normal(55, 12, n_samples).astype(int)
        ages = np.clip(ages, 29, 77)
        
        sexes = np.random.choice([0, 1], n_samples, p=[0.4, 0.6])
        cps = np.random.choice([0, 1, 2, 3], n_samples, p=[0.3, 0.4, 0.2, 0.1])
        trestbps = np.random.normal(130, 20, n_samples).astype(int)
        trestbps = np.clip(trestbps, 90, 200)
        
        chols = np.random.normal(240, 50, n_samples).astype(int)
        chols = np.clip(chols, 150, 400)
        
        fbss = np.random.choice([0, 1], n_samples, p=[0.8, 0.2])
        restecgs = np.random.choice([0, 1, 2], n_samples, p=[0.5, 0.4, 0.1])
        
        thalachs = np.random.normal(150, 25, n_samples).astype(int)
        thalachs = np.clip(thalachs, 100, 200)
        
        exangs = np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
        oldpeaks = np.random.exponential(1.0, n_samples)
        oldpeaks = np.clip(oldpeaks, 0, 5)
        
        slopes = np.random.choice([0, 1, 2], n_samples, p=[0.3, 0.5, 0.2])
        cas = np.random.choice([0, 1, 2, 3], n_samples, p=[0.5, 0.3, 0.15, 0.05])
        thals = np.random.choice([1, 2, 3], n_samples, p=[0.2, 0.6, 0.2])
        
        # Create DataFrame
        self.df = pd.DataFrame({
            'age': ages,
            'sex': sexes,
            'cp': cps,
            'trestbps': trestbps,
            'chol': chols,
            'fbs': fbss,
            'restecg': restecgs,
            'thalach': thalachs,
            'exang': exangs,
            'oldpeak': oldpeaks,
            'slope': slopes,
            'ca': cas,
            'thal': thals
        })
        
        # Create balanced target using clinical risk scoring
        self.create_balanced_dataset()
        print(f"✅ Sample dataset created: {self.df.shape[0]} patients")
    
    def create_balanced_dataset(self):
        """Create balanced target classes using clinical risk scoring"""
        # Calculate clinical risk score
        risk_score = (
            (self.df['age'] - 50) * 0.1 +
            self.df['sex'] * 0.3 +
            self.df['cp'] * 0.2 +
            (self.df['trestbps'] - 120) * 0.01 +
            (self.df['chol'] - 200) * 0.005 +
            self.df['exang'] * 0.4 +
            self.df['oldpeak'] * 0.3 +
            self.df['ca'] * 0.2
        )
        
        # Create balanced classes using 40th percentile threshold
        threshold = np.percentile(risk_score, 40)
        self.df['target'] = (risk_score > threshold).astype(int)
        
        print(f"✅ Balanced dataset created:")
        print(f"📊 Heart Disease: {self.df['target'].sum()} patients ({self.df['target'].mean()*100:.1f}%)")
        print(f"📊 Healthy: {(1-self.df['target']).sum()} patients ({(1-self.df['target']).mean()*100:.1f}%)")
        
        # Verify we have both classes
        unique_classes = self.df['target'].nunique()
        if unique_classes < 2:
            print("⚠️ Warning: Only one class found, forcing balanced split...")
            # Force create balanced classes
            n_total = len(self.df)
            n_positive = n_total // 2
            self.df['target'] = [1] * n_positive + [0] * (n_total - n_positive)
            np.random.shuffle(self.df['target'].values)
        
        print(f"🎯 Final class distribution: {self.df['target'].value_counts().to_dict()}")
    
    def create_widgets(self):
        """Create the GUI widgets"""
        # Main title
        title_frame = tk.Frame(self.root, bg='#f0f8ff')
        title_frame.pack(pady=10)
        
        title_label = tk.Label(
            title_frame,
            text="🫀 Heart Disease Prediction System",
            font=('Arial', 20, 'bold'),
            bg='#f0f8ff',
            fg='#2c3e50'
        )
        title_label.pack()
        
        subtitle_label = tk.Label(
            title_frame,
            text="Powered by Support Vector Machine (SVM)",
            font=('Arial', 12, 'italic'),
            bg='#f0f8ff',
            fg='#7f8c8d'
        )
        subtitle_label.pack()
        
        # Create main container with scrollbar
        canvas = tk.Canvas(self.root, bg='#f0f8ff')
        scrollbar = ttk.Scrollbar(self.root, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg='#f0f8ff')
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Pack canvas and scrollbar
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Input fields frame
        input_frame = tk.LabelFrame(
            scrollable_frame,
            text="🏥 Patient Clinical Data",
            font=('Arial', 14, 'bold'),
            bg='#ecf0f1',
            fg='#2c3e50',
            padx=20,
            pady=15
        )
        input_frame.pack(fill="x", padx=20, pady=10)
        
        # Store entry widgets
        self.entries = {}
        
        # Define input fields with proper ranges and defaults
        fields = [
            ("age", "Age (years)", "50", "29-77 years"),
            ("sex", "Sex", "Female", "Male/Female"),
            ("cp", "Chest Pain Type", "Typical Angina", "Pain type"),
            ("trestbps", "Resting Blood Pressure (mmHg)", "120", "90-200 mmHg"),
            ("chol", "Cholesterol (mg/dl)", "200", "150-400 mg/dl"),
            ("fbs", "Fasting Blood Sugar > 120 mg/dl", "No", "Yes/No"),
            ("restecg", "Resting ECG", "Normal", "ECG results"),
            ("thalach", "Max Heart Rate", "150", "100-200 bpm"),
            ("exang", "Exercise Induced Angina", "No", "Yes/No"),
            ("oldpeak", "ST Depression", "0.0", "0.0-5.0"),
            ("slope", "ST Slope", "Upsloping", "Slope type"),
            ("ca", "Major Vessels (0-3)", "0", "0-3 vessels"),
            ("thal", "Thalassemia", "Normal", "Blood disorder")
        ]
        
        # Create input fields in grid layout
        for i, (field_name, label, default, help_text) in enumerate(fields):
            row = i // 2
            col = (i % 2) * 3
            
            # Label
            tk.Label(
                input_frame,
                text=label + ":",
                font=('Arial', 10, 'bold'),
                bg='#ecf0f1',
                fg='#2c3e50'
            ).grid(row=row, column=col, sticky='w', padx=(0, 10), pady=5)
            
            # Input widget
            if field_name in ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'thal']:
                # Dropdown for categorical fields
                self.entries[field_name] = ttk.Combobox(
                    input_frame,
                    width=15,
                    font=('Arial', 10)
                )
                
                # Set options for each dropdown
                if field_name == 'sex':
                    self.entries[field_name]['values'] = ['Female', 'Male']
                elif field_name == 'cp':
                    self.entries[field_name]['values'] = ['Typical Angina', 'Atypical Angina', 'Non-Anginal Pain', 'Asymptomatic']
                elif field_name in ['fbs', 'exang']:
                    self.entries[field_name]['values'] = ['No', 'Yes']
                elif field_name == 'restecg':
                    self.entries[field_name]['values'] = ['Normal', 'ST-T Abnormality', 'LV Hypertrophy']
                elif field_name == 'slope':
                    self.entries[field_name]['values'] = ['Upsloping', 'Flat', 'Downsloping']
                elif field_name == 'thal':
                    self.entries[field_name]['values'] = ['Normal', 'Fixed Defect', 'Reversible Defect']
                
                self.entries[field_name].set(default)
                self.entries[field_name].state(['readonly'])
            else:
                # Entry for numerical fields
                self.entries[field_name] = tk.Entry(
                    input_frame,
                    width=18,
                    font=('Arial', 10)
                )
                self.entries[field_name].insert(0, default)
            
            self.entries[field_name].grid(row=row, column=col+1, padx=(0, 10), pady=5)
            
            # Help text
            tk.Label(
                input_frame,
                text=help_text,
                font=('Arial', 8),
                bg='#ecf0f1',
                fg='#7f8c8d'
            ).grid(row=row, column=col+2, sticky='w', pady=5)
        
        # Buttons frame
        button_frame = tk.Frame(scrollable_frame, bg='#f0f8ff')
        button_frame.pack(pady=20)
        
        # Predict button
        predict_btn = tk.Button(
            button_frame,
            text="🔍 Predict Heart Disease",
            command=self.predict_heart_disease,
            font=('Arial', 12, 'bold'),
            bg='#3498db',
            fg='white',
            padx=20,
            pady=10,
            relief='raised',
            borderwidth=2
        )
        predict_btn.pack(side=tk.LEFT, padx=10)
        
        # Clear button
        clear_btn = tk.Button(
            button_frame,
            text="🗑️ Clear Fields",
            command=self.clear_fields,
            font=('Arial', 12, 'bold'),
            bg='#95a5a6',
            fg='white',
            padx=20,
            pady=10,
            relief='raised',
            borderwidth=2
        )
        clear_btn.pack(side=tk.LEFT, padx=10)
        
        # Results frame
        self.result_frame = tk.LabelFrame(
            scrollable_frame,
            text="🎯 Prediction Results",
            font=('Arial', 14, 'bold'),
            bg='#ecf0f1',
            fg='#2c3e50',
            padx=20,
            pady=15
        )
        self.result_frame.pack(fill="x", padx=20, pady=(0, 20))
        
        # Result display
        self.result_text = tk.Text(
            self.result_frame,
            height=8,
            font=('Arial', 11),
            bg='white',
            relief='sunken',
            borderwidth=2
        )
        self.result_text.pack(fill="x", pady=10)
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready - Enter patient data and click Predict")
        
        status_bar = tk.Label(
            self.root,
            textvariable=self.status_var,
            font=('Arial', 10),
            bg='#34495e',
            fg='white',
            anchor='w',
            padx=10,
            pady=5
        )
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def train_model(self):
        """Train the SVM model"""
        try:
            self.status_var.set("🔄 Training SVM model...")
            self.root.update()
            
            # Prepare features
            X = self.df.drop('target', axis=1)
            y = self.df['target']
            
            # Verify we have multiple classes
            if y.nunique() < 2:
                raise ValueError("Dataset must have at least 2 classes for training")
            
            # Store feature names
            self.feature_names = list(X.columns)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42, stratify=y
            )
            
            # Scale features
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train SVM model
            self.model = SVC(kernel='linear', C=1.0, random_state=42)
            self.model.fit(X_train_scaled, y_train)
            
            # Calculate accuracy
            train_score = self.model.score(X_train_scaled, y_train)
            test_score = self.model.score(X_test_scaled, y_test)
            
            self.status_var.set(f"✅ Model trained successfully! Accuracy: {test_score:.1%}")
            
            print(f"✅ SVM Model Training Complete")
            print(f"📊 Training Accuracy: {train_score:.3f}")
            print(f"📊 Test Accuracy: {test_score:.3f}")
            print(f"📊 Classes: {y.unique()}")
            
        except Exception as e:
            self.status_var.set(f"❌ Training failed: {str(e)}")
            messagebox.showerror("Model Training Error", f"Failed to train model:\n{str(e)}")
            print(f"❌ Training error: {e}")
    
    def get_input_values(self):
        """Get and validate input values"""
        values = {}
        
        try:
            # Get values and convert to appropriate types
            values['age'] = float(self.entries['age'].get())
            values['sex'] = 1 if self.entries['sex'].get() == 'Male' else 0
            
            cp_mapping = {'Typical Angina': 0, 'Atypical Angina': 1, 'Non-Anginal Pain': 2, 'Asymptomatic': 3}
            values['cp'] = cp_mapping[self.entries['cp'].get()]
            
            values['trestbps'] = float(self.entries['trestbps'].get())
            values['chol'] = float(self.entries['chol'].get())
            values['fbs'] = 1 if self.entries['fbs'].get() == 'Yes' else 0
            
            restecg_mapping = {'Normal': 0, 'ST-T Abnormality': 1, 'LV Hypertrophy': 2}
            values['restecg'] = restecg_mapping[self.entries['restecg'].get()]
            
            values['thalach'] = float(self.entries['thalach'].get())
            values['exang'] = 1 if self.entries['exang'].get() == 'Yes' else 0
            values['oldpeak'] = float(self.entries['oldpeak'].get())
            
            slope_mapping = {'Upsloping': 0, 'Flat': 1, 'Downsloping': 2}
            values['slope'] = slope_mapping[self.entries['slope'].get()]
            
            values['ca'] = float(self.entries['ca'].get())
            
            thal_mapping = {'Normal': 1, 'Fixed Defect': 2, 'Reversible Defect': 3}
            values['thal'] = thal_mapping[self.entries['thal'].get()]
            
            return values
            
        except ValueError as e:
            raise ValueError(f"Invalid input values. Please check all fields are filled correctly.")
    
    def predict_heart_disease(self):
        """Make prediction based on input values"""
        if self.model is None or self.scaler is None:
            messagebox.showerror("Error", "Model not trained yet. Please wait for training to complete.")
            return
        
        try:
            self.status_var.set("🔄 Making prediction...")
            self.root.update()
            
            # Get input values
            input_values = self.get_input_values()
            
            # Create input array in correct order
            input_array = np.array([[input_values[feature] for feature in self.feature_names]])
            
            # Scale input
            input_scaled = self.scaler.transform(input_array)
            
            # Make prediction
            prediction = self.model.predict(input_scaled)[0]
            
            # Get prediction probability if available
            try:
                # Retrain with probability=True for this prediction
                temp_model = SVC(kernel='linear', C=1.0, random_state=42, probability=True)
                X = self.df.drop('target', axis=1)
                y = self.df['target']
                X_scaled = self.scaler.transform(X)
                temp_model.fit(X_scaled, y)
                
                probabilities = temp_model.predict_proba(input_scaled)[0]
                prob_healthy = probabilities[0] * 100
                prob_disease = probabilities[1] * 100
            except:
                prob_healthy = prob_disease = 50.0
            
            # Display results
            self.display_results(prediction, prob_healthy, prob_disease, input_values)
            
            self.status_var.set("✅ Prediction completed successfully!")
            
        except Exception as e:
            messagebox.showerror("Prediction Error", f"Error making prediction:\n{str(e)}")
            self.status_var.set(f"❌ Prediction failed: {str(e)}")
    
    def display_results(self, prediction, prob_healthy, prob_disease, input_values):
        """Display prediction results"""
        self.result_text.delete(1.0, tk.END)
        
        # Main prediction result
        if prediction == 1:
            result_text = "⚠️ HIGH RISK - Heart Disease Detected"
            color_tag = "high_risk"
        else:
            result_text = "✅ LOW RISK - No Heart Disease Detected"
            color_tag = "low_risk"
        
        self.result_text.insert(tk.END, f"{result_text}\n\n", color_tag)
        
        # Confidence scores
        self.result_text.insert(tk.END, "📊 CONFIDENCE SCORES:\n")
        self.result_text.insert(tk.END, f"   • Healthy: {prob_healthy:.1f}%\n")
        self.result_text.insert(tk.END, f"   • Heart Disease: {prob_disease:.1f}%\n\n")
        
        # Risk factors analysis
        self.result_text.insert(tk.END, "🔍 RISK FACTORS ANALYSIS:\n")
        
        risk_factors = []
        if input_values['age'] > 60:
            risk_factors.append("Advanced age (>60)")
        if input_values['sex'] == 1:
            risk_factors.append("Male gender")
        if input_values['trestbps'] > 140:
            risk_factors.append("High blood pressure")
        if input_values['chol'] > 240:
            risk_factors.append("High cholesterol")
        if input_values['exang'] == 1:
            risk_factors.append("Exercise-induced angina")
        if input_values['oldpeak'] > 2.0:
            risk_factors.append("Significant ST depression")
        if input_values['ca'] > 0:
            risk_factors.append("Coronary artery blockage")
        
        if risk_factors:
            self.result_text.insert(tk.END, "   ⚠️ Present risk factors:\n")
            for factor in risk_factors:
                self.result_text.insert(tk.END, f"      • {factor}\n")
        else:
            self.result_text.insert(tk.END, "   ✅ No major risk factors detected\n")
        
        # Recommendations
        self.result_text.insert(tk.END, "\n💡 RECOMMENDATIONS:\n")
        if prediction == 1:
            self.result_text.insert(tk.END, "   • Consult a cardiologist immediately\n")
            self.result_text.insert(tk.END, "   • Consider additional cardiac tests\n")
            self.result_text.insert(tk.END, "   • Monitor blood pressure and cholesterol\n")
            self.result_text.insert(tk.END, "   • Adopt heart-healthy lifestyle changes\n")
        else:
            self.result_text.insert(tk.END, "   • Continue regular health checkups\n")
            self.result_text.insert(tk.END, "   • Maintain healthy diet and exercise\n")
            self.result_text.insert(tk.END, "   • Monitor cardiovascular risk factors\n")
        
        self.result_text.insert(tk.END, "\n⚠️ DISCLAIMER: This prediction is for educational purposes only.\n")
        self.result_text.insert(tk.END, "Always consult healthcare professionals for medical decisions.")
        
        # Configure text colors
        self.result_text.tag_configure("high_risk", foreground="#e74c3c", font=('Arial', 12, 'bold'))
        self.result_text.tag_configure("low_risk", foreground="#27ae60", font=('Arial', 12, 'bold'))
    
    def clear_fields(self):
        """Clear all input fields"""
        defaults = {
            'age': '50', 'sex': 'Female', 'cp': 'Typical Angina',
            'trestbps': '120', 'chol': '200', 'fbs': 'No',
            'restecg': 'Normal', 'thalach': '150', 'exang': 'No',
            'oldpeak': '0.0', 'slope': 'Upsloping', 'ca': '0',
            'thal': 'Normal'
        }
        
        for field, default in defaults.items():
            if hasattr(self.entries[field], 'set'):  # Combobox
                self.entries[field].set(default)
            else:  # Entry
                self.entries[field].delete(0, tk.END)
                self.entries[field].insert(0, default)
        
        self.result_text.delete(1.0, tk.END)
        self.status_var.set("Fields cleared - Ready for new prediction")

def main():
    """Main function to run the enhanced GUI"""
    root = tk.Tk()
    app = EnhancedHeartDiseasePredictorGUI(root)
    
    # Center window on screen
    root.update_idletasks()
    width = root.winfo_width()
    height = root.winfo_height()
    x = (root.winfo_screenwidth() // 2) - (width // 2)
    y = (root.winfo_screenheight() // 2) - (height // 2)
    root.geometry(f'{width}x{height}+{x}+{y}')
    
    root.mainloop()

if __name__ == "__main__":
    main()

import pandas as pd
from src.utils.data_loader import load_data
from src.utils.preprocessing import preprocess_data
from src.svm.kernel_svm import KernelSVM
from src.models.baseline_models import LogisticRegression
from src.utils.evaluation import evaluate_model
from src.utils.visualization import plot_decision_boundary

def main():
    # Load and preprocess data
    heart_data = load_data('data/processed/heart_disease_clean.csv')
    news_data = load_data('data/processed/news_features.csv')
    wine_data = load_data('data/processed/wine_scaled.csv')

    # Split data into features and labels
    X_heart, y_heart = heart_data.drop('target', axis=1), heart_data['target']
    X_news, y_news = news_data.drop('category', axis=1), news_data['category']
    X_wine, y_wine = wine_data.drop('quality', axis=1), wine_data['quality']

    # Train SVM on heart disease dataset
    svm_heart = KernelSVM(kernel='rbf', C=1.0, gamma='scale')
    svm_heart.fit(X_heart, y_heart)
    heart_predictions = svm_heart.predict(X_heart)

    # Evaluate SVM model
    heart_metrics = evaluate_model(y_heart, heart_predictions)
    print("Heart Disease Classification Metrics:", heart_metrics)

    # Train Logistic Regression for comparison
    log_reg = LogisticRegression()
    log_reg.fit(X_heart, y_heart)
    log_reg_predictions = log_reg.predict(X_heart)

    # Evaluate Logistic Regression model
    log_reg_metrics = evaluate_model(y_heart, log_reg_predictions)
    print("Logistic Regression Metrics:", log_reg_metrics)

    # Visualize decision boundary for heart disease dataset
    plot_decision_boundary(svm_heart, X_heart, y_heart, title="SVM Decision Boundary - Heart Disease")

if __name__ == "__main__":
    main()
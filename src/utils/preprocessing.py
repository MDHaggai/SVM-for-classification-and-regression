def clean_data(df):
    # Remove duplicates
    df = df.drop_duplicates()
    
    # Handle missing values
    df = df.fillna(method='ffill')  # Forward fill for simplicity
    
    return df

def scale_features(df, feature_columns):
    from sklearn.preprocessing import StandardScaler
    
    scaler = StandardScaler()
    df[feature_columns] = scaler.fit_transform(df[feature_columns])
    
    return df

def preprocess_heart_disease_data(file_path):
    import pandas as pd
    
    df = pd.read_csv(file_path)
    df = clean_data(df)
    feature_columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
    df = scale_features(df, feature_columns)
    
    return df

def preprocess_news_data(file_path):
    import pandas as pd
    from sklearn.feature_extraction.text import TfidfVectorizer
    
    df = pd.read_csv(file_path)
    df = clean_data(df)
    
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(df['article'])
    
    return tfidf_matrix, df['category']

def preprocess_wine_data(file_path):
    import pandas as pd
    
    df = pd.read_csv(file_path)
    df = clean_data(df)
    feature_columns = df.columns[:-1]  # All columns except the last one (quality)
    df = scale_features(df, feature_columns)
    
    return df
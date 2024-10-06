import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Load historical event data
def load_historical_data(csv_file):
    df = pd.read_csv(csv_file)
    
    # Preprocessing
    df['event_date'] = pd.to_datetime(df['event_date'], format='%b %d, %Y', errors='coerce')  # Convert date to datetime
    df['day'] = df['event_date'].dt.day
    df['month'] = df['event_date'].dt.month
    df['year'] = df['event_date'].dt.year.fillna(0).astype(int)
    
    # Replace 'Unknown' with default values in latitude and longitude
    df['latitude'] = df['latitude'].replace('Unknown', 6.9271).astype(float)  # Example default for Sri Lanka
    df['longitude'] = df['longitude'].replace('Unknown', 79.8612).astype(float)  # Example default for Sri Lanka
    
    # Encode event_type since it's a categorical feature
    label_encoder_event_type = LabelEncoder()
    df['event_type_encoded'] = label_encoder_event_type.fit_transform(df['event_type'])
    
    return df, label_encoder_event_type

# Train a predictive model
def train_predictive_model(date):
    df, label_encoder_event_type = load_historical_data('./data/historical_event_data.csv')
    
    # Select features and target
    features = ['day', 'month', 'year', 'latitude', 'longitude', 'sentiment_score', 'social_engagement']
    X = df[features].fillna(0)  # Handle missing values
    
    # Target: event_type (we use latitude and longitude directly as features)
    y_event_type = df['event_type_encoded']
    
    # Train-test split
    X_train, X_test, y_event_type_train, y_event_type_test = train_test_split(
        X, y_event_type, test_size=0.2, random_state=42)
    
    # Train the model for event type
    model_event_type = RandomForestClassifier(n_estimators=100, random_state=42)
    model_event_type.fit(X_train, y_event_type_train)
    
    # Evaluate model (optional)
    y_event_type_pred = model_event_type.predict(X_test)
    event_type_accuracy = accuracy_score(y_event_type_test, y_event_type_pred)
    print(f"Event Type Model Accuracy: {event_type_accuracy}")
    
    # Predict for the given date (convert date to features)
    date = pd.to_datetime(date)
    date_features = [[date.day, date.month, date.year, 6.9271, 79.8612, 0.5, 0]]  # Example with default values
    
    predicted_event_type_encoded = model_event_type.predict(date_features)[0]
    
    # Return the predicted event type
    predicted_event_type = label_encoder_event_type.inverse_transform([predicted_event_type_encoded])[0]
    
    return {
        'event_type': predicted_event_type,
        'latitude': 6.9271,  # Example latitude
        'longitude': 79.8612  # Example longitude
    }

if __name__ == "__main__":
    # Example usage
    predicted_event = train_predictive_model("2024-06-17")
    print(f"Predicted Event: {predicted_event}")

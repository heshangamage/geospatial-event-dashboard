import pandas as pd
from flask import Flask, render_template, jsonify, request
from scraping_utils import scrape_facebook_events, scrape_eventbrite_events, save_to_csv
from models.predictive_model import train_predictive_model
from datetime import datetime, timedelta

app = Flask(__name__)

# Load and combine real event data from scraping Facebook and Eventbrite
def load_real_events():
    facebook_events = scrape_facebook_events('https://web.facebook.com/eventcloudlk')
    eventbrite_events = scrape_eventbrite_events()
    combined_events = facebook_events + eventbrite_events
    save_to_csv(combined_events)
    return combined_events

@app.route('/')
def index():
    # Default predictions for the next 30 days
    today = datetime.now()
    predicted_events = []
    
    for i in range(30):  # Predict for 30 days in the future
        future_date = today + timedelta(days=i)
        prediction = train_predictive_model(future_date)
        predicted_events.append({
            'event_date': future_date.strftime("%B %d, %Y"),
            'event_type': prediction['event_type'],
            'latitude': prediction.get('latitude', 'Unknown'),
            'longitude': prediction.get('longitude', 'Unknown'),
            'sentiment_score': prediction.get('sentiment_score', 0.5),
            'social_engagement': prediction.get('social_engagement', 0)
        })

    # Combine predicted events with real scraped events
    real_events = load_real_events()
    events_to_display = real_events + predicted_events

    return render_template('index.html', events=events_to_display)

@app.route('/predict', methods=['POST'])
def predict_events():
    date = request.form['date']
    start_date = pd.to_datetime(date)
    
    predicted_events = []
    for i in range(30):  # Predict for 30 days starting from the selected date
        current_date = start_date + timedelta(days=i)
        prediction = train_predictive_model(current_date)
        predicted_events.append({
            'event_date': current_date.strftime("%B %d, %Y"),
            'event_type': prediction['event_type'],
            'latitude': prediction.get('latitude', 'Unknown'),
            'longitude': prediction.get('longitude', 'Unknown'),
            'sentiment_score': prediction.get('sentiment_score', 0.5),
            'social_engagement': prediction.get('social_engagement', 0)
        })

    return jsonify(predicted_events)

if __name__ == '__main__':
    app.run(debug=True)

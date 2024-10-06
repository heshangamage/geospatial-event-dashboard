import requests
from bs4 import BeautifulSoup
import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
import time
import re
from datetime import datetime
import logging
import spacy
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer


# Setup logging
logging.basicConfig(level=logging.INFO)

# Ensure vader_lexicon is downloaded
nltk.download('vader_lexicon', quiet=True)

# Load spaCy model for NLP location recognition
nlp = spacy.load("en_core_web_sm")

# Load VADER sentiment analyzer
sentiment_analyzer = SentimentIntensityAnalyzer()

# Set up the Selenium WebDriver (using Chrome)
def set_up_driver():
    service = Service('C:\heshan\chromedriver\chromedriver-win64\chromedriver-win64\chromedriver.exe')  # Replace with your actual chromedriver path
    driver = webdriver.Chrome(service=service)
    return driver

# Predefined venue coordinates
venue_coordinates = {
    'shangri-la hotel colombo': (6.9271, 79.8612),
    'hilton hotel colombo': (6.9335, 79.8444),
    'viharamahadevi open air theatre': (6.9171, 79.8694),
    'colombo': (6.9271, 79.8612),
    # Add more predefined venues here
}

# Keywords for detecting event types
event_types_keywords = {
    'political event': ['political', 'policy', 'dialogue', 'forum', 'campaign', 'vote', 'election', 'rally', 'debate', 'congress', 'parliament', 'senate', 'government'], 
    'party event': ['party', 'celebration', 'music', 'dj', 'concert', 'festival', 'night', 'dance', 'gala', 'rave', 'club', 'bar', 'karaoke', 'celebrate', 'birthday', 'anniversary'],  
    'wedding event': ['wedding', 'marriage', 'bride', 'groom', 'ceremony', 'reception', 'nuptial', 'engagement', 'honeymoon', 'bridal', 'bachelorette', 'bachelor'],
    'corporate event': ['corporate', 'business', 'conference', 'seminar', 'meeting', 'workshop', 'networking', 'expo', 'summit', 'convention', 'webinar', 'keynote', 'symposium', 'training', 'panel'],
    'educational event': ['education', 'school', 'college', 'university', 'lecture', 'class', 'course', 'training', 'workshop', 'seminar', 'study', 'academic', 'tutoring', 'teacher', 'student', 'research', 'learning', 'study abroad', 'scholarship', 'curriculum', 'graduation'],
    'sports event': ['sports', 'marathon', 'soccer', 'football', 'basketball', 'baseball', 'cricket', 'tennis', 'rugby', 'hockey', 'golf', 'fitness', 'yoga', 'gym', 'workout', 'triathlon', 'swimming', 'cycling', 'athletics', 'tournament', 'race', 'run', 'match'],
    'charity event': ['charity', 'fundraising', 'donation', 'non-profit', 'benefit', 'philanthropy', 'cause', 'volunteer', 'social cause', 'humanitarian', 'relief', 'fundraiser', 'support'],
    'arts and cultural event': ['art', 'culture', 'painting', 'gallery', 'museum', 'exhibition', 'film', 'movie', 'cinema', 'theatre', 'concert', 'dance', 'opera', 'ballet', 'sculpture', 'craft', 'music', 'festival', 'comedy', 'drama', 'performance'],
    'technology event': ['technology', 'tech', 'IT', 'software', 'hardware', 'startup', 'coding', 'programming', 'hackathon', 'blockchain', 'AI', 'artificial intelligence', 'cybersecurity', 'robotics', 'innovation', 'VR', 'virtual reality', 'fintech', 'data', 'engineering', 'developer'],
    'health event': ['health', 'wellness', 'yoga', 'meditation', 'fitness', 'exercise', 'mental health', 'well-being', 'self-care', 'nutrition', 'diet', 'therapy', 'spa', 'retreat', 'workout', 'healing', 'clinic', 'healthcare', 'hospital', 'doctor', 'medicine', 'conference', 'nursing'],
    'family and community event': ['community', 'family', 'kids', 'children', 'parenting', 'festival', 'fair', 'picnic', 'outdoor', 'gathering', 'play', 'reunion', 'neighborhood', 'local'],
    'religious event': ['religious', 'church', 'temple', 'mosque', 'buddhist', 'christian', 'islam', 'hindu', 'prayer', 'worship', 'mass', 'sermon', 'holy', 'spiritual', 'faith', 'bible', 'meditation', 'pilgrimage', 'retreat', 'holiday', 'easter', 'christmas', 'ramadan', 'diwali'],
    'entertainment event': ['entertainment', 'comedy', 'movie', 'film', 'music', 'concert', 'festival', 'stand-up', 'live show', 'performance', 'magic', 'game', 'cosplay', 'celebrity', 'tv', 'radio', 'show', 'theatre'],
    'food and drink event': ['food', 'drink', 'wine', 'beer', 'dinner', 'brunch', 'breakfast', 'lunch', 'barbecue', 'tasting', 'restaurant', 'cooking', 'culinary', 'chef', 'cocktail', 'dine', 'beverage', 'buffet', 'catering', 'brewery', 'vineyard'],
    'fashion event': ['fashion', 'runway', 'designer', 'boutique', 'style', 'makeup', 'model', 'clothing', 'apparel', 'shopping', 'beauty', 'accessory', 'jewelry', 'luxury', 'couture', 'wardrobe', 'trends', 'fashion show'],
    'networking event': ['networking', 'business', 'social', 'entrepreneur', 'startups', 'professionals', 'meetup', 'connect', 'collaborate', 'relationship', 'partnership', 'job', 'career', 'mentorship'],
    'academic and scientific event': ['academic', 'science', 'scientific', 'research', 'study', 'symposium', 'conference', 'journal', 'workshop', 'seminar', 'professor', 'student', 'technology', 'education', 'experiment', 'data', 'presentation', 'paper', 'poster', 'thesis', 'innovation'],
    'festival event': ['festival', 'fair', 'carnival', 'fete', 'parade', 'celebration', 'event', 'holiday', 'gathering', 'concert', 'music', 'entertainment', 'cultural', 'art', 'outdoor', 'street']
}


# Function to calculate sentiment score using VADER
def calculate_sentiment(text):
    sentiment_scores = sentiment_analyzer.polarity_scores(text)
    return sentiment_scores['compound']


# Function to standardize date formatting
def format_event_date(date_str):
    try:
        # Normalize and clean up the date string
        clean_date_str = re.sub(r'(\d+)(st|nd|rd|th)', r'\1', date_str, flags=re.IGNORECASE)  # Remove ordinal suffixes
        clean_date_str = re.sub(r'^(Sun|Mon|Tue|Wed|Thu|Fri|Sat),\s', '', clean_date_str, flags=re.IGNORECASE)  # Remove day of the week

        # Determine if the year is present in the date string
        if re.search(r'\d{4}', clean_date_str) is None:
            clean_date_str += f" {datetime.now().year}"  # Append the current year if missing

        # Try parsing the date with common formats
        date_formats = ['%B %d %Y', '%b %d %Y', '%B %d, %Y', '%b %d, %Y']
        for format in date_formats:
            try:
                parsed_date = datetime.strptime(clean_date_str, format)
                return parsed_date.strftime('%Y-%m-%d')
            except ValueError:
                continue  # Try the next format if the current one fails

        # Log an error if no format matches
        logging.error(f"Date parsing failed for '{date_str}': No valid date format found.")
        return "Unknown"
    except Exception as e:
        logging.error(f"Unhandled exception while processing '{date_str}': {e}")
        return "Unknown"

    
# Function to extract event details from post
def extract_event_details(post, event_types_keywords):
    event = {}
    text = post.text.lower()

    # Extract event date
    event_date = extract_event_date(text)
    event['event_date'] = format_event_date(event_date) if event_date else 'Unknown'

    # Extract event year if present
    event_year = extract_event_year(text)
    if event_year and 'event_date' in event:
        event['event_date'] = f"{event['event_date']} {event_year}"

    # Extract event time
    event_time = extract_event_time(text)
    event['event_time'] = event_time if event_time else 'Unknown'

    # Identify event type
    event['event_type'] = identify_event_type(post.text, event_types_keywords)

    # Identify location and coordinates
    location = extract_location(text)
    latitude, longitude = get_location_coordinates(location)
    event['latitude'] = latitude if latitude else 'Unknown'
    event['longitude'] = longitude if longitude else 'Unknown'

     # Calculate sentiment score
    event['sentiment_score'] = calculate_sentiment(text)

    # Extract social engagement if available (like shares, likes, etc.)
    event['social_engagement'] = extract_social_engagement(post)

    return event

# Extract event date using regex
def extract_event_date(text):
    day_month_year_regex = r"(\b(?:january|february|march|april|may|june|july|august|september|october|november|december)\b\s\d{1,2}(?:st|nd|rd|th)?,?\s?\d{4})"
    match = re.search(day_month_year_regex, text)
    return match.group(0) if match else None

# Extract event year using regex
def extract_event_year(text):
    year_regex = r"#(20[0-9]{2})#"
    match = re.search(year_regex, text)
    return match.group(1) if match else None

# Extract event time using regex
def extract_event_time(text):
    time_regex = r"(\d{1,2}(?::\d{2})?\s?(?:am|pm))"
    match = re.search(time_regex, text)
    return match.group(0) if match else None

# Use spaCy to identify potential locations from text
def extract_location_with_nlp(text):
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ == "GPE":  # GPE = Geopolitical Entity (Location)
            return ent.text
    return None

# Extract venue from post
def extract_location(text):
    # Try extracting from predefined locations
    location_regex = r"(shangri-la hotel colombo|hilton hotel colombo|viharamahadevi open air theatre|colombo)"
    match = re.search(location_regex, text)
    
    if match:
        return match.group(0)
    
    # Fallback to NLP if no predefined location is found
    return extract_location_with_nlp(text)

# Get latitude and longitude based on extracted location
def get_location_coordinates(location):
    if location:
        # Check if location exists in predefined venues
        if location in venue_coordinates:
            return venue_coordinates[location]
        else:
            # Fallback to Geocoding API if location is not predefined
            return get_coordinates_from_location(location)
    return None, None

# Geocoding API for getting coordinates
def get_coordinates_from_location(location):
    api_key = 'AIzaSyBah2nFN4IIw0-HN_YUTRspoWuSIqmAZs4'
    geocode_url = f'https://maps.googleapis.com/maps/api/geocode/json?address={location}&key={api_key}'
    response = requests.get(geocode_url).json()

    if response['status'] == 'OK':
        geometry = response['results'][0]['geometry']['location']
        return geometry['lat'], geometry['lng']
    else:
        return None, None

# Function to identify event type from keywords
# Function to identify event type using predefined keywords and fallback on NLP
def identify_event_type(event_title, event_types_keywords):
    event_title_lower = event_title.lower()

    # Check if the event title matches any existing event category
    for event_type, keywords in event_types_keywords.items():
        if any(keyword in event_title_lower for keyword in keywords):
            return event_type.capitalize()  # Return the matched category

    # If no match is found, use NLP to extract entities and classify the event
    nlp_doc = nlp(event_title)
    event_type_nlp = None
    for ent in nlp_doc.ents:
        # We check if the entity is a commonly recognized event-related entity
        if ent.label_ in ["EVENT", "ORG", "GPE"]:  # EVENT: Events, ORG: Organizations, GPE: Locations
            event_type_nlp = ent.text.lower()
            break

    # Compare the NLP-extracted event type with predefined keywords to find the closest match
    if event_type_nlp:
        closest_event_type = find_closest_event_type(event_type_nlp, event_types_keywords)
        if closest_event_type:
            return closest_event_type.capitalize()

    # If no match is found via NLP, default to 'Other Event'
    return 'Other Event'

# Function to find closest event type from extracted NLP entity
def find_closest_event_type(extracted_event_type, event_types_keywords):
    max_similarity = 0
    closest_event_type = None
    extracted_event_vector = nlp(extracted_event_type).vector

    # Compare similarity between extracted NLP entity and predefined event types
    for event_type, keywords in event_types_keywords.items():
        event_type_vector = nlp(event_type).vector
        similarity = extracted_event_vector.dot(event_type_vector) / (
            (extracted_event_vector ** 2).sum() ** 0.5 * (event_type_vector ** 2).sum() ** 0.5
        )

        if similarity > max_similarity:
            max_similarity = similarity
            closest_event_type = event_type

    # Only return the event type if similarity is above a certain threshold (to avoid poor matches)
    if max_similarity > 0.6:  # You can adjust this threshold based on your needs
        return closest_event_type

    return None


# Scrape Facebook events using Selenium
def scrape_facebook_events(page_url):
    driver = set_up_driver()
    driver.get(page_url)
    time.sleep(20)  # Wait for dynamic content to load
    for _ in range(5):
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(3)
    page_source = driver.page_source
    soup = BeautifulSoup(page_source, 'html.parser')
    driver.quit()
    events = []
    for post in soup.find_all('div', class_='x1n2onr6 x1ja2u2z'):
        event_details = extract_event_details(post, event_types_keywords)  # Pass event_types_keywords here
        events.append(event_details)
    return events

# Updated scrape_eventbrite_events function with dynamic event categorization
def scrape_eventbrite_events():
    response = requests.get("https://www.eventbrite.com/d/sri-lanka--colombo/all-events/")
    soup = BeautifulSoup(response.text, 'html.parser')

    events = []
    
    for event_card in soup.find_all("div", class_="SearchResultPanelContentEventCardList-module__map_experiment_event_card___vyRC3"):
        # Get the event title
        title = event_card.find("h3", class_="Typography_root__487rx #3a3247 Typography_body-lg__487rx event-card__clamp-line--two Typography_align-match-parent__487rx").text.strip()

        # Get the date, time, and location text
        date_time_location = event_card.find_all("div", class_="Stack_root__1ksk7")
        
        if date_time_location:
            # Extract date and time from the first paragraph
            date_time_text = date_time_location[0].text.strip()

            # Use regex to extract date and time
            date_match = re.search(r'[A-Za-z]{3},\s[A-Za-z]{3}\s\d{1,2}', date_time_text)  # e.g., "Sat, Oct 12"
            time_match = re.search(r'\d{1,2}:\d{2}\s[APM]{2}', date_time_text)  # e.g., "11:00 AM"
            
            # Assign date and time
            if date_match:
                event_date = format_event_date(date_match.group(0))  # Assuming events are in the current year
            else:
                event_date = 'Unknown'

            event_time = time_match.group(0) if time_match else 'Unknown'

            # Extract location from the second paragraph, filtering out non-location data
            if len(date_time_location) > 1:
                location_text = date_time_location[1].text.strip()
                
                # Identify location by removing extra promotional text or phrases
                location_keywords = ['Just added', 'Going fast', 'Sales end soon', 'Moved to virtual event', 'Today', 'Tomorrow']
                for keyword in location_keywords:
                    location_text = location_text.replace(keyword, '').strip()

                # Remove time phrases and date keywords
                location_cleaned = re.sub(r"\d{1,2}:\d{2}\s[APM]{2}", "", location_text)  # Remove time
                location_cleaned = re.sub(r'\b[A-Za-z]{3},\s[A-Za-z]{3}\s\d{1,2}', '', location_cleaned)  # Remove date
                location_cleaned = location_cleaned.strip()

                location_parts = location_cleaned.split(",") 
                location = location_parts[-1].strip()  # Get the last part as the likely location
            else:
                location = 'Colombo'
        else:
            event_date = 'Unknown'
            event_time = 'Unknown'
            location = 'Colombo'

        # Get latitude and longitude for the location, checking predefined venues
        latitude, longitude = venue_coordinates.get(location.lower(), (6.9271, 79.8612))  # Default to Colombo coordinates if not found

        # Identify event type and update the keywords if necessary
        event_type = identify_event_type(title, event_types_keywords)

        # Append event details
        events.append({
            'event_type': event_type,  # Use the dynamically identified event type
            'event_date': event_date,
            'event_time': event_time,
            'latitude': latitude,
            'longitude': longitude,
            'sentiment_score': 0,
            'social_engagement': 0,
        })
    return events


# Function to extract social engagement (likes, shares, etc.)
def extract_social_engagement(post):
    likes = 0
    shares = 0
    
    # Find all possible sections that might contain engagement data (reactions like, love, etc.)
    like_ans_shares = post.find_all("div", class_="x6s0dn4 xi81zsa x78zum5 x6prxxf x13a6bvl xvq8zen xdj266r xat24cr x1d52u69 xktsk01 x889kno x1a8lsjc xkhd6sd x4uap5 x80vd3b x1q0q8m5 xso031l")
    
    for engagement in like_ans_shares:
        # Search for any reaction (Like, Love, Wow, etc.)
        reaction_spans = engagement.find_all('div', attrs={'aria-label': re.compile(r"(Like|Love|Wow|Haha|Sad|Angry): \d+ person")})
        
        # Add all the reaction counts together
        for reaction_span in reaction_spans:
            reaction_text = reaction_span['aria-label']
            reaction_match = re.search(r': (\d+) person', reaction_text)
            if reaction_match:
                likes += int(reaction_match.group(1))
    
    # Extract shares if present (this may need to be adapted if the shares are in a different part of the HTML)
    share_spans = post.find_all('div', attrs={'aria-label': re.compile(r"Share: \d+ person")})
    for share_span in share_spans:
        share_text = share_span['aria-label']
        shares_match = re.search(r'Share: (\d+)', share_text)
        if shares_match:
            shares = int(shares_match.group(1))

    return likes + shares



# Save events to a CSV file
def save_to_csv(all_events):
    df = pd.DataFrame(all_events)
    df.to_csv('./data/historical_event_data.csv', mode='a', header=False, index=False)
    print("Data saved to './data/historical_event_data.csv'")

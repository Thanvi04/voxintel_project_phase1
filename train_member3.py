import pandas as pd
from googletrans import Translator
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report
from geopy.geocoders import Nominatim
from timezonefinder import TimezoneFinder
import pytz
import requests

# Load dataset
df = pd.read_excel("final_dataset.xlsx")

# Preprocess
df["english"] = df["english"].str.lower().str.strip()
df["merged_intent"] = df["merged_intent"].str.lower().str.strip()

# Features and labels
X = df["english"]
y = df["merged_intent"]

# TF-IDF vectorizer
vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    max_features=7000,
    sublinear_tf=True
)

X_vec = vectorizer.fit_transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X_vec, y, test_size=0.2, random_state=42, stratify=y
)

# Train model
model = LinearSVC(C=2, class_weight="balanced")
model.fit(X_train, y_train)

# Training Accuracy
train_pred = model.predict(X_train)
train_accuracy = accuracy_score(y_train, train_pred)
print("Training Accuracy:", train_accuracy)

# Testing Accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Testing Accuracy:", accuracy)

# Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

# Translator
translator = Translator()

# Manual testing loop
while True:
    query = input("\nEnter Tulu sentence (or type exit): ")

    if query.lower() == "exit":
        break

    # Translate Tulu to English
    translated = translator.translate(query, dest='en')
    english_query = translated.text.lower().strip()
    print("Translated:", english_query)

    # Vectorize
    query_vec = vectorizer.transform([english_query])

    # Keyword override
    weather_keywords = ["ಬಿಸಿ", "ಹವಾಮಾನ", "ಮಳೆ", "ಚಳಿ", "ಸೂರ್ಯ", "ದೊಂಬು"]
    time_keywords = ["ಪೊರ್ತು"]
    date_keywords = ["ದಿನಾಂಕ"]
    news_keywords = ["ಸುದ್ದಿ"]

    if any(word in query for word in weather_keywords) or "weather" in english_query:
        intent = "weather"

    elif any(word in query for word in time_keywords) or "time" in english_query:
        intent = "time"

    elif any(word in query for word in date_keywords) or "date" in english_query:
        intent = "date"

    elif any(word in query for word in news_keywords) or "news" in english_query:
        intent = "news"

    else:
        # Confidence check
        decision_scores = model.decision_function(query_vec)
        max_score = decision_scores.max()

        if max_score < 0.5:
            print("ಕ್ಷಮಿಸಿ, ನನಗೆ ಅರ್ಥ ಆಗಿಲ್ಲ")
            continue

        prediction = model.predict(query_vec)
        intent = prediction[0]

    print("Predicted Intent:", intent)

    # Response generation
    if intent == "date":
        today = datetime.now().strftime("%d-%m-%Y")
        print("ಇವತ್ತಿನ ದಿನಾಂಕ:", today)

    elif intent == "time":
        if "ಎನ್ನ ಜಾಗೆಡ್" in query or "my place" in english_query:
            location_name = input("Enter your location: ")
            geolocator = Nominatim(user_agent="time_app")
            location = geolocator.geocode(location_name)

            if location:
                tf = TimezoneFinder()
                timezone_str = tf.timezone_at(lat=location.latitude, lng=location.longitude)

                if timezone_str:
                    tz = pytz.timezone(timezone_str)
                    current_time = datetime.now(tz).strftime("%I:%M %p")
                    print(f"{location_name} ಸಮಯ:", current_time)
                else:
                    print("Timezone not found")
            else:
                print("Location not found")
        else:
            current_time = datetime.now().strftime("%I:%M %p")
            print("ಈಗಿನ ಸಮಯ:", current_time)

    elif intent == "weather":
        city = input("Enter city name: ")

        api_key ="9210410964825df1fcc4b1b23e26c409"

        url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"

        response = requests.get(url)
        data = response.json()
        print(data)

        if response.status_code == 200:
            temp = data["main"]["temp"]
            condition = data["weather"][0]["description"]

            print(f"{city} ಹವಾಮಾನ: {temp}°C, {condition}")
        else:
            print("Weather information not found")
    
    elif intent == "news":
        api_key = "57dd543822c341ada2e139a034f0d740"

        url = f"https://newsapi.org/v2/everything?q=india&SSlanguage=en&sortBy=publishedAt&apiKey={api_key}"

        response = requests.get(url)
        data = response.json()

        if data["status"] == "ok" and len(data["articles"]) > 0:
            print("ಇಂದಿನ ಮುಖ್ಯ ಸುದ್ದಿಗಳು:")

            for i, article in enumerate(data["articles"][:5], start=1):
                print(f"{i}. {article['title']}")
        else:
            print("News not found")
    else:
        print("Response not implemented yet.")
        
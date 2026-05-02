import pickle
from googletrans import Translator
from datetime import datetime
from geopy.geocoders import Nominatim
from timezonefinder import TimezoneFinder
import pytz
import requests

# Load model and vectorizer
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

translator = Translator()

# Chatbot loop
while True:
    query = input("\nEnter your question in Tulu (or type exit): ")

    if query.lower() == "exit":
        break

    # Translate Tulu to English
    translated = translator.translate(query, dest='en')
    english_query = translated.text.lower().strip()
    print("Translated to English:", english_query)

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

        api_key = "9210410964825df1fcc4b1b23e26c409"

        url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"

        response = requests.get(url)
        data = response.json()

        if response.status_code == 200:
            temp = data["main"]["temp"]
            condition = data["weather"][0]["description"]

            print(f"{city} ಹವಾಮಾನ: {temp}°C, {condition}")
        else:
            print("Weather information not found")

    else:
        print("Intent:", intent)
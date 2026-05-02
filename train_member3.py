import pandas as pd
import requests
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from google import genai

# ---------------- GEMINI SETUP ----------------
client = genai.Client(api_key="AIzaSyASAeaVWcp42AA8zo1EgVnMwv3q3v3sjZA")

# ---------------- GEMINI TRANSLATION ----------------
def translate_to_tulu(text):
    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=f"""
Translate the following English text into Tulu written in Kannada script.
Only give Tulu output. No explanation.

{text}
"""
        )

        # Proper extraction
        if hasattr(response, "text") and response.text:
            return response.text.strip()

        if response.candidates:
            return response.candidates[0].content.parts[0].text.strip()

        return "⚠️ Translation failed"

    except Exception:
        return text  # fallback if quota issue


# ---------------- LOAD DATA ----------------
df = pd.read_excel("dataset res.xlsx")

df["english"] = df["english"].str.lower().str.strip()
df["merged_intent"] = df["merged_intent"].str.lower().str.strip()

# ---------------- RESPONSE DICTIONARY ----------------
response_dict = {}

for _, row in df.iterrows():
    key = row["english"]

    eng = row.get("response_english", row.get("response", "No response"))
    tulu = row.get("response_tulu", eng)

    response_dict[key] = {
        "eng": eng,
        "tulu": tulu
    }

# ---------------- MODEL ----------------
X = df["english"]
y = df["merged_intent"]

vectorizer = TfidfVectorizer(ngram_range=(1,2), max_features=7000)
X_vec = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_vec, y, test_size=0.2, random_state=42, stratify=y
)

model = LinearSVC()
model.fit(X_train, y_train)

print("Accuracy:", accuracy_score(y_test, model.predict(X_test)))

# ---------------- MAIN LOOP ----------------
while True:
    query = input("\nEnter sentence: ").strip().lower()

    if query == "exit":
        break

    print("\n--- User Input ---")
    print("English:", query)

    # -------- INTENT --------
    query_vec = vectorizer.transform([query])
    intent = model.predict(query_vec)[0]

    print("\nPredicted Intent:", intent)

    # -------- WEATHER --------
    if intent == "weather":
        city = input("Enter city: ")
        api_key = "396d46ccd112b9daac037e017f99ffbd"

        url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
        data = requests.get(url).json()

        if data.get("cod") == 200:
            temp = data["main"]["temp"]
            condition = data["weather"][0]["description"]

            english_response = f"Today in {city}, temperature is {temp}°C "
            tulu_response = f"{city} ದ ಇನಿ ತಾಪಮಾನ {temp} ಡಿಗ್ರಿ"

        else:
            english_response = "Weather not found"
            tulu_response = "ಹವಾಮಾನ ಗೊತ್ತುಜ್ಜಿ"

    # -------- TIME --------
    elif intent == "time":
        current_time = datetime.now().strftime('%I:%M %p')
        english_response = f"The current time is {current_time}"
        tulu_response = f"ಇತ್ತೆದ ಸಮಯ  {current_time}"

    # -------- DATE --------
    elif intent == "date":
        today = datetime.now().strftime('%d-%m-%Y')
        english_response = f"Today's date is {today}"
        tulu_response = f"ಇನಿತ ದಿನಾಂಕ {today}"

    # -------- NEWS --------
    elif intent == "news":
        api_key = "ffd88ae576ae4e6fac076cf33481d542"
        url = f"https://newsapi.org/v2/everything?q=india&language=en&apiKey={api_key}"

        data = requests.get(url).json()

        print("\n--- Response ---")
        print("English: Here are today's top news")
        print("Tulu:", translate_to_tulu("Here are today's top news"))

        if data["status"] == "ok":
            headlines = [article["title"] for article in data["articles"][:5]]

            # 🔥 ONE GEMINI CALL
            combined_text = "\n".join(headlines)
            translated = translate_to_tulu(combined_text)

            tulu_lines = translated.split("\n")

            for i, eng in enumerate(headlines):
                tulu = tulu_lines[i] if i < len(tulu_lines) else translate_to_tulu(eng)

                print(f"\nNews {i+1}:")
                print("English:", eng)
                print("Tulu:", tulu)
        else:
            print("News not found")

        continue

    # -------- DATASET RESPONSE --------
    else:
        if query in response_dict:
            english_response = response_dict[query]["eng"]
            tulu_response = response_dict[query]["tulu"]
        else:
            english_response = "I understand you. Tell me more."
            tulu_response = "ಎಂಕ್ ನಿನನ್ ಅರ್ಥ ಆಂಡ್. ಎಂಕ್ ಜಾಸ್ತಿ ಪನ್ಲೆ"
    # -------- FINAL OUTPUT --------
    print("\n--- Response ---")
    print("English:", english_response)
    print("Tulu:", tulu_response)
    print("test")
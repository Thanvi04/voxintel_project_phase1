import pandas as pd
import re

# 1. Load dataset
df = pd.read_excel("dataset.xlsx")

print("\n========== ORIGINAL DATA ==========")
print(df.head())

# 2. DATA VALIDATION
print("\n========== DATASET INFO ==========")
print(df.info())

print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())

# 3. CATEGORY CHECKING
print("\n========== UNIQUE CATEGORIES ==========")
if 'Type' in df.columns:
    print(df['Type'].unique())

# 4. CLEANING
df = df.dropna()
df = df.drop_duplicates()

df.columns = df.columns.str.strip()

# Text cleaning function
def clean_text(text):
    text = str(text)
    text = text.lower().strip()
    text = re.sub(r'[^\w\s]', '', text)  # remove punctuation
    return text

# Apply cleaning
if 'Tulu' in df.columns:
    df['Tulu'] = df['Tulu'].apply(clean_text)

if 'English' in df.columns:
    df['English'] = df['English'].apply(clean_text)

# 5. SAVE CLEANED DATASET
df.to_csv("cleaned_dataset.csv", index=False)
df.to_excel("cleaned_dataset.xlsx", index=False)

print("\n✅ CLEANED DATASET SAVED SUCCESSFULLY")
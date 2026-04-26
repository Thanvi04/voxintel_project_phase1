import pandas as pd
df = pd.read_excel("cleaned_dataset.xlsx")
df["merged_intent"] = df["merged_intent"].str.lower().str.strip()
intent_map = {
    "help": "communication",
    "instruction": "communication",
    "polite_expression": "communication",
    "future_action": "task",
    "study": "task",
    "placement": "movement",
    "opinion": "status",
    "appreciation": "status",
    "greeting": "communication",
    "information_request": "communication",
    "safety": "action",
    "positioning": "movement",
    "time": "date",
    "wait": "movement",
    
}
df["merged_intent"] = df["merged_intent"].replace(intent_map)
df.to_excel("final_dataset.xlsx", index=False)
print("Updated dataset saved as final_dataset.xlsx")
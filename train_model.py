
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from pathlib import Path

# Find the dataset folder
data_folder = Path("data")

# Check that the dataset exists
fake_path = data_folder / "Fake.csv"
real_path = data_folder / "True.csv"

if not fake_path.exists() or not real_path.exists():
    raise FileNotFoundError(
        "Dataset missing! Put Fake.csv and True.csv "
        "inside the data folder."
    )

# Load the news data
fake = pd.read_csv(fake_path)
real = pd.read_csv(real_path)

# Assign labels: 0 = fake, 1 = real
fake["label"] = 0
real["label"] = 1

# Combine the datasets
data = pd.concat([fake, real], ignore_index=True)

# Clean missing and empty article text
data = data.dropna(subset=["text"])
data["text"] = data["text"].astype(str).str.strip()
data = data[data["text"] != ""]
data = data.drop_duplicates(subset=["text"])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    data["text"],
    data["label"],
    test_size=0.2,
    random_state=42,
    stratify=data["label"]
)

# Build the text classification pipeline
model = Pipeline([
    ("tfidf", TfidfVectorizer(
        stop_words="english",
        max_features=30000,
        ngram_range=(1, 2),
        max_df=0.95
    )),
    ("classifier", LogisticRegression(max_iter=1000))
])

# Train the model
print("Training the model...")
model.fit(X_train, y_train)

# Evaluate the model
predictions = model.predict(X_test)

print("\nTest accuracy:",
      round(accuracy_score(y_test, predictions), 4))

print("\nClassification report:")
print(classification_report(
    y_test,
    predictions,
    target_names=["Fake", "Real"]
))

# Save the trained model
joblib.dump(model, "fake_news_model.joblib")

print("\nModel saved successfully!")

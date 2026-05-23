import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("IMDB Dataset.csv")

#print(df.head())

# Input and output
x = df['review']
y = df['sentiment']

# Split dataset
x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42
)

# Convert text into numbers
vectorizer = TfidfVectorizer(stop_words='english')

x_train = vectorizer.fit_transform(x_train)
x_test = vectorizer.transform(x_test)

# Train model
model = LogisticRegression()

model.fit(x_train, y_train)

# Prediction
prediction = model.predict(x_test)

accuracy = accuracy_score(y_test, prediction)

print("Accuracy:", accuracy)

# Custom sentence
user_input = input("Enter your review: ")

text = [user_input]

text_vector = vectorizer.transform(text)

result = model.predict(text_vector)

print("Prediction:", result)
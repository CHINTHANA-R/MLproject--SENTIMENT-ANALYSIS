import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Load the dataset
data = pd.read_csv('sentiment_data.csv')

# Prepare the feature and target variables
X = data['text']  # Feature
y = data['sentiment']  # Target

# Convert text data to numerical data
vectorizer = CountVectorizer()
X_vectorized = vectorizer.fit_transform(X)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)

# Create and train the model
model = MultinomialNB()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)

# Example usage
new_texts = ["helooo hiii."]
new_texts_vectorized = vectorizer.transform(new_texts)
predictions = model.predict(new_texts_vectorized)
print("Predictions for new texts:", predictions)

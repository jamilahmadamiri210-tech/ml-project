# 🔐 Malicious URL Detection - Complete Notebook Script

# 1️⃣ Import Libraries
import pandas as pd
import numpy as np
import re
from urllib.parse import urlparse
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

sns.set(style="whitegrid")  # Set plot style

# 2️⃣ Load Dataset
data = pd.read_csv("malicious_phish.csv")
print("Dataset loaded successfully.")

# 3️⃣ Print the dataset and understand tabular data
print("\nFirst 5 rows:")
print(data.head())

print("\nDataset Info:")
print(data.info())

print("\nDataset Description (numeric features):")
print(data.describe())

# 4️⃣ Check for NaN values and remove if present
print("\nChecking for NaN values:")
print(data.isnull().sum())
data.dropna(inplace=True)
print("Dataset shape after removing NaNs:", data.shape)

# 5️⃣ Feature Engineering Function (14 features)
def extract_features(url):
    parsed = urlparse(url)
    hostname = parsed.netloc
    path = parsed.path
    return pd.Series({
        "has_ip": 1 if re.search(r'\d+\.\d+\.\d+\.\d+', url) else 0,
        "url_length": len(url),
        "hostname_length": len(hostname),
        "dot_count": url.count('.'),
        "dash_count": url.count('-'),
        "slash_count": url.count('/'),
        "question_count": url.count('?'),
        "percent_count": url.count('%'),
        "at_count": url.count('@'),
        "directory_count": path.count('/'),
        "embedded_domain_count": hostname.count('.') - 1,
        "is_shortened": 1 if any(s in url.lower() for s in ["bit.ly","tinyurl","goo.gl","ow.ly","t.co"]) else 0,
        "has_suspicious_words": 1 if any(word in url.lower() for word in ["login","update","free","secure","paypal","bank"]) else 0,
        "digit_count": sum(c.isdigit() for c in url),
        "letter_count": sum(c.isalpha() for c in url),
        "first_dir_length": len(path.split('/')[1]) if len(path.split('/')) > 1 else 0,
        "tld_length": len(hostname.split('.')[-1]) if '.' in hostname else 0
    })

# Apply feature extraction
features = data['url'].apply(extract_features)
data = pd.concat([data, features], axis=1)

# 6️⃣ Encode Labels (text to numeric)
data['type'] = data['type'].replace({'benign':0,'defacement':1,'phishing':2,'malware':3})

# 7️⃣ Visualize dataset
# Scatter plot example
sns.scatterplot(x='url_length', y='digit_count', hue='type', data=data)
plt.title("URL Length vs Digit Count by Type")
plt.show()

# Bar plot example: average url_length by type
data.groupby('type')['url_length'].mean().plot(kind='bar', figsize=(6,4))
plt.title("Average URL Length by Type")
plt.show()

# Boxplot for outliers
plt.figure(figsize=(10,6))
sns.boxplot(x='type', y='url_length', data=data)
plt.title("Boxplot: URL Length by Type")
plt.show()

# Line plot example
data.groupby('type')['dot_count'].mean().plot(kind='line', marker='o')
plt.title("Average Dot Count by Type")
plt.show()

# 8️⃣ Handle outliers (remove extreme url_length values)
Q1 = data['url_length'].quantile(0.25)
Q3 = data['url_length'].quantile(0.75)
IQR = Q3 - Q1
data = data[(data['url_length'] >= Q1 - 1.5*IQR) & (data['url_length'] <= Q3 + 1.5*IQR)]
print("Dataset shape after removing outliers:", data.shape)

# 9️⃣ Prepare Features and Labels
X = data.drop(['url','type'], axis=1)
y = data['type']

# Optional: scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 10️⃣ Split dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 11️⃣ Train Machine Learning Model (Random Forest as example)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 12️⃣ Test model
y_pred = model.predict(X_test)

# 13️⃣ Compare predictions with actual outputs
comparison = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print("\nComparison of Actual vs Predicted:")
print(comparison.head(10))

# Line plot to visualize comparison
plt.figure(figsize=(12,5))
plt.plot(comparison['Actual'].values, label='Actual', marker='o')
plt.plot(comparison['Predicted'].values, label='Predicted', marker='x')
plt.title("Actual vs Predicted Labels")
plt.xlabel("Sample Index")
plt.ylabel("Label")
plt.legend()
plt.show()

# 14️⃣ Confusion Matrix and Classification Report
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 🔹 Function to predict new URL (takes input from user)
def predict_url_user_input():
    url = input("Enter the URL to predict: ")
    features = extract_features(url).values.reshape(1,-1)
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)[0]
    labels = {0:"Benign", 1:"Defacement", 2:"Phishing", 3:"Malware"}
    print(f"URL: {url}")
    print(f"Prediction: {labels[prediction]}")

# Example usage: user can input a URL
predict_url_user_input()

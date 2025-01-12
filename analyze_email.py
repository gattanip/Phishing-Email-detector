import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import numpy as np

# Predefined list of phishing keywords
PHISHING_KEYWORDS = ["urgent", "click here", "free", "offer", "win", "exclusive", "action required", "verify"]

# Sample dataset for demonstration (replace with a larger, real dataset for production use)
dataset = [
    ("Please verify your account information by clicking here.", 1),
    ("Your account will be deactivated if you do not respond immediately.", 1),
    ("Exclusive offer just for you!", 1),
    ("Win a free iPhone by entering your details.", 1),
    ("Team meeting at 3 PM tomorrow in the conference room.", 0),
    ("Can you review the attached document and provide feedback?", 0),
    ("Please find the invoice for last month attached.", 0),
    ("Looking forward to catching up next week.", 0)
]

# Prepare dataset for training
emails, labels = zip(*dataset)

# Vectorize the email content
vectorizer = CountVectorizer(stop_words='english', max_features=500)
X = vectorizer.fit_transform(emails)
y = np.array(labels)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train a simple Naive Bayes classifier
classifier = MultinomialNB()
classifier.fit(X_train, y_train)

# Test the model
predictions = classifier.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Save the model and vectorizer for reuse
joblib.dump(classifier, "phishing_email_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

# Function to analyze email content
def analyze_email(content):
    """Analyze email content for phishing characteristics."""
    # Check for suspicious links
    suspicious_links = re.findall(r'http[s]?://\S+', content)

    # Check for phishing keywords
    phishing_keyword_matches = [kw for kw in PHISHING_KEYWORDS if kw.lower() in content.lower()]

    # Vectorize the email and predict phishing likelihood
    vectorizer = joblib.load("vectorizer.pkl")
    classifier = joblib.load("phishing_email_model.pkl")
    vectorized_email = vectorizer.transform([content])
    prediction = classifier.predict(vectorized_email)[0]
    phishing_probability = classifier.predict_proba(vectorized_email)[0][1]

    # Generate a report
    report = {
        "suspicious_links": suspicious_links,
        "phishing_keywords": phishing_keyword_matches,
        "is_phishing": bool(prediction),
        "phishing_probability": phishing_probability,
    }

    return report

# Example usage
if __name__ == "__main__":
    email_content = "Your account is at risk. Click here to secure it now: http://phishing-site.com"
    result = analyze_email(email_content)
    print("Phishing Analysis Report:")
    print(f"- Suspicious Links: {result['suspicious_links']}")
    print(f"- Phishing Keywords: {result['phishing_keywords']}")
    print(f"- Is Phishing: {'Yes' if result['is_phishing'] else 'No'}")
    print(f"- Phishing Probability: {result['phishing_probability'] * 100:.2f}%")

import nltk
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

nltk.download('punkt')

data = {
    'Question': [
        'Hi', 'Hello', 'Hey', 'Good morning', 'Good evening',
        'Bye', 'Goodbye', 'See you later', 'Take care',
        'What is AI', 'Explain AI', 'Tell me about artificial intelligence',
        'What is machine learning', 'Explain machine learning', 'How does ML work',
        'Tell me a joke', 'Make me laugh', 'Do you know any jokes',
        'What is your name', 'Who are you', 'Introduce yourself',
        'Who created you', 'Who made you', 'Tell me about your creator',
        'What is your purpose', 'What can you do', 'How can you help me',
        'What is Python', 'Explain Python', 'Tell me about Python',
        'What is your favorite color', 'Do you have hobbies', 'What do you like',
        'How are you', 'How are you doing', 'Are you okay'
    ],
    'Intent': [
        'greeting', 'greeting', 'greeting', 'greeting', 'greeting',
        'goodbye', 'goodbye', 'goodbye', 'goodbye',
        'ai_explanation', 'ai_explanation', 'ai_explanation',
        'ml_explanation', 'ml_explanation', 'ml_explanation',
        'joke', 'joke', 'joke',
        'bot_info', 'bot_info', 'bot_info',
        'creator_info', 'creator_info', 'creator_info',
        'bot_capability', 'bot_capability', 'bot_capability',
        'python_explanation', 'python_explanation', 'python_explanation',
        'personal_question', 'personal_question', 'personal_question',
        'greeting', 'greeting', 'greeting'
    ]
}
df = pd.DataFrame(data)

encoder = LabelEncoder()
df['Intent'] = encoder.fit_transform(df['Intent'])

X_train, X_test, y_train, y_test = train_test_split(df['Question'], df['Intent'], test_size=0.2, random_state=42)

model = make_pipeline(CountVectorizer(), MultinomialNB())
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

def get_intent(user_input):
    intent = model.predict([user_input])[0]
    return encoder.inverse_transform([intent])[0]

responses = {
    'greeting': "Hello! How can I assist you?",
    'goodbye': "Goodbye! Have a great day!",
    'ai_explanation': "Artificial Intelligence (AI) is the simulation of human intelligence in machines that are programmed to think and learn.",
    'ml_explanation': "Machine Learning (ML) is a subset of AI that enables systems to learn from data and improve over time without being explicitly programmed.",
    'joke': "Why did the computer go to therapy? It had too many bytes of anxiety!",
    'bot_info': "I'm a simple chatbot created by Dan Zuchte! I'm here to answer your questions and make you smile.",
    'creator_info': "I was created by Dan Zuchte. They're pretty awesome, right?",
    'bot_capability': "I can answer questions, tell jokes, and help you learn about AI and machine learning. What would you like to know?",
    'python_explanation': "Python is a popular programming language known for its simplicity and versatility. It’s widely used in web development, data science, AI, and more.",
    'personal_question': "I don’t have personal preferences, but I’m here to help you with whatever you need!",
    'default': "I'm not sure about that. Can you rephrase your question?"
}

print("Chatbot: Hi! Ask me anything (type 'bye' to exit)")

while True:
    user_input = input("You: ").strip().lower()
    if user_input == 'bye':
        print("Chatbot:", responses['goodbye'])
        break
    intent = get_intent(user_input)
    response = responses.get(intent, "I'm not sure about that.")
    print("Chatbot:", response)

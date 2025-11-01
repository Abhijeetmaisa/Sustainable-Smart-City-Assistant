import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import joblib
import os

def train_model():
    """Train the chatbot model using sentence transformers"""
    print("🤖 Starting model training...")
    
    # Load the dataset
    if not os.path.exists('qa_dataset.csv'):
        print("❌ Error: qa_dataset.csv not found!")
        return
    
    data = pd.read_csv('qa_dataset.csv')
    print(f"📊 Loaded {len(data)} question-answer pairs")
    
    # Initialize the sentence transformer model
    print("🧠 Loading sentence transformer model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Generate embeddings for all questions
    print("⚡ Generating embeddings...")
    questions = data['question'].tolist()
    embeddings = model.encode(questions, show_progress_bar=True)
    
    # Save the embeddings and data
    print("💾 Saving model and embeddings...")
    model_data = {
        'embeddings': embeddings,
        'questions': questions,
        'answers': data['answer'].tolist(),
        'categories': data['category'].tolist()
    }
    
    joblib.dump(model_data, 'chatbot_model.pkl')
    print("✅ Model training completed successfully!")
    print("📁 Saved as 'chatbot_model.pkl'")

if __name__ == "__main__":
    train_model()
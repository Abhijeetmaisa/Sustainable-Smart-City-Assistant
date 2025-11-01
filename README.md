# Smart City Sustainability Chatbot

A Python-based chatbot built with Streamlit that provides expert answers to sustainability and smart city questions using AI-powered semantic search.

## Features

- 🤖 **AI-Powered Responses**: Uses sentence transformers for semantic similarity matching
- 🌿 **Sustainability Focus**: Trained on 50+ expert Q&A pairs about smart cities and sustainability
- 💬 **Interactive Chat**: Modern Streamlit interface with conversation history
- 📊 **Confidence Scoring**: Shows how confident the AI is in each response
- 🏷️ **Category Classification**: Organizes topics by categories like Energy, Transportation, etc.
- ⚡ **Offline Operation**: Works completely offline once trained

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the Model
```bash
python train_model.py
```

### 3. Run the Chatbot
```bash
streamlit run app.py
```

The chatbot will open in your browser at `http://localhost:8501`

## Usage

1. **Ask Questions**: Type questions about sustainability, smart cities, or environmental topics
2. **Try Samples**: Click on sample questions for quick testing
3. **View Confidence**: Each answer shows a confidence score
4. **Browse Categories**: See all available topic categories in the sidebar

## Example Questions

- "How can cities reduce air pollution?"
- "What are the benefits of smart grids?"
- "How does urban farming help cities?"
- "What is sustainable transportation?"

## Dataset

The chatbot is trained on a curated dataset covering:

- **Air Quality & Pollution Control**
- **Water Management & Conservation** 
- **Waste Management & Circular Economy**
- **Energy Efficiency & Renewable Energy**
- **Sustainable Transportation**
- **Green Buildings & Infrastructure**
- **Urban Planning & Development**
- **Climate Adaptation & Resilience**

## Technical Details

- **Model**: SentenceTransformers with all-MiniLM-L6-v2
- **Search**: Semantic similarity using cosine similarity
- **Framework**: Streamlit for the web interface
- **Storage**: Joblib for model persistence

## Customization

### Adding New Q&A Pairs
1. Edit `qa_dataset.csv` to add new questions, answers, and categories
2. Re-run `python train_model.py` to retrain the model
3. Restart the app

### Adjusting Confidence Threshold
In `app.py`, modify the `threshold` parameter in the `get_response()` function to change when the bot says "I don't know"

## Project Structure

```
smart-city-chatbot/
├── app.py                 # Main Streamlit application
├── train_model.py         # Model training script
├── qa_dataset.csv         # Question-answer dataset
├── requirements.txt       # Python dependencies
├── chatbot_model.pkl      # Trained model (generated)
└── README.md             # This file
```

## Deployment

### Local Development
```bash
streamlit run app.py
```

### Production Deployment
The app can be deployed on:
- **Streamlit Cloud** (recommended)
- **Heroku**
- **AWS/Google Cloud**
- **Local server**

## Contributing

1. Add new Q&A pairs to the dataset
2. Improve the user interface
3. Add new features like multilingual support
4. Optimize model performance

## License

Open source - feel free to use and modify for your smart city projects!

---

🌍 **Built for sustainable smart cities • Powered by AI**
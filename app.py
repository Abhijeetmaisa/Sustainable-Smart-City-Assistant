import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sentence_transformers import SentenceTransformer, util
import os
import time

# Configure page
st.set_page_config(
    page_title="Smart City Sustainability Chatbot",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #2E8B57, #228B22);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .main-header h1 {
        color: white;
        text-align: center;
        margin: 0;
    }
    .chat-container {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .user-message {
        background-color: #e3f2fd;
        padding: 0.8rem;
        border-radius: 15px;
        margin: 0.5rem 0;
        border-left: 4px solid #2196f3;
    }
    .bot-message {
        background-color: #e8f5e8;
        padding: 0.8rem;
        border-radius: 15px;
        margin: 0.5rem 0;
        border-left: 4px solid #4caf50;
    }
    .confidence-score {
        background-color: #fff3cd;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.8rem;
        display: inline-block;
        margin-top: 0.5rem;
    }
    .category-tag {
        background-color: #d4edda;
        color: #155724;
        padding: 0.2rem 0.8rem;
        border-radius: 15px;
        font-size: 0.8rem;
        display: inline-block;
        margin-right: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load the trained model and sentence transformer"""
    try:
        if not os.path.exists('chatbot_model.pkl'):
            st.error("❌ Model not found! Please run 'python train_model.py' first.")
            return None, None
        
        model_data = joblib.load('chatbot_model.pkl')
        sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        return model_data, sentence_model
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None, None

def get_response(question, model_data, sentence_model, threshold=0.3):
    """Get the best matching response for a question"""
    if not question.strip():
        return "Please ask a question about sustainability or smart cities!", 0.0, "General"
    
    # Encode the user question
    question_embedding = sentence_model.encode([question])
    
    # Calculate similarities with all stored questions
    similarities = util.cos_sim(question_embedding, model_data['embeddings'])[0]
    
    # Find the best match
    best_match_idx = similarities.argmax().item()
    confidence = similarities[best_match_idx].item()
    
    # Return response if confidence is above threshold
    if confidence >= threshold:
        return (
            model_data['answers'][best_match_idx],
            confidence,
            model_data['categories'][best_match_idx]
        )
    else:
        return (
            "I'm not sure about that question. Could you try asking about sustainability, energy efficiency, waste management, transportation, or other smart city topics?",
            confidence,
            "Unknown"
        )

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🌿 Smart City Sustainability Chatbot</h1>
    </div>
    """, unsafe_allow_html=True)
    
    # Load model
    model_data, sentence_model = load_model()
    
    if model_data is None or sentence_model is None:
        st.stop()
    
    # Sidebar with information
    with st.sidebar:
        st.header("ℹ️ About")
        st.write("""
        This chatbot is trained on sustainability and smart city topics. 
        Ask questions about:
        
        🌱 **Environmental Topics**
        - Air & water quality
        - Waste management
        - Carbon reduction
        
        🏙️ **Smart City Features**
        - Smart grids & buildings
        - Traffic management
        - Urban planning
        
        🚀 **Sustainable Technologies**
        - Renewable energy
        - Green infrastructure
        - Electric vehicles
        """)
        
        st.header("📊 Statistics")
        st.metric("Questions in Database", len(model_data['questions']))
        
        # Display categories
        st.header("🏷️ Categories")
        categories = list(set(model_data['categories']))
        for cat in sorted(categories):
            st.write(f"• {cat}")
    
    # Main chat interface
    st.subheader("💬 Ask Your Question")
    
    # Initialize session state for chat history
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Sample questions for quick access
    st.write("**Try these sample questions:**")
    col1, col2, col3 = st.columns(3)
    
    sample_questions = [
        "How can cities reduce air pollution?",
        "What are smart grids?",
        "Benefits of green buildings?",
        "How does bike sharing help cities?",
        "What is urban farming?",
        "How to improve water conservation?"
    ]
    
    for i, question in enumerate(sample_questions[:6]):
        col = [col1, col2, col3][i % 3]
        if col.button(f"💡 {question[:25]}...", key=f"sample_{i}"):
            st.session_state.user_question = question
    
    # User input
    user_question = st.text_input(
        "Enter your question:",
        key="user_input",
        placeholder="e.g., How can cities reduce their carbon footprint?"
    )
    
    # Handle sample question selection
    if 'user_question' in st.session_state:
        user_question = st.session_state.user_question
        del st.session_state.user_question
    
    # Process question
    if user_question:
        with st.spinner("🤔 Thinking..."):
            # Simulate typing delay
            time.sleep(0.5)
            
            # Get response
            answer, confidence, category = get_response(
                user_question, model_data, sentence_model
            )
            
            # Add to chat history
            st.session_state.chat_history.append({
                'question': user_question,
                'answer': answer,
                'confidence': confidence,
                'category': category
            })
    
    # Display chat history
    if st.session_state.chat_history:
        st.subheader("💬 Conversation")
        
        for chat in reversed(st.session_state.chat_history[-5:]):  # Show last 5 exchanges
            # User message
            st.markdown(f"""
            <div class="user-message">
                <strong>🧑 You:</strong> {chat['question']}
            </div>
            """, unsafe_allow_html=True)
            
            # Bot response
            confidence_color = "🟢" if chat['confidence'] > 0.7 else "🟡" if chat['confidence'] > 0.4 else "🔴"
            
            st.markdown(f"""
            <div class="bot-message">
                <strong>🤖 Assistant:</strong> {chat['answer']}
                <br><br>
                <span class="category-tag">{chat['category']}</span>
                <span class="confidence-score">
                    {confidence_color} Confidence: {chat['confidence']:.1%}
                </span>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("---")
    
    # Clear chat button
    if st.session_state.chat_history:
        if st.button("🗑️ Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 1rem;'>
        <p>🌍 Built for sustainable smart cities • Powered by AI</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
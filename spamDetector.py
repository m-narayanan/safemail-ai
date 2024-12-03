import streamlit as st
import pickle
import numpy as np
import uuid

# Configure page at the very top of the script
st.set_page_config(
    page_title="AI Spam Detector", 
    page_icon="🕵️‍♀️", 
    layout="wide"
)

# Load the model and vectorizer
@st.cache_resource
def load_model():
    model = pickle.load(open('spam.pkl', 'rb'))
    cv = pickle.load(open('vectorizer.pkl', 'rb'))
    return model, cv

model, cv = load_model()

# Model accuracy
MODEL_ACCURACY = 0.9777

# Initialize session state
def initialize_session_state():
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "Spam Detector"
    if 'last_prediction' not in st.session_state:
        st.session_state.last_prediction = None
    if 'feedback_given' not in st.session_state:
        st.session_state.feedback_given = False
    if 'user_input' not in st.session_state:
        st.session_state.user_input = ""

# Function to reset session state
def reset_session_state():
    st.session_state.last_prediction = None
    st.session_state.feedback_given = False
    st.session_state.user_input = ""

# Initialize session state
initialize_session_state()

# Function to classify the email
def classify_email(email):
    vect = cv.transform([email]).toarray()
    prediction = model.predict(vect)
    confidence = model.predict_proba(vect).max()
    return prediction[0], confidence

# Main Streamlit application
def main():
    # Sidebar navigation
    st.sidebar.title("🕵️‍♀️ SafeMail AI")
    st.session_state.current_page = st.sidebar.selectbox(
        "Choose an Option", 
        ["Spam Detector", "Conversation History", "Settings"]
    )

    # Custom CSS for styling
    st.markdown("""
    <style>
    .stApp {
        background-color: #61b8af;
    }
    .stTextArea {
        background-color: black;
        border-radius: 10px;
        border: 1px solid #e0e0e0;
    }
    .spam-result {
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 15px;
        font-weight: bold;
    }
    .spam-detected {
        background-color: #ffebee;
        border: 2px solid #ff1744;
        color: #d32f2f;
    }
    .safe-email {
        background-color: #e8f5e9;
        border: 2px solid #4caf50;
        color: #2e7d32;
    }
    </style>
    """, unsafe_allow_html=True)

    # Page content based on selected option
    if st.session_state.current_page == "Spam Detector":
        # Title and introduction
        st.title("🕵️‍♀️ AI Email Spam Detective")
        st.markdown("""
        ### Detect spam emails with advanced machine learning
        Our intelligent system helps you identify unwanted emails quickly and accurately.
        """)

        # Model accuracy display
        st.markdown(f"**Model Accuracy:** {MODEL_ACCURACY:.2%} 📊")

        # Email input area with session state
        st.markdown("### Enter Email for Spam Check")
        st.session_state.user_input = st.text_area(
            "Paste your email content here:", 
            height=250, 
            value=st.session_state.user_input,
            key="email_input"
        )

        # Classify button
        if st.button("🔍 Detect Spam"):
            # Reset feedback state
            st.session_state.feedback_given = False
            
            if st.session_state.user_input:
                # Classify email
                prediction, confidence = classify_email(st.session_state.user_input)
                
                # Store last prediction
                st.session_state.last_prediction = {
                    'prediction': prediction,
                    'confidence': confidence
                }
                
                # Determine result
                if prediction == 1:  # Spam
                    result_message = "🚨 SPAM DETECTED"
                    result_class = "spam-result spam-detected"
                    explanation = "This email appears to be spam. Be cautious!"
                else:  # Ham
                    result_message = "✅ SAFE EMAIL"
                    result_class = "spam-result safe-email"
                    explanation = "This email looks safe to open."

                # Display result
                st.markdown(f"<div class='{result_class}'>{result_message}</div>", unsafe_allow_html=True)
                st.write(f"**Confidence:** {confidence:.2%}")
                st.write(explanation)


        # Only show feedback if a prediction has been made
        if st.session_state.last_prediction is not None:
            # Feedback mechanism
            st.markdown("### Feedback")
            feedback = st.radio(
				"Was our spam detection accurate?", 
				["Select", "Yes", "No", "Unsure"], 
				horizontal=True,
				key="feedback_radio",  # Ensure this key is consistent
				index=0  # This ensures "Select" is the default selected option
			)

            # Handle feedback
            if feedback != "Select" and not st.session_state.feedback_given:
                st.session_state.feedback_given = True
                st.success("Thank you for your feedback, it will help to refine the model later when I implement future enhancement.")
                
                # Clear chat button
                if st.button("🔄 Start New Chat"):
                    # Reset everything
                    reset_session_state()
                    st.session_state.feedback_radio = 0
                    st.experimental_rerun()

    elif st.session_state.current_page == "Conversation History":
        st.title("Conversation History")
        st.write("🚧 This feature is under future enhancement.")

    elif st.session_state.current_page == "Settings":
        st.title("Settings")
        st.write("🚧 This feature is under future enhancement.")
        
	# Footer with additional information
    st.sidebar.markdown("---")
    st.sidebar.write("### About")
    st.sidebar.write("This application uses machine learning to classify emails as spam or ham.")
    st.sidebar.write("Developed by Narayanan M.")
    
    # GitHub button
    github_url = "https://github.com/m-narayanan"  # Replace with your GitHub repo URL
    st.sidebar.markdown(f"[![GitHub](https://img.shields.io/badge/GitHub-Repo-blue?style=for-the-badge&logo=github)]({github_url})")


# Run the application
if __name__ == "__main__":
    main()
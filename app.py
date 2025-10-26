import streamlit as st
import time

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Spotify Hit Predictor",
    page_icon="🎵",
    layout="wide"
)

# --- 2. SPOTIFY COLOR THEME ---
st.markdown("""
<style>
/* Main app background */
.stApp {
    background-color: #121212;
    color: #FFFFFF;
}

/* Sidebar background */
[data-testid="stSidebar"] {
    background-color: #040404;
}

/* Headers and text */
h1, h2, h3, h4, h5, h6, .stMarkdown, .stWrite, .stSubheader {
    color: #FFFFFF;
}

/* Spotify Green Button */
.stButton > button {
    background-color: #1DB954;
    color: #FFFFFF;
    border: none;
    border-radius: 25px; /* Rounded buttons */
    padding: 10px 24px;
    font-weight: bold;
}
.stButton > button:hover {
    background-color: #1ED760;
    color: #FFFFFF;
}

/* Text input box */
.stTextInput > div > div > input {
    background-color: #282828;
    color: #FFFFFF;
    border-radius: 25px;
    border: 1px solid #535353;
}

/* Metric box for the score */
[data-testid="stMetric"] {
    background-color: #282828;
    padding: 20px;
    border-radius: 10px;
}
[data-testid="stMetricLabel"] {
    color: #B3B3B3; /* Lighter gray for the label */
}
</style>
""", unsafe_allow_html=True)


# --- 3. SIDEBAR ---
with st.sidebar:
    st.header("About This Project")
    st.write("""
    This app predicts Spotify popularity using a Random Forest model.
    
    It was built by:
    - Nikita Belii
    - Tasianna Giordano
    - Martin Gonzalez
    - Anthony Gutierrez
    """)
    st.write("---")
    st.subheader("Project Links")
    st.write("[GitHub Repository](https://github.com/tasitasitasi/SPOTIFY)")
    st.write("[Kaggle Dataset](https://www.kaggle.com/datasets/zaheenhamidani/ultimate-spotify-tracks-db)")


# --- 4. MAIN PAGE TITLE ---
st.title("Spotify Hit Predictor 🎵")

# --- FILENAME UPDATED HERE ---
st.image("Spotify_Primary_Logo_RGB_Green.png", width=200) 

st.write("This tool predicts a song's popularity score (0-100) based on its audio features.")
st.write("---")


# --- 5. LAYOUT WITH COLUMNS ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("Enter a Song ID:")
    track_id = st.text_input(
        label="Paste a Spotify Track ID here:",
        placeholder="Example: 7lPN2DXiMsVn7XUKtREZ37"
    )
    
    if st.button("Predict Popularity"):
        if track_id:
            st.write(f"Analyzing Track ID: {track_id}...")
            
            with st.spinner('Asking the model...'):
                time.sleep(2) # Fake "work"
            
            # --- FAKE PREDICTION SECTION ---
            fake_score = 78
            # ---------------------------------
            
            st.session_state.prediction = fake_score
            st.success("Prediction complete!")

        else:
            st.error("Please enter a Track ID first.")

# --- 6. RESULTS COLUMN (NOW WITH NEW VISUALS) ---
with col2:
    st.subheader("Prediction Result:")
    
    if "prediction" in st.session_state:
        # Get the score
        score = st.session_state.prediction
        
        # --- FILENAME UPDATED HERE (for the placeholder) ---
        st.image("Spotify_Primary_Logo_RGB_Green.png", 
                 caption="Album Art (API Connection Needed)",
                 width=150)
        
        # --- 1. The Score Metric Box ---
        st.metric(
            label="Predicted Popularity Score",
            value=f"{score} / 100"
        )
        
        # --- 2. NEW: The Popularity Bar ---
        st.progress(score / 100)
        
        st.write("This score represents the estimated popularity on Spotify.")
        
        # --- 3. NEW: The Expander ---
        with st.expander("See how this score was calculated"):
            st.write("""
            The model analyzed several audio features.
            
            (This is where you can show your **Feature Importance** chart
            from your `bias.py` script to explain *why* the model
            chose this score.)
            """)

    else:
        st.info("Your predicted score will appear here once you enter a Track ID and click 'Predict'.")
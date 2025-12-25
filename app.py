import streamlit as st
import speech_recognition as sr
import librosa
import numpy as np
import os

# --- TITLE & CONFIGURATION ---
st.set_page_config(page_title="Call Sentinel", page_icon="🛡️")

st.title("🛡️ Call Sentinel: Vishing Detector")
st.markdown("""
**Context:** Detects AI-generated voice scams by analyzing audio patterns and suspicious keywords.
""")

# --- MODULE 1: THE "EARS" (Transcription) ---
def transcribe_audio(file_path):
    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(file_path) as source:
            audio_data = recognizer.record(source)
            # Uses Google Web Speech API (Free & Fast for Hackathons)
            text = recognizer.recognize_google(audio_data)
            return text
    except sr.UnknownValueError:
        return "Error: Could not understand audio (Quality too low)."
    except sr.RequestError:
        return "Error: API unavailable."
    except Exception as e:
        return f"Error processing file: {e}"

# --- MODULE 2: THE "BRAIN" (Keyword Analysis) ---
def analyze_keywords(text):
    # List of common scam phrases
    SCAM_KEYWORDS = [
        "gift card", "urgent", "wire transfer", "social security", 
        "police", "arrest", "verify your account", "otp", "password", 
        "refund", "amazon", "bank account"
    ]
    
    text_lower = text.lower()
    detected = [word for word in SCAM_KEYWORDS if word in text_lower]
    
    # Calculate Risk Score (0 to 100)
    # If 3 or more keywords are found, risk is 100%
    risk_score = min(len(detected) * 35, 100)
    
    return detected, risk_score

# --- MODULE 3: THE "LIE DETECTOR" (Audio Artifacts) ---
def analyze_audio_features(file_path):
    # Load audio file
    y, sr_rate = librosa.load(file_path)
    
    # 1. Analyze Silence/Pauses (AI sometimes has unnatural perfect silence)
    non_silent_intervals = librosa.effects.split(y, top_db=20)
    silence_ratio = 1 - (np.sum([i[1]-i[0] for i in non_silent_intervals]) / len(y))
    
    # 2. Analyze Pitch Flatness (Robotic voices are often more monotone)
    # We measure "Spectral Flatness"
    flatness = librosa.feature.spectral_flatness(y=y)
    avg_flatness = np.mean(flatness)
    
    # Heuristic Logic for Hackathon Demo:
    # High flatness often indicates noise or synthetic consistency.
    # This is a SIMPLIFIED logic for the prototype.
    fake_prob = 0
    if avg_flatness < 0.01: 
        fake_prob += 20 # Too clean
    if silence_ratio > 0.3:
        fake_prob += 30 # Too many pauses (robotic pacing)
        
    return fake_prob, avg_flatness

# --- THE UI LOGIC ---
uploaded_file = st.file_uploader("Upload Call Recording (.wav)", type=["wav"])

if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/wav')
    
    if st.button("🔍 Analyze Call for Fraud"):
        with st.spinner("Analyzing Voice Patterns & Content..."):
            
            # Save temp file
            with open("temp_audio.wav", "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # 1. Run Keyword Analysis
            transcript = transcribe_audio("temp_audio.wav")
            keywords, text_risk = analyze_keywords(transcript)
            
            # 2. Run Audio Analysis
            fake_prob, flatness = analyze_audio_features("temp_audio.wav")
            
            # 3. Calculate Final Threat Score
            # We weight keywords higher because they are definite proof of intent
            final_score = (text_risk * 0.7) + (fake_prob * 0.3)
            
            # --- DISPLAY RESULTS ---
            st.divider()
            
            # Big Result Banner
            if final_score > 50:
                st.error(f"🚨 HIGH THREAT DETECTED (Risk: {int(final_score)}%)")
            else:
                st.success(f"✅ Low Threat Detected (Risk: {int(final_score)}%)")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📝 Content Analysis")
                st.write(f"**Transcript:** \"{transcript}\"")
                if keywords:
                    st.warning(f"**Suspicious Words:** {', '.join(keywords)}")
                else:
                    st.info("No suspicious keywords found.")
            
            with col2:
                st.subheader("🤖 Voice Analysis")
                st.metric("Audio Flatness", f"{flatness:.4f}")
                if fake_prob > 0:
                    st.write("⚠️ Anomalies in speech pacing detected.")
                else:
                    st.write("Specch flow appears natural.")
from dotenv import load_dotenv
load_dotenv(override=True)

import os
import tempfile
import streamlit as st
from openai import OpenAI

# Assuming ai_agent is in the same directory
from ai_agent import get_ai_response, clear_conversation

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Mahesh AI Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =====================================================
# CUSTOM CSS
# =====================================================

st.markdown("""
<style>
.main-title{ font-size: 3rem; font-weight: 800; color: white; }
.sub-text{ color: #94a3b8; margin-bottom: 20px; }
.info-box{ background: rgba(255,255,255,0.04); padding: 18px; border-radius: 18px; border: 1px solid rgba(255,255,255,0.08); }
.stButton>button{ border-radius: 12px; height: 44px; width: 100%; background: #7c3aed; color: white; border: none; }
</style>
""", unsafe_allow_html=True)

# =====================================================
# OPENAI CLIENT
# =====================================================

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# =====================================================
# SESSION STATE
# =====================================================

if "history" not in st.session_state: st.session_state.history = []
if "last_input" not in st.session_state: st.session_state.last_input = ""
if "processing" not in st.session_state: st.session_state.processing = False
if "pending_audio" not in st.session_state: st.session_state.pending_audio = None

# =====================================================
# CORE FUNCTIONS
# =====================================================

def generate_response(user_input: str):
    if not user_input or user_input == st.session_state.last_input: return None
    st.session_state.last_input = user_input
    st.session_state.processing = True
    try:
        ai_response = get_ai_response(user_input)
        if not ai_response: ai_response = "Unable to generate response."
    except Exception as e:
        ai_response = f"Error: {str(e)}"
    st.session_state.history.append(("You", user_input))
    st.session_state.history.append(("Mahesh", ai_response))
    st.session_state.processing = False
    return ai_response

def text_to_speech(text: str):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
            speech_path = f.name
            with client.audio.speech.with_streaming_response.create(
                model="tts-1", voice="onyx", input=text
            ) as response:
                response.stream_to_file(speech_path)
        return speech_path
    except: return None

# =====================================================
# UI LAYOUT
# =====================================================

st.markdown('<div class="main-title">Mahesh AI Assistant 🤖</div>', unsafe_allow_html=True)
chat_mode = st.toggle("Enable Text Mode", value=False)

# =====================================================
# VOICE MODE WITH REVIEW LOGIC
# =====================================================

if not chat_mode:
    st.subheader("🎤 Voice Input")
    
    if st.session_state.pending_audio is None:
        audio = st.audio_input("Record your voice")
        if audio:
            st.session_state.pending_audio = audio
            st.rerun()
    else:
        st.info("Audio recorded. Please review before sending:")
        st.audio(st.session_state.pending_audio)
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("✅ Send to AI"):
                with st.spinner("Processing..."):
                    audio_data = st.session_state.pending_audio
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
                        f.write(audio_data.getbuffer())
                        path = f.name
                    with open(path, "rb") as audio_file:
                        transcript = client.audio.transcriptions.create(model="whisper-1", file=audio_file)
                    
                    user_text = transcript.text.strip()
                    st.session_state.pending_audio = None
                    ai_text = generate_response(user_text)
                    if ai_text:
                        speech = text_to_speech(ai_text)
                        if speech: st.audio(speech, format="audio/mp3")
                    st.rerun()
        with col2:
            if st.button("❌ Discard"):
                st.session_state.pending_audio = None
                st.rerun()

# =====================================================
# CHAT HISTORY
# =====================================================

for role, msg in st.session_state.history:
    with st.chat_message("user" if role == "You" else "assistant"):
        st.markdown(msg)

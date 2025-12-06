# ========================================
# AI STORY GENERATOR - STREAMLIT VERSION
# Premium Dark Theme with Neon Accents
# ========================================

import streamlit as st
from transformers import pipeline, AutoTokenizer
import torch
import soundfile as sf
import numpy as np
import random
import re
from gtts import gTTS
import time
import base64
from pathlib import Path

# Try to import Parler-TTS
try:
    from parler_tts import ParlerTTSForConditionalGeneration
    PARLER_AVAILABLE = True
except ImportError:
    try:
        from transformers import ParlerTTSForConditionalGeneration
        PARLER_AVAILABLE = True
    except ImportError:
        PARLER_AVAILABLE = False
        ParlerTTSForConditionalGeneration = None

# ========================================
# PAGE CONFIGURATION
# ========================================

st.set_page_config(
    page_title="AI Story Generator",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========================================
# CUSTOM CSS - PREMIUM DARK THEME
# ========================================

def load_css():
    st.markdown("""
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&family=Playfair+Display:wght@700&display=swap');
    
    /* Main App Background */
    .stApp {
        background: linear-gradient(135deg, #0a0a0a 0%, #1a1a1a 100%);
        font-family: 'Inter', sans-serif;
    }
    
    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Title Styling */
    .main-title {
        font-family: 'Playfair Display', serif;
        font-size: 3.5rem;
        font-weight: 700;
        text-align: center;
        background: linear-gradient(135deg, #00f5ff 0%, #00d4ff 50%, #0099ff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
        text-shadow: 0 0 30px rgba(0, 245, 255, 0.3);
    }
    
    .subtitle {
        text-align: center;
        color: #888;
        font-size: 1.1rem;
        margin-bottom: 2rem;
        font-weight: 300;
    }
    
    /* Card Containers */
    .story-card {
        background: linear-gradient(135deg, #1a1a1a 0%, #252525 100%);
        border: 1px solid #333;
        border-radius: 16px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
    }
    
    .audio-card {
        background: linear-gradient(135deg, #1a1a1a 0%, #252525 100%);
        border: 1px solid #00f5ff;
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 0 30px rgba(0, 245, 255, 0.15);
    }
    
    /* Story Text */
    .story-text {
        color: #e0e0e0;
        font-size: 1.1rem;
        line-height: 1.8;
        font-weight: 300;
        padding: 1.5rem;
        background: #0f0f0f;
        border-radius: 12px;
        border-left: 4px solid #00f5ff;
        margin: 1rem 0;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #00f5ff 0%, #0099ff 100%);
        color: #000;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0, 245, 255, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 25px rgba(0, 245, 255, 0.5);
    }
    
    /* Sidebar */
    .css-1d391kg {
        background: #0f0f0f;
        border-right: 1px solid #333;
    }
    
    /* Input Fields */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea {
        background-color: #1a1a1a;
        color: #e0e0e0;
        border: 1px solid #333;
        border-radius: 8px;
        padding: 0.75rem;
    }
    
    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus {
        border-color: #00f5ff;
        box-shadow: 0 0 10px rgba(0, 245, 255, 0.3);
    }
    
    /* Select Boxes */
    .stSelectbox > div > div {
        background-color: #1a1a1a;
        color: #e0e0e0;
        border: 1px solid #333;
        border-radius: 8px;
    }
    
    /* Progress Bar */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #00f5ff 0%, #0099ff 100%);
    }
    
    /* Info Boxes */
    .info-box {
        background: linear-gradient(135deg, #1a1a1a 0%, #252525 100%);
        border-left: 4px solid #00f5ff;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        color: #e0e0e0;
    }
    
    /* Stats */
    .stat-container {
        display: flex;
        justify-content: space-around;
        margin: 1.5rem 0;
    }
    
    .stat-box {
        background: #1a1a1a;
        border: 1px solid #333;
        border-radius: 12px;
        padding: 1rem 1.5rem;
        text-align: center;
        min-width: 120px;
    }
    
    .stat-value {
        font-size: 2rem;
        font-weight: 700;
        color: #00f5ff;
        margin-bottom: 0.25rem;
    }
    
    .stat-label {
        font-size: 0.9rem;
        color: #888;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Section Headers */
    .section-header {
        color: #00f5ff;
        font-size: 1.5rem;
        font-weight: 600;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #333;
    }
    
    /* Labels */
    .stMarkdown label {
        color: #00f5ff !important;
        font-weight: 500;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background-color: #1a1a1a;
        border: 1px solid #333;
        border-radius: 8px;
        color: #e0e0e0;
    }
    
    /* Neon Glow Effect */
    .glow {
        text-shadow: 0 0 10px rgba(0, 245, 255, 0.5);
    }
    
    /* Audio Player */
    audio {
        width: 100%;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

# ========================================
# STORY GENERATOR CLASS
# ========================================

class StorytellingGenerator:
    VOICES = {
        "Warm Storyteller": {
            "parler": "A warm, engaging narrator with moderate pace and expressive delivery, perfect for bedtime stories.",
            "gtts_tld": "com"
        },
        "Dramatic Narrator": {
            "parler": "A deep, dramatic voice with slow pace and theatrical delivery, ideal for epic tales and adventures.",
            "gtts_tld": "co.uk"
        },
        "Calm Narrator": {
            "parler": "A calm, soothing narrator with gentle pace and clear pronunciation, great for peaceful stories.",
            "gtts_tld": "com.au"
        },
        "Energetic Narrator": {
            "parler": "An energetic, upbeat voice with fast pace and enthusiastic delivery, perfect for exciting adventures.",
            "gtts_tld": "com"
        },
        "Mysterious Narrator": {
            "parler": "A mysterious, whispery voice with varying pace and enigmatic delivery, ideal for suspenseful tales.",
            "gtts_tld": "co.uk"
        }
    }
    
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    @st.cache_resource
    def load_story_model(_self):
        return pipeline(
            "text-generation",
            model="roneneldan/TinyStories-33M",
            device=0 if _self.device == "cuda" else -1
        )
    
    @st.cache_resource
    def load_parler_model(_self):
        if not PARLER_AVAILABLE or ParlerTTSForConditionalGeneration is None:
            return None, None
        try:
            model = ParlerTTSForConditionalGeneration.from_pretrained(
                "parler-tts/parler-tts-mini-v1"
            ).to(_self.device)
            tokenizer = AutoTokenizer.from_pretrained("parler-tts/parler-tts-mini-v1")
            return model, tokenizer
        except:
            return None, None
    
    def clean_text(self, text):
        spam = ['Forum', 'Topics', 'Posted by', 'Follow us', '@', 'http', 'www.']
        for marker in spam:
            if marker.lower() in text.lower():
                pos = text.lower().find(marker.lower())
                text = text[:pos]
                break
        
        text = re.sub(r'\S+@\S+\.\S+', '', text)
        text = re.sub(r'https?://\S+', '', text)
        text = re.sub(r'www\.\S+', '', text)
        text = ' '.join(text.split())
        
        if not text.endswith(('.', '!', '?')):
            last = max(text.rfind('.'), text.rfind('!'), text.rfind('?'))
            if last > len(text) * 0.6:
                text = text[:last + 1]
        
        return text.strip()
    
    def generate_story(self, prompt, target_words, style, tone):
        story_gen = self.load_story_model()
        
        style_prompts = {
            "Fairy Tale": "Once upon a time, in a magical kingdom, ",
            "Adventure": "In a land of adventure and mystery, ",
            "Sci-Fi": "In a future filled with technology and wonder, ",
            "Mystery": "On a dark and mysterious night, ",
            "Comedy": "In a funny and silly world, ",
            "Standard": random.choice([
                "Once upon a time, ", "One day, ", "Long ago, ", "In a magical land, "
            ])
        }
        
        tone_settings = {
            "Neutral": {"temp": 0.7, "rep_penalty": 1.3},
            "Serious": {"temp": 0.6, "rep_penalty": 1.4},
            "Playful": {"temp": 0.8, "rep_penalty": 1.2},
            "Dramatic": {"temp": 0.75, "rep_penalty": 1.3},
            "Whimsical": {"temp": 0.85, "rep_penalty": 1.1}
        }
        
        settings = tone_settings.get(tone, tone_settings["Neutral"])
        prefix = style_prompts.get(style, style_prompts["Standard"])
        
        if not any(prompt.lower().startswith(s.lower()[:10]) for s in style_prompts.values()):
            full_prompt = prefix + prompt
        else:
            full_prompt = prompt
        
        tokens = int(target_words * 1.3)
        
        output = story_gen(
            full_prompt,
            max_new_tokens=tokens,
            min_new_tokens=int(tokens * 0.6),
            temperature=settings["temp"],
            do_sample=True,
            top_k=50,
            top_p=0.9,
            repetition_penalty=settings["rep_penalty"],
            pad_token_id=50256,
            eos_token_id=50256,
            early_stopping=True,
            no_repeat_ngram_size=3
        )
        
        story = self.clean_text(output[0]["generated_text"])
        return story
    
    def generate_audio_parler(self, text, voice_name):
        model, tokenizer = self.load_parler_model()
        if model is None:
            return None
        
        voice = self.VOICES[voice_name]
        
        input_ids = tokenizer(voice["parler"], return_tensors="pt").input_ids.to(self.device)
        prompt_input_ids = tokenizer(text, return_tensors="pt").input_ids.to(self.device)
        
        with torch.no_grad():
            generation = model.generate(
                input_ids=input_ids,
                prompt_input_ids=prompt_input_ids,
                attention_mask=torch.ones_like(input_ids)
            )
        
        audio = generation.cpu().numpy().squeeze()
        sample_rate = model.config.sampling_rate
        
        return audio, sample_rate
    
    def generate_audio_gtts(self, text, voice_name):
        voice = self.VOICES[voice_name]
        tts = gTTS(text=text, lang='en', tld=voice["gtts_tld"], slow=False)
        filename = f"story_audio_gtts_{int(time.time())}.mp3"
        tts.save(filename)
        return filename

# ========================================
# HELPER FUNCTIONS
# ========================================

def get_audio_download_link(file_path, text="Download Audio"):
    with open(file_path, "rb") as f:
        audio_bytes = f.read()
    b64 = base64.b64encode(audio_bytes).decode()
    return f'<a href="data:audio/mp3;base64,{b64}" download="{file_path}" style="color: #00f5ff; text-decoration: none; font-weight: 600;">📥 {text}</a>'

# ========================================
# MAIN APP
# ========================================

def main():
    load_css()
    
    # Initialize session state
    if 'story' not in st.session_state:
        st.session_state.story = None
    if 'audio_files' not in st.session_state:
        st.session_state.audio_files = {}
    if 'generator' not in st.session_state:
        st.session_state.generator = StorytellingGenerator()
    
    # Header
    st.markdown('<h1 class="main-title">📚 AI Story Generator</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Create magical stories with AI-powered narration</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown('<div class="section-header">⚙️ Story Configuration</div>', unsafe_allow_html=True)
        
        prompt = st.text_area(
            "📝 Story Idea",
            placeholder="Enter your story prompt...",
            height=100,
            help="Describe what your story should be about"
        )
        
        col1, col2 = st.columns(2)
        with col1:
            length = st.selectbox(
                "📏 Length",
                ["Short (150)", "Medium (300)", "Long (500)"],
                index=1
            )
        with col2:
            style = st.selectbox(
                "🎨 Style",
                ["Standard", "Fairy Tale", "Adventure", "Sci-Fi", "Mystery", "Comedy"]
            )
        
        tone = st.selectbox(
            "🎭 Tone",
            ["Neutral", "Serious", "Playful", "Dramatic", "Whimsical"]
        )
        
        st.markdown('<div style="margin: 2rem 0;"></div>', unsafe_allow_html=True)
        generate_btn = st.button("✨ Generate Story", use_container_width=True)
    
    # Main Content
    if generate_btn and prompt:
        # Parse length
        word_count = int(length.split("(")[1].split(")")[0])
        
        # Generate story
        with st.spinner("🎨 Crafting your story..."):
            progress_bar = st.progress(0)
            for i in range(100):
                time.sleep(0.01)
                progress_bar.progress(i + 1)
            
            story = st.session_state.generator.generate_story(
                prompt, word_count, style, tone
            )
            st.session_state.story = story
            st.session_state.audio_files = {}
        
        st.success("✅ Story generated successfully!")
    
    # Display Story
    if st.session_state.story:
        st.markdown('<div class="section-header">📖 Your Story</div>', unsafe_allow_html=True)
        
        st.markdown(f'<div class="story-card"><div class="story-text">{st.session_state.story}</div></div>', 
                   unsafe_allow_html=True)
        
        # Statistics
        word_count = len(st.session_state.story.split())
        char_count = len(st.session_state.story)
        
        st.markdown(f"""
        <div class="stat-container">
            <div class="stat-box">
                <div class="stat-value">{word_count}</div>
                <div class="stat-label">Words</div>
            </div>
            <div class="stat-box">
                <div class="stat-value">{char_count}</div>
                <div class="stat-label">Characters</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Audio Generation Section
        st.markdown('<div class="section-header">🎤 Audio Narration</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            voice = st.selectbox(
                "🎙️ Narrator Voice",
                list(st.session_state.generator.VOICES.keys())
            )
        with col2:
            engine = st.selectbox(
                "🎵 Audio Engine",
                ["Parler-TTS (Premium)", "gTTS (Fast)"]
            )
        
        # Generate Audio Button
        if st.button("🎵 Generate Audio", use_container_width=True):
            engine_key = "parler" if "Parler" in engine else "gtts"
            
            with st.spinner(f"🎤 Generating audio with {engine_key.upper()}..."):
                try:
                    if engine_key == "parler":
                        result = st.session_state.generator.generate_audio_parler(
                            st.session_state.story, voice
                        )
                        if result:
                            audio, sr = result
                            filename = f"story_parler_{int(time.time())}.wav"
                            sf.write(filename, audio, sr)
                            st.session_state.audio_files["parler"] = filename
                        else:
                            st.warning("Parler-TTS not available, using gTTS instead")
                            engine_key = "gtts"
                    
                    if engine_key == "gtts":
                        filename = st.session_state.generator.generate_audio_gtts(
                            st.session_state.story, voice
                        )
                        st.session_state.audio_files["gtts"] = filename
                    
                    st.success(f"✅ Audio generated with {engine_key.upper()}!")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
        
        # Display Generated Audios
        if st.session_state.audio_files:
            st.markdown('<div class="section-header">🎧 Generated Audio Files</div>', unsafe_allow_html=True)
            
            for engine_name, filepath in st.session_state.audio_files.items():
                st.markdown(f'<div class="audio-card">', unsafe_allow_html=True)
                st.markdown(f"**{engine_name.upper()} Audio**")
                st.audio(filepath)
                st.markdown(get_audio_download_link(filepath, f"Download {engine_name.upper()} Audio"), 
                           unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Ask for regeneration
            st.markdown('<div class="info-box">💡 Want to try a different voice or engine? Select options above and click "Generate Audio" again!</div>', 
                       unsafe_allow_html=True)
    
    else:
        # Welcome Screen
        st.markdown("""
        <div class="info-box">
            <h3 style="color: #00f5ff; margin-top: 0;">👋 Welcome to AI Story Generator!</h3>
            <p>Create amazing stories with just a few clicks:</p>
            <ol style="line-height: 2;">
                <li>Enter your story idea in the sidebar</li>
                <li>Choose length, style, and tone</li>
                <li>Click "Generate Story"</li>
                <li>Review your story</li>
                <li>Generate audio narration (optional)</li>
                <li>Try different voices and engines!</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

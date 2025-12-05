import streamlit as st
from transformers import pipeline, AutoProcessor, BarkModel
import torch
import scipy.io.wavfile
import numpy as np
import random
import re
import base64
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="AI Story Generator",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for aesthetic design
st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .story-box {
        background: white;
        padding: 30px;
        border-radius: 15px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        margin: 20px 0;
    }
    .title-box {
        background: rgba(255,255,255,0.95);
        padding: 40px;
        border-radius: 20px;
        text-align: center;
        margin-bottom: 30px;
        box-shadow: 0 10px 40px rgba(0,0,0,0.2);
    }
    .success-box {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        margin: 20px 0;
    }
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 12px 30px;
        border-radius: 25px;
        font-size: 16px;
        font-weight: bold;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 20px rgba(0,0,0,0.2);
    }
    h1 {
        color: #667eea;
        font-size: 3em;
        margin-bottom: 10px;
    }
    h2 {
        color: #764ba2;
    }
    .subtitle {
        color: #666;
        font-size: 1.2em;
        margin-top: 10px;
    }
</style>
""", unsafe_allow_html=True)


class StreamlitStoryGenerator:
    """Story Generator optimized for Streamlit"""
    
    VOICE_PROFILES = {
        "Deep Narrator 🎭": {
            "preset": "v2/en_speaker_6",
            "description": "Dramatic, warm voice perfect for epic tales"
        },
        "Friendly Narrator 🌟": {
            "preset": "v2/en_speaker_3",
            "description": "Engaging, cheerful voice for adventures"
        },
        "Mysterious Narrator 🌙": {
            "preset": "v2/en_speaker_9",
            "description": "Dark, captivating voice for thrillers"
        },
        "Wise Narrator 🦉": {
            "preset": "v2/en_speaker_7",
            "description": "Calm, thoughtful voice for wisdom tales"
        }
    }
    
    STORY_LENGTHS = {
        "Quick Tale (150 words)": {"tokens": 200, "words": 150},
        "Standard Story (300 words)": {"tokens": 400, "words": 300},
        "Extended Story (500 words)": {"tokens": 650, "words": 500},
    }
    
    @staticmethod
    @st.cache_resource
    def load_models():
        """Load models with caching"""
        with st.spinner("🔮 Loading AI models... (this may take 30-60 seconds)"):
            story_gen = pipeline(
                "text-generation",
                model="roneneldan/TinyStories-33M",
                device=0 if torch.cuda.is_available() else -1
            )
            
            processor = AutoProcessor.from_pretrained("suno/bark-small")
            bark_model = BarkModel.from_pretrained("suno/bark-small")
            
            if torch.cuda.is_available():
                bark_model = bark_model.to("cuda")
                
        return story_gen, processor, bark_model
    
    def __init__(self):
        self.story_generator, self.processor, self.bark_model = self.load_models()
    
    def enhance_story_prompt(self, prompt):
        """Add storytelling elements"""
        story_starters = [
            "Once upon a time, there was ",
            "One day, ",
            "In a magical land, there lived ",
            "Long ago, ",
        ]
        
        if not any(prompt.lower().startswith(starter.lower()[:10]) for starter in story_starters):
            starter = random.choice(story_starters)
            prompt = starter + prompt.lower()
        
        return prompt
    
    def clean_story_text(self, text):
        """Clean unwanted content from generated story"""
        spam_markers = [
            'Forum', 'Topics', 'Posts', 'Last post', 'As always,', 'Thank you',
            'Follow us', 'Posted by', 'Email me', 'http', 'www.', '@'
        ]
        
        for marker in spam_markers:
            if marker.lower() in text.lower():
                pos = text.lower().find(marker.lower())
                before_text = text[:pos]
                last_period = max(before_text.rfind('.'), before_text.rfind('!'), before_text.rfind('?'))
                if last_period > len(before_text) * 0.3:
                    text = before_text[:last_period + 1]
                    break
        
        # Remove URLs, emails, mentions
        text = re.sub(r'\S+@\S+\.\S+', '', text)
        text = re.sub(r'https?://\S+', '', text)
        text = re.sub(r'www\.\S+', '', text)
        text = re.sub(r'@\w+', '', text)
        
        text = ' '.join(text.split())
        
        if text and not text.rstrip().endswith(('.', '!', '?', '"')):
            for ending in ['. ', '! ', '? ', '."', '!"', '?"']:
                last_sentence = text.rfind(ending)
                if last_sentence > len(text) * 0.6:
                    text = text[:last_sentence + 1]
                    break
        
        return text.strip()
    
    def ensure_single_complete_story(self, text):
        """Ensure single complete story"""
        text = self.clean_story_text(text)
        
        story_starters = [
            'Once upon a time', 'One day', 'Long ago',
            'In a magical land', 'There once was', 'Many years ago'
        ]
        
        starter_positions = []
        for starter in story_starters:
            pos = 0
            while True:
                pos = text.find(starter, pos)
                if pos == -1:
                    break
                if pos == 0 or (pos > 0 and text[pos-2:pos] in ['. ', '! ', '? ']):
                    starter_positions.append(pos)
                pos += len(starter)
        
        if len(starter_positions) > 1:
            first_story = text[starter_positions[0]:starter_positions[1]]
            for ending in ['. ', '! ', '? ']:
                last_end = first_story.rfind(ending)
                if last_end > len(first_story) * 0.5:
                    text = text[starter_positions[0]:starter_positions[0] + last_end + 1]
                    break
        
        sentences = text.split('.')
        if len(sentences) > 2:
            last_sentence = sentences[-1].strip()
            if len(last_sentence) < 10 or not any(last_sentence.endswith(p) for p in ['!', '?', '"', '.']):
                text = '. '.join(sentences[:-1]) + '.'
        
        text = ' '.join(text.split())
        if text and not text[-1] in '.!?"':
            text += '.'
        
        return text.strip()
    
    def generate_story(self, prompt, max_tokens=400, progress_bar=None):
        """Generate complete story"""
        enhanced_prompt = self.enhance_story_prompt(prompt)
        
        if progress_bar:
            progress_bar.progress(30)
        
        story_output = self.story_generator(
            enhanced_prompt,
            max_new_tokens=max_tokens,
            min_new_tokens=int(max_tokens * 0.6),
            temperature=0.7,
            do_sample=True,
            top_k=50,
            top_p=0.9,
            repetition_penalty=1.3,
            num_return_sequences=1,
            pad_token_id=50256,
            eos_token_id=50256,
            early_stopping=True,
            no_repeat_ngram_size=3
        )
        
        if progress_bar:
            progress_bar.progress(60)
        
        story_text = story_output[0]["generated_text"]
        story_text = self.ensure_single_complete_story(story_text)
        
        if progress_bar:
            progress_bar.progress(100)
        
        return story_text
    
    def split_into_chunks(self, text, max_chars=200):
        """Split text into natural chunks"""
        sentences = []
        current = ""
        
        for char in text:
            current += char
            if char in '.!?' and len(current) > 20:
                sentences.append(current.strip())
                current = ""
        
        if current.strip():
            sentences.append(current.strip())
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) < max_chars:
                current_chunk += " " + sentence
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def text_to_speech(self, text, voice_preset, progress_bar=None):
        """Convert text to speech"""
        chunks = self.split_into_chunks(text, max_chars=200)
        audio_segments = []
        sample_rate = None
        
        total_chunks = len(chunks)
        for i, chunk in enumerate(chunks):
            if progress_bar:
                progress = int((i / total_chunks) * 100)
                progress_bar.progress(progress)
            
            if i == 0:
                narration_chunk = f"♪ [reading a story book] ♪ {chunk}"
            else:
                narration_chunk = f"♪ {chunk}"
            
            inputs = self.processor(
                narration_chunk,
                voice_preset=voice_preset,
                return_tensors="pt"
            )
            
            if torch.cuda.is_available():
                inputs = {k: v.to("cuda") for k, v in inputs.items()}
            
            with torch.no_grad():
                speech_output = self.bark_model.generate(
                    **inputs,
                    temperature=0.6,
                    fine_temperature=0.4,
                    coarse_temperature=0.6,
                    do_sample=True,
                    semantic_temperature=0.8
                )
                audio_array = speech_output.cpu().numpy().squeeze()
            
            audio_segments.append(audio_array)
            
            if sample_rate is None:
                sample_rate = self.bark_model.generation_config.sample_rate
        
        # Combine audio
        silence = np.zeros(int(sample_rate * 0.3))
        full_audio_parts = []
        for i, segment in enumerate(audio_segments):
            full_audio_parts.append(segment)
            if i < len(audio_segments) - 1:
                full_audio_parts.append(silence)
        
        full_audio = np.concatenate(full_audio_parts)
        
        if progress_bar:
            progress_bar.progress(100)
        
        return full_audio, sample_rate


def get_audio_download_link(audio_data, sample_rate, filename="story.wav"):
    """Create download link for audio"""
    audio_int16 = (audio_data * 32767).astype(np.int16)
    
    # Save to temporary file
    scipy.io.wavfile.write(filename, sample_rate, audio_int16)
    
    with open(filename, "rb") as f:
        audio_bytes = f.read()
    
    b64 = base64.b64encode(audio_bytes).decode()
    return f'<a href="data:audio/wav;base64,{b64}" download="{filename}">📥 Download Audio Story</a>'


def main():
    # Header
    st.markdown("""
    <div class="title-box">
        <h1>📚 AI Story Generator</h1>
        <p class="subtitle">Create magical stories with AI narration</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize generator
    if 'generator' not in st.session_state:
        st.session_state.generator = StreamlitStoryGenerator()
    
    generator = st.session_state.generator
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Story Settings")
        
        # Voice selection
        st.subheader("🎙️ Choose Narrator")
        voice_name = st.selectbox(
            "Select voice",
            options=list(generator.VOICE_PROFILES.keys()),
            help="Each voice has a unique personality"
        )
        voice_info = generator.VOICE_PROFILES[voice_name]
        st.caption(voice_info["description"])
        
        st.divider()
        
        # Length selection
        st.subheader("📏 Story Length")
        length_name = st.selectbox(
            "Select length",
            options=list(generator.STORY_LENGTHS.keys()),
            help="Longer stories take more time to generate"
        )
        length_info = generator.STORY_LENGTHS[length_name]
        st.caption(f"Approximately {length_info['words']} words")
        
        st.divider()
        
        # Info
        st.info("💡 **Tips:**\n- Simple prompts work best\n- Be creative with themes\n- Generation takes 1-3 minutes")
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("✍️ Your Story Prompt")
        
        # Story input
        story_prompt = st.text_area(
            "Describe your story:",
            placeholder="Example: a brave knight discovered a dragon's lair deep in the mountains\n\nOr try: a curious cat who loved to explore magical forests",
            height=120,
            help="Describe the characters, setting, or theme you want"
        )
        
        # Example prompts
        with st.expander("Need inspiration? Try these examples:"):
            examples = [
                "a young wizard learning magic at a secret school",
                "a detective solving a mysterious case in an old mansion",
                "a friendly robot who wanted to become a painter",
                "a girl who could talk to animals in the forest",
                "a pirate searching for hidden treasure on a remote island",
                "a time traveler visiting the age of dinosaurs"
            ]
            for ex in examples:
                if st.button(f"📝 {ex}", key=ex, use_container_width=True):
                    story_prompt = ex
                    st.rerun()
    
    with col2:
        st.header("🎬 Generate")
        
        if st.button("✨ Create Story", type="primary", use_container_width=True):
            if not story_prompt or len(story_prompt.strip()) < 10:
                st.error("⚠️ Please enter a story prompt (at least 10 characters)")
            else:
                # Story generation
                st.markdown("### 📖 Generating Story...")
                progress_story = st.progress(0)
                
                with st.spinner("Crafting your tale..."):
                    story_text = generator.generate_story(
                        story_prompt,
                        max_tokens=length_info['tokens'],
                        progress_bar=progress_story
                    )
                
                st.success("✅ Story generated!")
                
                # Display story
                st.markdown("### 📜 Your Story")
                st.markdown(f"""
                <div class="story-box">
                    {story_text}
                </div>
                """, unsafe_allow_html=True)
                
                # Audio generation
                st.markdown("### 🎤 Generating Audio...")
                progress_audio = st.progress(0)
                
                with st.spinner("Creating narration..."):
                    audio_data, sample_rate = generator.text_to_speech(
                        story_text,
                        voice_info['preset'],
                        progress_bar=progress_audio
                    )
                
                st.success("✅ Audio narration complete!")
                
                # Audio player
                st.markdown("### 🔊 Listen to Your Story")
                
                # Save and display audio
                filename = "generated_story.wav"
                audio_int16 = (audio_data * 32767).astype(np.int16)
                scipy.io.wavfile.write(filename, sample_rate, audio_int16)
                
                st.audio(filename, format="audio/wav")
                
                # Download button
                st.markdown(
                    get_audio_download_link(audio_data, sample_rate, filename),
                    unsafe_allow_html=True
                )
                
                # Stats
                duration = len(audio_data) / sample_rate
                word_count = len(story_text.split())
                
                st.markdown(f"""
                <div class="success-box">
                    <h3>📊 Story Statistics</h3>
                    <p>📝 Words: {word_count}</p>
                    <p>⏱️ Duration: {duration:.1f} seconds ({duration/60:.1f} minutes)</p>
                    <p>🎙️ Voice: {voice_name}</p>
                </div>
                """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
        <div style='text-align: center; color: white; padding: 20px;'>
            <p>Powered by AI • TinyStories-33M + Bark TTS</p>
            <p>Made with ❤️ using Streamlit</p>
        </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()

# ========================================
# AI STORY GENERATOR - USER-CONTROLLED VERSION
# With Parler-TTS for natural narration
# Story first, then audio with permission
# ========================================

# STEP 1: Install dependencies
print("📦 Installing dependencies...")
print("This may take 2-3 minutes on first run...\n")

# Install transformers with Parler-TTS support
!pip install -q --upgrade transformers accelerate
!pip install -q torch scipy soundfile sentencepiece protobuf

# Try to install parler-tts package
!pip install -q git+https://github.com/huggingface/parler-tts.git

# Install gTTS as reliable fallback
!pip install -q gTTS

print("✅ Dependencies installed!\n")

# STEP 2: Import libraries
print("📚 Importing libraries...")

# Import core libraries
from transformers import pipeline, AutoTokenizer
import torch
import scipy.io.wavfile
import soundfile as sf
import numpy as np
import random
import re
from gtts import gTTS
import gc
from IPython.display import Audio, display, clear_output
import time

# Try to import Parler-TTS with fallback handling
ParlerTTSForConditionalGeneration = None
PARLER_AVAILABLE = False

try:
    from parler_tts import ParlerTTSForConditionalGeneration
    PARLER_AVAILABLE = True
    print("✅ Parler-TTS imported from parler_tts package")
except ImportError:
    try:
        from transformers import ParlerTTSForConditionalGeneration
        PARLER_AVAILABLE = True
        print("✅ Parler-TTS imported from transformers")
    except ImportError:
        print("⚠️  Parler-TTS not available - will use gTTS only")
        print("   (This is fine! gTTS works great for stories)\n")

print("✅ All libraries loaded!\n")

# ========================================
# STORYTELLING GENERATOR CLASS
# ========================================

class StorytellingGenerator:
    """User-controlled storytelling with audio generation"""
    
    # Voice descriptions
    VOICES = {
        1: {
            "name": "Warm Storyteller",
            "parler": "A warm, engaging narrator with moderate pace and expressive delivery, perfect for bedtime stories.",
            "gtts_tld": "com"
        },
        2: {
            "name": "Dramatic Narrator",
            "parler": "A deep, dramatic voice with slow pace and theatrical delivery, ideal for epic tales and adventures.",
            "gtts_tld": "co.uk"
        },
        3: {
            "name": "Calm Narrator",
            "parler": "A calm, soothing narrator with gentle pace and clear pronunciation, great for peaceful stories.",
            "gtts_tld": "com.au"
        },
        4: {
            "name": "Energetic Narrator",
            "parler": "An energetic, upbeat voice with fast pace and enthusiastic delivery, perfect for exciting adventures.",
            "gtts_tld": "com"
        },
        5: {
            "name": "Mysterious Narrator",
            "parler": "A mysterious, whispery voice with varying pace and enigmatic delivery, ideal for suspenseful tales.",
            "gtts_tld": "co.uk"
        }
    }
    
    def __init__(self):
        """Initialize models"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.story_gen = None
        self.parler_model = None
        self.parler_tokenizer = None
        self.parler_loaded = False
        self.current_story = None
        
    def load_story_model(self):
        """Load story generation model"""
        if self.story_gen is not None:
            return
            
        print(f"🖥️  Using device: {self.device.upper()}")
        print("📖 Loading story generator...")
        start = time.time()
        
        self.story_gen = pipeline(
            "text-generation",
            model="roneneldan/TinyStories-33M",
            device=0 if self.device == "cuda" else -1
        )
        
        print(f"✅ Story generator loaded in {time.time()-start:.1f}s\n")
    
    def load_parler_model(self):
        """Load Parler-TTS model only when needed"""
        if self.parler_loaded or not PARLER_AVAILABLE or ParlerTTSForConditionalGeneration is None:
            return False
            
        try:
            print("\n🎤 Loading Parler-TTS (storytelling model)...")
            print("   This may take 30-60 seconds...")
            start = time.time()
            
            self.parler_model = ParlerTTSForConditionalGeneration.from_pretrained(
                "parler-tts/parler-tts-mini-v1"
            ).to(self.device)
            
            self.parler_tokenizer = AutoTokenizer.from_pretrained(
                "parler-tts/parler-tts-mini-v1"
            )
            
            self.parler_loaded = True
            print(f"✅ Parler-TTS loaded in {time.time()-start:.1f}s")
            print("   Ready for audiobook-style narration!\n")
            return True
            
        except Exception as e:
            print(f"⚠️  Parler-TTS failed to load: {str(e)[:80]}")
            print("   Will use gTTS instead\n")
            return False
    
    def clean_text(self, text):
        """Clean generated story"""
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
    
    def generate_story(self, prompt, target_words=300, style="standard", tone="neutral"):
        """Generate story with user customization"""
        self.load_story_model()
        
        print(f"\n📝 Generating story...")
        print(f"   Length: ~{target_words} words")
        print(f"   Style: {style}")
        print(f"   Tone: {tone}\n")
        
        # Apply style modifications to prompt
        style_prompts = {
            "fairy_tale": "Once upon a time, in a magical kingdom, ",
            "adventure": "In a land of adventure and mystery, ",
            "sci_fi": "In a future filled with technology and wonder, ",
            "mystery": "On a dark and mysterious night, ",
            "comedy": "In a funny and silly world, ",
            "standard": random.choice([
                "Once upon a time, ", 
                "One day, ", 
                "Long ago, ", 
                "In a magical land, "
            ])
        }
        
        # Apply tone to temperature
        tone_settings = {
            "neutral": {"temp": 0.7, "rep_penalty": 1.3},
            "serious": {"temp": 0.6, "rep_penalty": 1.4},
            "playful": {"temp": 0.8, "rep_penalty": 1.2},
            "dramatic": {"temp": 0.75, "rep_penalty": 1.3},
            "whimsical": {"temp": 0.85, "rep_penalty": 1.1}
        }
        
        settings = tone_settings.get(tone, tone_settings["neutral"])
        prefix = style_prompts.get(style, style_prompts["standard"])
        
        # Add prefix if prompt doesn't already have one
        if not any(prompt.lower().startswith(s.lower()[:10]) for s in style_prompts.values()):
            full_prompt = prefix + prompt
        else:
            full_prompt = prompt
        
        tokens = int(target_words * 1.3)
        start = time.time()
        
        output = self.story_gen(
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
        word_count = len(story.split())
        
        print(f"✅ Story generated in {time.time()-start:.1f}s")
        print(f"   Words: {word_count}\n")
        
        self.current_story = story
        return story
    
    def text_to_speech_parler(self, text, voice_num=1):
        """Generate audio with Parler-TTS"""
        if not self.load_parler_model():
            return None
        
        voice = self.VOICES[voice_num]
        print(f"\n🎤 Generating audio with Parler-TTS...")
        print(f"   Voice: {voice['name']}")
        print(f"   Style: {voice['parler'][:60]}...")
        
        start = time.time()
        
        input_ids = self.parler_tokenizer(voice["parler"], return_tensors="pt").input_ids.to(self.device)
        prompt_input_ids = self.parler_tokenizer(text, return_tensors="pt").input_ids.to(self.device)
        
        with torch.no_grad():
            generation = self.parler_model.generate(
                input_ids=input_ids,
                prompt_input_ids=prompt_input_ids,
                attention_mask=torch.ones_like(input_ids)
            )
        
        audio = generation.cpu().numpy().squeeze()
        sample_rate = self.parler_model.config.sampling_rate
        
        duration = len(audio) / sample_rate
        print(f"✅ Audio generated in {time.time()-start:.1f}s")
        print(f"   Duration: {duration:.1f}s\n")
        
        return audio, sample_rate
    
    def text_to_speech_gtts(self, text, voice_num=1):
        """Generate audio with gTTS"""
        voice = self.VOICES[voice_num]
        print(f"\n🎤 Generating audio with gTTS...")
        print(f"   Voice: {voice['name']}")
        
        start = time.time()
        
        tts = gTTS(text=text, lang='en', tld=voice["gtts_tld"], slow=False)
        filename = "story_audio.mp3"
        tts.save(filename)
        
        duration = len(text.split()) / 2.5
        print(f"✅ Audio generated in {time.time()-start:.1f}s")
        print(f"   Estimated duration: {duration:.1f}s\n")
        
        return filename
    
    def generate_audio(self, story=None, voice_num=1, engine="auto"):
        """Generate audio for story"""
        if story is None:
            story = self.current_story
            
        if story is None:
            print("❌ No story available. Generate a story first!")
            return None
        
        # Determine which engine to use
        if engine == "auto":
            use_parler = PARLER_AVAILABLE and ParlerTTSForConditionalGeneration is not None
        elif engine == "parler":
            use_parler = PARLER_AVAILABLE and ParlerTTSForConditionalGeneration is not None
            if not use_parler:
                print("⚠️  Parler-TTS not available, using gTTS")
        else:
            use_parler = False
        
        # Generate audio
        if use_parler:
            try:
                audio, rate = self.text_to_speech_parler(story, voice_num)
                if audio is not None:
                    filename = "story_narration.wav"
                    sf.write(filename, audio, rate)
                    print(f"💾 Saved: {filename}")
                    display(Audio(filename, autoplay=False))
                    return filename
            except Exception as e:
                print(f"⚠️  Parler-TTS error: {str(e)[:80]}")
                print("🔄 Switching to gTTS...\n")
        
        # Use gTTS (fallback or by choice)
        filename = self.text_to_speech_gtts(story, voice_num)
        display(Audio(filename, autoplay=False))
        return filename


# ========================================
# USER INTERFACE FUNCTIONS
# ========================================

def create_story_interactive():
    """Interactive story creation with user control"""
    gen = StorytellingGenerator()
    
    print("\n" + "="*70)
    print("📚 AI STORY GENERATOR - INTERACTIVE MODE")
    print("="*70)
    
    # Step 1: Get story parameters
    print("\n📝 STEP 1: STORY CUSTOMIZATION")
    print("-" * 70)
    
    prompt = input("\n💡 Enter your story idea/prompt:\n   → ").strip()
    if not prompt:
        prompt = "a brave adventurer discovering a magical world"
        print(f"   (Using default: {prompt})")
    
    print("\n📏 Choose story length:")
    print("   1. Short (150 words)")
    print("   2. Medium (300 words)")
    print("   3. Long (500 words)")
    length_choice = input("   → ").strip()
    length_map = {"1": 150, "2": 300, "3": 500}
    target_words = length_map.get(length_choice, 300)
    
    print("\n🎨 Choose story style:")
    print("   1. Fairy Tale (classic once upon a time)")
    print("   2. Adventure (exciting journey)")
    print("   3. Sci-Fi (futuristic)")
    print("   4. Mystery (suspenseful)")
    print("   5. Comedy (funny)")
    print("   6. Standard (flexible)")
    style_choice = input("   → ").strip()
    style_map = {
        "1": "fairy_tale", "2": "adventure", "3": "sci_fi",
        "4": "mystery", "5": "comedy", "6": "standard"
    }
    style = style_map.get(style_choice, "standard")
    
    print("\n🎭 Choose story tone:")
    print("   1. Neutral (balanced)")
    print("   2. Serious (formal)")
    print("   3. Playful (fun)")
    print("   4. Dramatic (intense)")
    print("   5. Whimsical (imaginative)")
    tone_choice = input("   → ").strip()
    tone_map = {
        "1": "neutral", "2": "serious", "3": "playful",
        "4": "dramatic", "5": "whimsical"
    }
    tone = tone_map.get(tone_choice, "neutral")
    
    # Step 2: Generate story
    print("\n" + "="*70)
    print("📖 STEP 2: GENERATING STORY")
    print("="*70)
    
    story = gen.generate_story(prompt, target_words, style, tone)
    
    # Display story
    print("\n" + "="*70)
    print("📖 YOUR GENERATED STORY")
    print("="*70)
    print(f"\n{story}\n")
    print("="*70)
    print(f"📊 Statistics: {len(story.split())} words, {len(story)} characters")
    print("="*70)
    
    # Step 3: Ask for audio permission
    print("\n" + "="*70)
    print("🎤 STEP 3: AUDIO NARRATION (OPTIONAL)")
    print("="*70)
    
    audio_choice = input("\n🔊 Would you like to generate audio narration? (yes/no): ").strip().lower()
    
    if audio_choice in ['yes', 'y', 'yeah', 'sure', 'ok']:
        print("\n🎙️  Choose narrator voice:")
        print("   1. Warm Storyteller (bedtime stories)")
        print("   2. Dramatic Narrator (epic tales)")
        print("   3. Calm Narrator (peaceful)")
        print("   4. Energetic Narrator (exciting)")
        print("   5. Mysterious Narrator (suspenseful)")
        voice_choice = input("   → ").strip()
        voice_num = int(voice_choice) if voice_choice in "12345" else 1
        
        print("\n🎤 Choose audio engine:")
        print("   1. Parler-TTS (best quality, slower)")
        print("   2. gTTS (fast, reliable)")
        print("   3. Auto (use best available)")
        engine_choice = input("   → ").strip()
        engine_map = {"1": "parler", "2": "gtts", "3": "auto"}
        engine = engine_map.get(engine_choice, "auto")
        
        print("\n" + "="*70)
        print("🎵 GENERATING AUDIO")
        print("="*70)
        
        filename = gen.generate_audio(story, voice_num, engine)
        
        print("\n" + "="*70)
        print("✅ COMPLETE!")
        print("="*70)
        print(f"📁 Audio file: {filename}")
        print("🎧 Play the audio above")
        print("📥 Download from Colab's file browser")
        print("="*70)
    else:
        print("\n✅ Story generation complete!")
        print("   You can generate audio later by calling:")
        print("   gen.generate_audio(story, voice_num=1, engine='auto')")
    
    return gen, story


def quick_story(prompt, words=300, style="standard", tone="neutral", 
                voice=1, audio=True, engine="auto"):
    """Quick story generation with single function"""
    gen = StorytellingGenerator()
    
    print("\n" + "="*70)
    print("📚 QUICK STORY GENERATION")
    print("="*70)
    
    # Generate story
    story = gen.generate_story(prompt, words, style, tone)
    
    # Display story
    print("\n" + "="*70)
    print("📖 YOUR STORY")
    print("="*70)
    print(f"\n{story}\n")
    print("="*70)
    
    # Generate audio if requested
    if audio:
        print("\n🔊 Audio generation enabled")
        audio_file = gen.generate_audio(story, voice, engine)
    else:
        print("\n⏭️  Audio generation skipped")
        audio_file = None
    
    return gen, story, audio_file


# ========================================
# MAIN EXECUTION
# ========================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("📚 AI STORY GENERATOR - USER CONTROLLED")
    print("="*70)
    print("\n✨ Features:")
    print("   • Generate story FIRST, then decide on audio")
    print("   • Full customization of style, tone, and length")
    print("   • Choose voice and audio engine")
    print("   • Parler-TTS for audiobook quality or gTTS for speed")
    
    print("\n" + "="*70)
    print("🚀 USAGE OPTIONS")
    print("="*70)
    
    print("\n1️⃣  INTERACTIVE MODE (Recommended):")
    print("   gen, story = create_story_interactive()")
    print("   → Full control with step-by-step prompts")
    
    print("\n2️⃣  QUICK MODE (One-liner):")
    print("   gen, story, audio = quick_story(")
    print("       prompt='a dragon adventure',")
    print("       words=300,")
    print("       style='adventure',")
    print("       tone='dramatic',")
    print("       voice=2,")
    print("       audio=True,")
    print("       engine='auto'")
    print("   )")
    
    print("\n3️⃣  MANUAL CONTROL:")
    print("   gen = StorytellingGenerator()")
    print("   story = gen.generate_story('your prompt', 300, 'fairy_tale', 'playful')")
    print("   # Review story, then:")
    print("   audio = gen.generate_audio(story, voice_num=1, engine='parler')")
    
    print("\n" + "="*70)
    print("📖 STYLE OPTIONS: fairy_tale, adventure, sci_fi, mystery, comedy, standard")
    print("🎭 TONE OPTIONS: neutral, serious, playful, dramatic, whimsical")
    print("🎙️  VOICES: 1-5 (Warm, Dramatic, Calm, Energetic, Mysterious)")
    print("🎤 ENGINES: parler (best), gtts (fast), auto (smart choice)")
    print("="*70)
    
    # Auto-start interactive mode
    print("\n🎬 Starting Interactive Mode in 3 seconds...")
    print("   (Or interrupt and call functions manually)")
    time.sleep(3)
    
    gen, story = create_story_interactive()


# ========================================
# EXAMPLES FOR REFERENCE
# ========================================

"""
📚 EXAMPLE USAGE:

# 1. Interactive (best for beginners)
gen, story = create_story_interactive()

# 2. Quick story with audio
gen, story, audio = quick_story(
    "a robot learning to love",
    words=200,
    style="sci_fi",
    tone="playful",
    voice=4,
    audio=True,
    engine="auto"
)

# 3. Story only, audio later
gen, story, _ = quick_story(
    "a mysterious castle",
    words=300,
    style="mystery",
    audio=False  # No audio yet
)
# Later, after reading the story:
audio = gen.generate_audio(story, voice=5, engine="parler")

# 4. Full manual control
gen = StorytellingGenerator()
story = gen.generate_story(
    prompt="a magical cat",
    target_words=250,
    style="fairy_tale",
    tone="whimsical"
)
print(story)
# Decide if you want audio:
audio = gen.generate_audio(story, voice_num=1, engine="auto")
"""

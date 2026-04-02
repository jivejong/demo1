import streamlit as st
import json
import io
import re
import os
import time
from PIL import Image
from gtts import gTTS

# ── DEPENDENCIES ──────────────────────────────────────────────────────────────
# pip install groq google-generativeai gtts pillow streamlit

try:
    from groq import Groq
except ImportError:
    st.error("Missing dependency: run `pip install groq`")
    st.stop()

try:
    from google import genai
except ImportError:
    st.error("Missing dependency: run `pip install google-generativeai`")
    st.stop()

# ── 1. PAGE CONFIG ────────────────────────────────────────────────────────────
st.set_page_config(page_title="Agentic Poet", page_icon="📸", layout="wide")

# ── 2. API KEY SETUP ──────────────────────────────────────────────────────────
# Gemini: vision only (Groq is text-only, so we keep Gemini for the Visionary)
gemini_key = (
    st.secrets.get("GENAI_API_KEY")
    or st.secrets.get("GEMINI_API_KEY")
)
if not gemini_key:
    st.error("Missing Gemini API key in Streamlit Secrets (GENAI_API_KEY or GEMINI_API_KEY).")
    st.stop()

if "GROQ_API_KEY" not in st.secrets:
    st.error("Missing 'GROQ_API_KEY' in Streamlit Secrets.")
    st.stop()

gemini_client = genai.Client(api_key=gemini_key)
groq_client   = Groq(api_key=st.secrets["GROQ_API_KEY"].strip())

GROQ_MODEL   = "llama-3.3-70b-versatile"
GEMINI_MODEL = "gemini-1.5-flash"        # only used for vision

# ── 3. SESSION STATE ──────────────────────────────────────────────────────────
for key, default in [("camera_key", 0), ("final_output", None)]:
    if key not in st.session_state:
        st.session_state[key] = default

# ── 4. UTILITIES ──────────────────────────────────────────────────────────────

def clean_json(text: str) -> str:
    """Extract the first {...} block from a string (strips markdown fences)."""
    match = re.search(r'\{.*\}', text, re.DOTALL)
    return match.group(0) if match else text.strip()


def call_groq(prompt: str, max_tokens: int = 400) -> str:
    """Single Groq call with JSON mode enforced — no markdown stripping needed."""
    response = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content


def keyword_precheck(entities: list, poem: str) -> bool:
    """
    Fast Python check BEFORE burning a Groq call on moderation.
    Returns True if at least one entity keyword appears in the poem.
    This handles the obvious pass case cheaply.
    """
    poem_lower = poem.lower()
    return any(str(e).lower() in poem_lower for e in entities)


def get_mood_music(mood: str):
    """Maestro Agent: load MP3 from audio_library/ by mood name."""
    path = f"audio_library/{mood.lower()}.mp3"
    if os.path.exists(path):
        with open(path, "rb") as f:
            return f.read()
    return None


# ── 5. THE AGENTS ─────────────────────────────────────────────────────────────

def agent_visionary(image_file) -> dict:
    """
    AGENT 1 — VISIONARY (Gemini, vision required)
    Analyzes the image and returns structured scene data.
    This is the ONLY Gemini call in the pipeline.
    """
    raw_img = Image.open(image_file)
    raw_img.thumbnail((1024, 1024))  # resize before sending — saves tokens

    prompt = """
    You are the Visionary agent. Analyze this image carefully.
    Return ONLY a raw JSON object with these exact keys:
    {
      "description": "A 2-sentence factual summary of the scene.",
      "entities": ["list", "of", "key", "objects", "or", "subjects"],
      "setting": "brief scene setting (e.g. urban street at dusk)"
    }
    """
    response = gemini_client.models.generate_content(
        model=GEMINI_MODEL,
        contents=[prompt, raw_img]
    )
    return json.loads(clean_json(response.text))


def agent_bard(description: str, setting: str, entities: list) -> str:
    """
    AGENT 2 — BARD (Groq, text only)
    Receives the Visionary's structured output and composes a poem.
    Separated from the Visionary so the handoff is explicit and visible.
    """
    prompt = f"""
    You are the Bard agent. A Visionary has analyzed an image and handed you this data:

    Scene Description: {description}
    Setting: {setting}
    Key Entities: {entities}

    Your task: Write a 4-line rhythmic poem that is clearly inspired by
    the scene and references at least one of the key entities.

    Return ONLY a JSON object:
    {{"poem": "line one / line two / line three / line four"}}
    """
    data = json.loads(call_groq(prompt, max_tokens=200))
    return data["poem"]


def agent_moderator(entities: list, poem: str, description: str) -> dict:
    """
    AGENT 3 — MODERATOR (Groq, text only)
    Two-stage verification:
      Stage A: Fast Python keyword check (free).
      Stage B: LLM semantic check only if Stage A fails.
    Returns {verified: bool, reason: str}
    """
    # Stage A: cheap keyword pre-check
    if keyword_precheck(entities, poem):
        return {
            "verified": True,
            "reason": "Python keyword check passed — entity found in poem."
        }

    # Stage B: semantic LLM check (only runs if Stage A failed)
    prompt = f"""
    You are the Moderator agent. Your job is strict quality control.

    Scene entities: {entities}
    Scene description: {description}
    Poem to review: "{poem}"

    Does this poem thematically relate to the scene entities and description?
    Be strict. If the poem could describe ANY scene, it fails.

    Return ONLY a JSON object:
    {{"verified": true_or_false, "reason": "one sentence explanation"}}
    """
    return json.loads(call_groq(prompt, max_tokens=150))


def agent_sentiment(poem: str, description: str) -> tuple:
    """
    AGENT 4 — SENTIMENT / MAESTRO (Groq, text only)
    Reads the approved poem and picks the best mood for music selection.
    Separated from the Bard so mood analysis is its own visible step.
    """
    prompt = f"""
    You are the Sentiment agent. Analyze the emotional tone of this poem
    and the scene it describes, then select the single best mood category.

    Poem: "{poem}"
    Scene: "{description}"

    Choose EXACTLY one mood from: MELANCHOLY, WHIMSICAL, EPIC, EERIE

    Return ONLY a JSON object:
    {{"mood": "ONE_MOOD", "reason": "one sentence justification"}}
    """
    data = json.loads(call_groq(prompt, max_tokens=100))
    return data["mood"].upper(), data.get("reason", "")


# ── 6. ORCHESTRATOR ───────────────────────────────────────────────────────────

MAX_BARD_RETRIES = 2  # how many times to ask Bard to rewrite before giving up

def run_pipeline(image_file):
    """
    Full multi-agent pipeline with real closed-loop retry on Moderator rejection.

    Flow:
      Visionary (Gemini) -> Bard (Groq) -> Moderator (Groq)
           ^___________________________|  (retry up to MAX_BARD_RETRIES times)
      -> Sentiment (Groq) -> Narrator (gTTS) -> Maestro (local files)
    """
    with st.status("Orchestrating Multi-Agent Workflow...", expanded=True) as status:

        # ── AGENT 1: VISIONARY ─────────────────────────────────────────────
        st.write("🔍 **Visionary Agent**: Analyzing image...")
        try:
            scene = agent_visionary(image_file)
        except Exception as e:
            st.error(f"Visionary failed: {e}")
            return None

        st.write(f"   → Scene understood. Entities: `{', '.join(scene['entities'])}`")

        # ── AGENT 2 + 3: BARD → MODERATOR (closed-loop retry) ─────────────
        poem       = None
        mod_result = None
        attempt    = 0

        while attempt <= MAX_BARD_RETRIES:
            attempt += 1
            label = f"(Attempt {attempt})" if attempt > 1 else ""

            st.write(f"✍️ **Bard Agent**: Composing poem... {label}")
            try:
                poem = agent_bard(scene["description"], scene["setting"], scene["entities"])
            except Exception as e:
                st.error(f"Bard failed: {e}")
                return None

            st.write("   → Poem drafted. Sending to Moderator...")
            st.write("⚖️ **Moderator Agent**: Verifying poem relevance...")

            try:
                mod_result = agent_moderator(scene["entities"], poem, scene["description"])
            except Exception as e:
                st.warning(f"Moderator error on attempt {attempt}: {e}. Retrying Bard...")
                mod_result = {"verified": False, "reason": f"Moderator error: {e}"}

            if mod_result["verified"]:
                st.write(f"   ✅ Approved: {mod_result['reason']}")
                break
            else:
                if attempt <= MAX_BARD_RETRIES:
                    st.write(f"   ⚠️ Rejected: {mod_result['reason']} — asking Bard to rewrite...")
                else:
                    st.write(f"   ⚠️ Rejected after {MAX_BARD_RETRIES + 1} attempts — proceeding with best effort.")

        # ── AGENT 4: SENTIMENT / MAESTRO ──────────────────────────────────
        st.write("🎭 **Sentiment Agent**: Determining mood for music selection...")
        try:
            mood, mood_reason = agent_sentiment(poem, scene["description"])
        except Exception as e:
            st.warning(f"Sentiment agent failed ({e}), defaulting to WHIMSICAL.")
            mood, mood_reason = "WHIMSICAL", "Default fallback."

        st.write(f"   → Mood detected: **{mood}** — {mood_reason}")

        # ── NARRATOR: gTTS (local, no API call) ───────────────────────────
        st.write("🎙️ **Narrator**: Recording poem recitation...")
        tts      = gTTS(text=poem, lang='en')
        voice_io = io.BytesIO()
        tts.write_to_fp(voice_io)

        # ── MAESTRO: local file lookup ─────────────────────────────────────
        music_bytes = get_mood_music(mood)

        status.update(label="✅ All Agents Complete!", state="complete", expanded=False)

        return {
            "scene":       scene,
            "poem":        poem,
            "moderator":   mod_result,
            "mood":        mood,
            "mood_reason": mood_reason,
            "voice":       voice_io.getvalue(),
            "music":       music_bytes,
        }


# ── 7. UI ─────────────────────────────────────────────────────────────────────

st.title("📸 The Agentic Poet")
st.caption("A Multimodal AI Performance: Vision → Poetry → Moderation → Sound")
st.markdown("---")

# Sidebar: architecture explainer (great for portfolio demos)
with st.sidebar:
    st.header("🏗️ Agent Architecture")
    st.markdown("""
    | Agent | Model | Role |
    |---|---|---|
    | 🔍 Visionary | Gemini Flash | Image → Scene data |
    | ✍️ Bard | Groq Llama 3.3 | Scene → Poem |
    | ⚖️ Moderator | Groq Llama 3.3 | Verify poem relevance |
    | 🎭 Sentiment | Groq Llama 3.3 | Poem → Mood |
    | 🎙️ Narrator | gTTS (local) | Poem → Voice |
    | 🎵 Maestro | Local files | Mood → Music |
    """)
    st.divider()
    st.caption("Gemini: vision only | Groq: all text agents")
    st.caption("Moderator retries Bard up to 2x on failure")

camera_img = st.camera_input(
    "Take a photo to begin",
    key=f"cam_{st.session_state.camera_key}"
)

# Process only when new image arrives and no output yet
if camera_img and st.session_state.final_output is None:
    results = run_pipeline(camera_img)
    if results:
        st.session_state.final_output = results
        st.rerun()

# ── RESULTS DISPLAY ───────────────────────────────────────────────────────────
if st.session_state.final_output:
    out = st.session_state.final_output

    col1, col2 = st.columns(2)

    with col1:
        with st.expander("🔍 Visionary Report", expanded=True):
            st.write(out["scene"]["description"])
            st.caption(f"Setting: {out['scene']['setting']}")
            st.caption(f"Entities: {', '.join(out['scene']['entities'])}")

    with col2:
        with st.expander("✍️ Bard's Poem", expanded=True):
            st.info(out["poem"])
            verified = out["moderator"]["verified"]
            icon     = "✅" if verified else "⚠️"
            st.caption(f"{icon} Moderator: {out['moderator']['reason']}")

    st.divider()
    st.subheader(f"🎭 The Performance  ·  Mood: **{out['mood']}**")
    st.caption(f"_{out['mood_reason']}_")

    if out["music"]:
        c1, c2 = st.columns(2)
        with c1:
            st.write("**🎙️ Narrator**")
            st.audio(out["voice"], format="audio/mp3")
        with c2:
            st.write("**🎵 Maestro**")
            st.audio(out["music"], format="audio/mp3")

        if st.button("▶️ PLAY COMBINED PERFORMANCE", use_container_width=True):
            st.components.v1.html(
                """
                <script>
                    var audios = window.parent.document.querySelectorAll('audio');
                    audios.forEach(a => { a.currentTime = 0; a.play(); });
                </script>
                """,
                height=0,
            )
            st.balloons()
    else:
        st.warning("⚠️ No music file found in /audio_library/ for this mood. Playing voice only.")
        st.audio(out["voice"], format="audio/mp3")

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("🔄 START OVER", type="primary", use_container_width=True):
        st.session_state.final_output = None
        st.session_state.camera_key += 1
        st.rerun()

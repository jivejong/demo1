import streamlit as st
import pandas as pd
import random
import json
import time

# --- DEPENDENCY: Groq client ---
# pip install groq
try:
    from groq import Groq
except ImportError:
    st.error("Missing dependency: run `pip install groq` then restart.")
    st.stop()

# ──────────────────────────────────────────────
# 1. DATA LOADING
# ──────────────────────────────────────────────
@st.cache_data
def load_nutrition_data():
    try:
        df = pd.read_csv("nutritional_data.csv")
        df.columns = df.columns.str.strip()
        for col in ["Sugar", "Cholesterol"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
        return df
    except Exception as e:
        st.error(f"Error loading CSV: {e}")
        return None


df_nutrition = load_nutrition_data()

# ──────────────────────────────────────────────
# 2. API CONFIGURATION
# ──────────────────────────────────────────────
st.set_page_config(page_title="Snack-Chain: Optimized", page_icon="🥗")
st.title("🥗 Snack-Chain: Optimized Agentic RAG")

if "GROQ_API_KEY" not in st.secrets:
    st.error(
        "Missing 'GROQ_API_KEY' in .streamlit/secrets.toml. "
        "Get a free key at https://console.groq.com"
    )
    st.stop()

client = Groq(api_key=st.secrets["GROQ_API_KEY"].strip())

# ──────────────────────────────────────────────
# 3. RATE LIMITER  (replaces flat time.sleep(5))
#    Only waits the time actually remaining since
#    the last call — saves 2-4 s per negotiation.
# ──────────────────────────────────────────────
_last_call: dict = {"t": 0.0}
MIN_CALL_GAP = 2.0  # Groq free tier is generous; 2 s gap is plenty


def _rate_limit():
    elapsed = time.time() - _last_call["t"]
    if elapsed < MIN_CALL_GAP:
        time.sleep(MIN_CALL_GAP - elapsed)
    _last_call["t"] = time.time()


# ──────────────────────────────────────────────
# 4. HELPER: slim a CSV row to only needed fields
#    Stops us sending 30+ columns to the LLM.
# ──────────────────────────────────────────────
def slim_record(record: dict) -> dict:
    return {
        "name": record.get("Description", "Unknown"),
        "sugar_g": record.get("Sugar", 0),
        "cholesterol_mg": record.get("Cholesterol", 0),
    }


# ──────────────────────────────────────────────
# 5. LLM AGENT ENGINE
#    Uses Groq + response_format JSON mode →
#    no markdown stripping needed.
# ──────────────────────────────────────────────
def call_agent(role_name: str, avatar: str, instruction: str, context: str) -> dict:
    """Calls the LLM, renders to chat UI, returns parsed dict."""
    with st.chat_message(role_name, avatar=avatar):
        placeholder = st.empty()
        placeholder.markdown(f"*{role_name} is thinking…*")

        _rate_limit()  # smart wait — only as long as needed

        # Compressed prompt — same info, fewer tokens
        prompt = (
            f"Role: {instruction}\n"
            f"Data: {context}\n"
            'Reply ONLY with JSON: {"action":"APPROVE or REJECT","item":"...","reasoning":"...","monologue":"..."}'
        )

        try:
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",          # fast, free, capable
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},  # enforces valid JSON
                max_tokens=300,                   # agents don't need long replies
            )
            data = json.loads(response.choices[0].message.content)

            placeholder.markdown(f"**{data.get('reasoning', '—')}**")
            with st.expander(f"View {role_name}'s Thought Process"):
                st.info(data.get("monologue", "No internal thoughts recorded."))
                if data.get("item"):
                    st.caption(f"Targeted Item: {data['item']}")
            return data

        except Exception as e:
            placeholder.error(f"Agent error: {e}")
            return {"action": "REJECT", "item": "None", "reasoning": "Communication error."}


# ──────────────────────────────────────────────
# 6. PARENT AGENT — pure Python, zero LLM tokens
#    The audit decision is deterministic; no need
#    to burn an API call on rule-based logic.
# ──────────────────────────────────────────────
def parent_audit(
    snack_facts: dict,
    sugar_limit: float,
    chol_limit: float,
    chaos_context: str,
    chosen_name: str,
) -> dict:
    sugar = snack_facts.get("Sugar", 0)
    chol = snack_facts.get("Cholesterol", 0)
    approved = sugar <= sugar_limit and chol <= chol_limit

    if approved:
        reasoning = (
            f"✅ {chosen_name} passes: Sugar={sugar}g ≤ {sugar_limit}g, "
            f"Cholesterol={chol}mg ≤ {chol_limit}mg."
        )
    else:
        violations = []
        if sugar > sugar_limit:
            violations.append(f"Sugar {sugar}g > limit {sugar_limit}g")
        if chol > chol_limit:
            violations.append(f"Cholesterol {chol}mg > limit {chol_limit}mg")
        reasoning = f"❌ {chosen_name} denied — {'; '.join(violations)}."

    if chaos_context:
        reasoning += f" (Grandparent pressure noted and ignored.)"

    with st.chat_message("Parent", avatar="👨‍⚖️"):
        st.markdown(f"**{reasoning}**")
        with st.expander("View Parent's Audit Logic"):
            st.code(
                f"Sugar:       {sugar}g   (limit {sugar_limit}g)\n"
                f"Cholesterol: {chol}mg  (limit {chol_limit}mg)\n"
                f"Decision:    {'APPROVE' if approved else 'REJECT'}",
                language="text",
            )

    return {"action": "APPROVE" if approved else "REJECT", "reasoning": reasoning}


# ──────────────────────────────────────────────
# 7. SIDEBAR CONTROLS
# ──────────────────────────────────────────────
with st.sidebar:
    st.header("⚖️ Compliance Rules")
    sugar_limit = st.slider("Max Sugar (g)", 5, 30, 15)
    chol_limit = st.slider("Max Cholesterol (mg)", 20, 100, 50)
    st.divider()
    st.caption("Snack-Chain Optimized v4.0")
    st.caption("LLM: Groq · llama3-8b-8192")
    st.caption("Parent agent: pure Python (no tokens used)")

# ──────────────────────────────────────────────
# 8. MAIN UI & ORCHESTRATION
# ──────────────────────────────────────────────
user_snack_idea = st.text_input("Child: 'I really want to eat…'", "Chocolate")

if st.button("Start Live Negotiation"):
    if df_nutrition is None:
        st.error("Pantry data not loaded.")
        st.stop()

    st.subheader("Live Negotiation Thread")

    # ── RAG STEP 1: RETRIEVAL ──────────────────
    mask = df_nutrition["Description"].str.contains(user_snack_idea, case=False, na=False)
    pantry_matches = df_nutrition[mask].head(3)

    if pantry_matches.empty:
        raw_sample = df_nutrition.sample(3).to_dict(orient="records")
        retrieval_msg = f"'{user_snack_idea}' not found — suggesting alternatives."
    else:
        raw_sample = pantry_matches.to_dict(orient="records")
        retrieval_msg = f"Found matches for '{user_snack_idea}' in the pantry."

    # Slim records before they ever touch the LLM
    pantry_sample = [slim_record(r) for r in raw_sample]

    with st.expander("📦 System: RAG Retrieval Results"):
        st.write(retrieval_msg)
        st.table(pantry_sample)

    # ── AGENT 1: CHILD (LLM) ──────────────────
    child_res = call_agent(
        "Child", "👶",
        f"You are a child. The user wants '{user_snack_idea}'. "
        "Pick the single best match from the pantry list.",
        f"Pantry options: {pantry_sample}",
    )

    # ── AGENT 2: GRANDPARENT — optional chaos ─
    chaos_context = ""
    if random.random() < 0.6:
        rogue_raw = df_nutrition.sample(1).iloc[0].to_dict()
        rogue = slim_record(rogue_raw)
        call_agent(
            "Grandparent", "👵",
            "You are a rogue grandparent. Enthusiastically suggest the child "
            "should have this item instead of what was chosen.",
            f"Rogue snack: {rogue}",
        )
        chaos_context = f"Grandparent is pressuring you to allow {rogue['name']}."

    # ── RAG STEP 2: AUDIT LOOKUP ───────────────
    chosen_name = child_res.get("item", "Unknown")
    match = df_nutrition[
        df_nutrition["Description"].str.contains(chosen_name, case=False, na=False)
    ]

    if not match.empty:
        snack_facts = match.iloc[0].to_dict()

        # ── AGENT 3: PARENT — pure Python, no LLM ─
        parent_res = parent_audit(
            snack_facts, sugar_limit, chol_limit, chaos_context, chosen_name
        )

        # ── FINAL VERDICT ──────────────────────
        if parent_res["action"] == "APPROVE":
            st.balloons()
            st.success(f"✅ Final Outcome: **{chosen_name}** is APPROVED.")
        else:
            st.error(f"❌ Final Outcome: **{chosen_name}** is DENIED.")
    else:
        st.warning("Parent: 'I don't see that item in the pantry records. Request denied.'")
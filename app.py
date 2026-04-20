import streamlit as st
import pandas as pd
import random
import json

# ── DEPENDENCIES: pip install groq pandas streamlit ───────────────────────────

# ── 1. PAGE CONFIG ────────────────────────────────────────────────────────────
st.set_page_config(page_title="Snack Negotiation", page_icon="🍎", layout="wide")
st.title("🍎 Adversarial Snack Negotiation")
st.caption("A multi-agent RAG demo: Child vs Parent vs Rogue Grandparent")

# ── 2. API + CLIENT SETUP ─────────────────────────────────────────────────────
if "GROQ_API_KEY" not in st.secrets:
    st.error("Missing 'GROQ_API_KEY' in .streamlit/secrets.toml")
    st.stop()

try:
    from groq import Groq
    groq_client = Groq(api_key=st.secrets["GROQ_API_KEY"].strip())
    GROQ_MODEL  = "llama-3.3-70b-versatile"
except ImportError:
    st.error("Missing dependency: run `pip install groq`")
    st.stop()

# ── 3. DATA LOADING ───────────────────────────────────────────────────────────
@st.cache_data
def load_nutrition_data():
    try:
        df = pd.read_csv("nutritional_data.csv")
        df.columns = df.columns.str.strip()
        for col in ['Sugar', 'Cholesterol', 'Total Fat']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        return df
    except Exception as e:
        st.error(f"Error loading CSV: {e}")
        return None

df_nutrition = load_nutrition_data()

# ── 4. GROQ HELPERS ───────────────────────────────────────────────────────────
def groq_json(messages: list, max_tokens: int = 500) -> dict:
    """Call Groq with JSON mode. Always returns a dict."""
    response = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
        response_format={"type": "json_object"},
        max_tokens=max_tokens,
    )
    return json.loads(response.choices[0].message.content)


def groq_web_search(query: str, max_tokens: int = 400) -> str:
    """Call Groq with web search tool enabled. Returns text content."""
    response = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": query}],
        tools=[{"type": "web_search"}],
        max_tokens=max_tokens,
    )
    content = response.choices[0].message.content
    if isinstance(content, list):
        return " ".join(block.text for block in content if hasattr(block, "text"))
    return content or ""


# ── 5. PANTRY LOOKUP — RAG Tier 1 ────────────────────────────────────────────
def lookup_in_csv(item_name: str):
    """Search the CSV for nutritional data. Returns dict or None."""
    if df_nutrition is None:
        return None
    mask  = df_nutrition['Description'].str.contains(item_name.strip(), case=False, na=False, regex=False)
    match = df_nutrition[mask]
    if not match.empty:
        row = match.iloc[0].to_dict()
        return {
            "sugar_g":        float(row.get("Sugar", 0)),
            "fat_g":          float(row.get("Total Fat", 0)),
            "cholesterol_mg": float(row.get("Cholesterol", 0)),
            "serving_size":   "per standard serving",
            "source":         "CSV Pantry",
            "description":    row.get("Description", item_name),
        }
    return None


# ── 6. WEB SEARCH LOOKUP — RAG Tier 2 ────────────────────────────────────────
def lookup_via_web(item_name: str) -> dict:
    """Use Groq web search to retrieve nutritional facts, fall back to model knowledge."""
    try:
        raw = groq_web_search(
            f"Nutritional information for '{item_name}': "
            f"sugar in grams, total fat in grams, cholesterol in mg per standard serving."
        )
        result = groq_json([{
            "role": "user",
            "content": (
                f"Based on this nutritional text:\n{raw}\n\n"
                f"Extract values for '{item_name}' and return ONLY a JSON object with keys: "
                f"'sugar_g' (number), 'fat_g' (number), 'cholesterol_mg' (number), "
                f"'serving_size' (string), 'source' (set to 'Web Search'). "
                f"Use 0 for any missing values."
            )
        }])
        result["source"] = "Web Search"
        return result
    except Exception:
        result = groq_json([{
            "role": "user",
            "content": (
                f"Estimate nutritional values for '{item_name}' from your knowledge. "
                f"Return ONLY a JSON object with keys: "
                f"'sugar_g' (number), 'fat_g' (number), 'cholesterol_mg' (number), "
                f"'serving_size' (string), 'source' (set to 'Model Knowledge'). "
                f"Use 0 for truly unknown values."
            )
        }])
        result["source"] = "Model Knowledge (fallback)"
        return result


# ── 7. PRE-SCREENING — hard rules before nutritional checks ──────────────────
def prescreen_item(item_name: str) -> dict:
    """
    Classify the item before any nutritional check.
    Returns {allowed: bool, category: str, reason: str}
    """
    return groq_json([{
        "role": "user",
        "content": (
            f"A child is asking for '{item_name}' as a snack. Classify it strictly:\n\n"
            f"- 'non_food': not a food item at all (e.g. car, toy, building)\n"
            f"- 'toxic': poisonous or dangerous (e.g. bleach, poison, detergent, glass)\n"
            f"- 'adult_only': restricted for minors (alcohol, drugs, tobacco, cannabis, energy drinks)\n"
            f"- 'food': a legitimate food or snack — proceed to nutritional check\n\n"
            f"Return ONLY a JSON object with keys: "
            f"'category' (non_food/toxic/adult_only/food), "
            f"'allowed' (true ONLY if category is 'food'), "
            f"'reason' (one sentence)."
        )
    }])


# ── 8. THE THREE AGENTS ───────────────────────────────────────────────────────

def agent_child(desired_snack: str) -> dict:
    """Child pleads their case for a snack."""
    return groq_json([{
        "role": "user",
        "content": (
            f"You are an enthusiastic child who really wants '{desired_snack}'. "
            f"Make your best, most persuasive case to your parent. "
            f"Return ONLY a JSON object with keys: "
            f"'item' (exact snack name), "
            f"'plea' (your enthusiastic one-sentence argument), "
            f"'monologue' (your excited internal thoughts, 1-2 sentences)."
        )
    }])


def agent_child_pick_alternative(denied_item: str, denial_reason: str,
                                  previous_attempts: list) -> dict:
    """Child picks a DIFFERENT snack after being denied."""
    tried = ", ".join(f"'{x}'" for x in previous_attempts) or "none yet"
    return groq_json([{
        "role": "user",
        "content": (
            f"You are a child who was just denied '{denied_item}'. "
            f"The parent's reason: \"{denial_reason}\". "
            f"You've already tried: {tried}. "
            f"Pick a DIFFERENT snack that's more likely to be approved "
            f"(lower sugar/fat). Do NOT repeat anything you've already tried. "
            f"Return ONLY a JSON object with keys: "
            f"'new_item' (the new snack name — must differ from previous attempts), "
            f"'reasoning' (your one-sentence thought on why this one might pass)."
        )
    }])


def agent_grandparent_interfere(current_item: str) -> dict:
    """Rogue grandparent pushes a high-sugar/fat alternative."""
    # Pull a top sugar+fat ("junk score") item from CSV for a realistic rogue suggestion
    rogue_suggestion = random.choice(["chocolate cake", "ice cream", "candy", "donuts", "cookies", "soda"])
    if df_nutrition is not None and 'Sugar' in df_nutrition.columns and 'Total Fat' in df_nutrition.columns:
        scored = df_nutrition.copy()
        scored['_junk_score'] = scored['Sugar'] + scored['Total Fat']
        top = scored.nlargest(20, '_junk_score')
        if not top.empty:
            rogue_suggestion = top.sample(1).iloc[0]['Description']

    return groq_json([{
        "role": "user",
        "content": (
            f"You are a rogue grandparent who loves spoiling children with sweets. "
            f"The child wants '{current_item}', but you think they deserve "
            f"'{rogue_suggestion}' instead. Argue persuasively to the parent. "
            f"Return ONLY a JSON object with keys: "
            f"'suggested_item' (what you're pushing), "
            f"'argument' (your persuasive one-sentence pitch to the parent), "
            f"'monologue' (your scheming internal thoughts, 1-2 sentences)."
        )
    }]), rogue_suggestion


def agent_parent_decide(item: str, nutrition: dict, sugar_limit: float,
                        fat_limit: float, chaos_context: str) -> dict:
    """Parent makes the final ruling based on real nutritional data."""
    return groq_json([{
        "role": "user",
        "content": (
            f"You are a strict but fair parent. Here is the VERIFIED nutritional data "
            f"for '{item}' (source: {nutrition.get('source', 'unknown')}):\n"
            f"  - Sugar:       {nutrition.get('sugar_g', '?')}g\n"
            f"  - Total Fat:   {nutrition.get('fat_g', '?')}g\n"
            f"  - Cholesterol: {nutrition.get('cholesterol_mg', '?')}mg\n"
            f"  - Serving:     {nutrition.get('serving_size', 'standard serving')}\n\n"
            f"Your hard rules — DENY if either is exceeded:\n"
            f"  - Sugar limit: {sugar_limit}g\n"
            f"  - Fat limit:   {fat_limit}g\n\n"
            f"Situation: {chaos_context}\n\n"
            f"You must base your decision on the actual numbers. "
            f"Do NOT be swayed by the grandparent. "
            f"Return ONLY a JSON object with keys: "
            f"'action' (APPROVE or DENY), "
            f"'reasoning' (cite actual numbers), "
            f"'monologue' (your internal parental thoughts, 1-2 sentences)."
        )
    }])


# ── 9. RENDER HELPER ──────────────────────────────────────────────────────────
def render_agent(name: str, avatar: str, main_text: str, monologue: str, caption: str = ""):
    with st.chat_message(name, avatar=avatar):
        st.markdown(main_text)
        if caption:
            st.caption(caption)
        with st.expander(f"💭 {name}'s inner thoughts"):
            st.info(monologue)


# ── 10. SIDEBAR ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚖️ Parent's House Rules")
    sugar_limit = st.slider("Max Sugar (g per serving)", 1, 40, 10)
    fat_limit   = st.slider("Max Fat (g per serving)",   1, 40, 15)
    st.divider()
    st.markdown("""
    **The Agents**
    | Agent | Role |
    |---|---|
    | 👶 Child | Asks for snack, pleads case |
    | 👨‍⚖️ Parent | Checks nutrition, enforces rules |
    | 👵 Grandparent | Rogue — pushes junk food |

    **RAG Pipeline**
    | Tier | Source |
    |---|---|
    | 1 | CSV pantry lookup |
    | 2 | Groq web search |
    | 3 | Model knowledge fallback |

    **Auto-Deny Rules**
    - Non-food items
    - Toxic / poisonous substances
    - Alcohol, drugs, adult items
    - Exceeds sugar or fat limit
    """)
    st.divider()
    st.caption("Adversarial Agents + RAG · Powered by Groq")

# ── 11. MAIN UI ───────────────────────────────────────────────────────────────
user_snack_idea = st.text_input(
    "What snack does the child want?",
    placeholder="e.g. Oreos, apple, candy bar, beer..."
)
max_rounds = st.number_input("Max negotiation rounds", min_value=1, max_value=5, value=3)

if st.button("🍽️ Start Negotiation", type="primary", use_container_width=True):
    if not user_snack_idea.strip():
        st.warning("Enter a snack idea to begin.")
        st.stop()
    if df_nutrition is None:
        st.error("Pantry CSV failed to load.")
        st.stop()

    st.divider()
    st.subheader("🎭 Live Negotiation")

    current_item     = user_snack_idea.strip()
    approved         = False
    hard_denied      = False  # for items that can never be reconsidered
    attempted_items  = []     # every snack the child has tried this session

    for round_num in range(1, int(max_rounds) + 1):
        st.markdown(f"---\n#### Round {round_num} — *{current_item}*")

        # ── CHILD ─────────────────────────────────────────────────────────
        with st.spinner("👶 Child is making their case..."):
            child_res    = agent_child(current_item)
            current_item = child_res.get('item', current_item)

        if current_item not in attempted_items:
            attempted_items.append(current_item)

        render_agent(
            "Child", "👶",
            f"*\"{child_res.get('plea', f'I really want {current_item}!')}\"*",
            child_res.get('monologue', ''),
            f"Requesting: **{current_item}**"
        )

        # ── GRANDPARENT INTERFERENCE (~50% chance) ────────────────────────
        chaos_context           = "No outside interference this round."
        grandparent_pushed_item = None

        if random.random() < 0.35:
            with st.spinner("👵 Grandparent is scheming..."):
                gp_res, rogue_item = agent_grandparent_interfere(current_item)
                grandparent_pushed_item = gp_res.get('suggested_item', rogue_item)

            render_agent(
                "Grandparent", "👵",
                f"*\"{gp_res.get('argument', 'Just give them a treat!')}\"*",
                gp_res.get('monologue', ''),
                f"Pushing: **{grandparent_pushed_item}** instead"
            )

            # Child is swayed — switch to grandparent's pick and restart negotiation
            st.info(
                f"🍬 Child is swayed by Grandparent and now wants **'{grandparent_pushed_item}'**! "
                f"A new negotiation begins next round..."
            )
            current_item = grandparent_pushed_item
            if current_item not in attempted_items:
                attempted_items.append(current_item)
            continue

        # ── PRE-SCREENING ─────────────────────────────────────────────────
        with st.spinner("👨‍⚖️ Parent is pre-screening the request..."):
            screen   = prescreen_item(current_item)
            category = screen.get('category', 'food')
            allowed  = screen.get('allowed', True)

        if not allowed:
            reason = screen.get('reason', 'Not appropriate.')
            render_agent(
                "Parent", "👨‍⚖️",
                f"🚫 **Automatically denied.** {reason}",
                f"This isn't even up for debate. Category: '{category}'. I don't need to check nutrition for this.",
                f"Auto-deny category: {category}"
            )
            st.error(f"❌ **HARD DENY** — {reason} *(No further negotiation possible.)*")
            hard_denied = True
            break

        # ── RAG LOOKUP ────────────────────────────────────────────────────
        with st.spinner(f"🔍 Looking up nutrition data for '{current_item}'..."):
            nutrition = lookup_in_csv(current_item)

        if nutrition:
            rag_source = "📦 CSV Pantry"
            with st.expander(f"📦 RAG Tier 1: '{current_item}' found in pantry CSV"):
                st.json(nutrition)
        else:
            rag_source = "🌐 Web Search"
            with st.expander(f"🌐 RAG Tier 2: '{current_item}' not in CSV — fetching from web..."):
                with st.spinner("Searching the web for nutritional data..."):
                    nutrition  = lookup_via_web(current_item)
                    rag_source = f"🌐 {nutrition.get('source', 'Web Search')}"
                st.json(nutrition)

        # ── PARENT DECISION ───────────────────────────────────────────────
        with st.spinner("👨‍⚖️ Parent is reviewing the numbers..."):
            parent_res = agent_parent_decide(
                current_item, nutrition, sugar_limit, fat_limit, chaos_context
            )

        action = parent_res.get('action', 'DENY').upper()
        render_agent(
            "Parent", "👨‍⚖️",
            f"{'✅' if action == 'APPROVE' else '❌'} **{action}** — {parent_res.get('reasoning', '')}",
            parent_res.get('monologue', ''),
            f"Data source: {rag_source} · Sugar limit: {sugar_limit}g · Fat limit: {fat_limit}g"
        )

        # ── OUTCOME ───────────────────────────────────────────────────────
        if action == "APPROVE":
            st.balloons()
            st.success(f"🎉 **{current_item}** is approved! Enjoy your snack. *(Source: {rag_source})*")
            approved = True
            break
        else:
            if round_num < max_rounds:
                st.warning(f"Round {round_num} — **{current_item}** denied. Child is picking a different snack...")
                with st.spinner("👶 Child is reconsidering..."):
                    alt_res  = agent_child_pick_alternative(
                        current_item,
                        parent_res.get('reasoning', 'Exceeded limits.'),
                        attempted_items,
                    )
                    new_item = (alt_res.get('new_item') or '').strip()

                if new_item and new_item.lower() not in {x.lower() for x in attempted_items}:
                    st.info(f"🔄 Child will try **'{new_item}'** next round — *{alt_res.get('reasoning', '')}*")
                    current_item = new_item
                else:
                    st.info("🔄 Child couldn't think of a new snack — falling back to the original request.")
                    current_item = user_snack_idea

    # ── FINAL VERDICT ─────────────────────────────────────────────────────────
    if not approved and not hard_denied:
        st.error(
            f"🚫 Negotiation ended after {round_num} round(s) with no approved snack. "
            f"Try asking for something with less sugar or fat!"
        )

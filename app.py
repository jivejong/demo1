import streamlit as st
import pandas as pd
import random
import json
import time

# --- 1. DATA LOADING (The 'Knowledge Base') ---
@st.cache_data
def load_nutrition_data():
    try:
        df = pd.read_csv("nutritional_data.csv")
        df.columns = df.columns.str.strip()
        for col in ['Sugar', 'Cholesterol']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        return df
    except Exception as e:
        st.error(f"Error loading CSV: {e}")
        return None

df_nutrition = load_nutrition_data()

# --- 2. API CONFIGURATION ---
st.set_page_config(page_title="Snack-Chain: Final RAG", page_icon="💬")
st.title("💬 Snack-Chain: Final RAG Negotiation")

if "GROQ_API_KEY" not in st.secrets:
    st.error("Missing 'GROQ_API_KEY' in .streamlit/secrets.toml")
    st.stop()

try:
    from groq import Groq
    groq_client = Groq(api_key=st.secrets["GROQ_API_KEY"].strip())
    GROQ_MODEL = "llama-3.3-70b-versatile"
except ImportError:
    st.error("Missing dependency: run `pip install groq`")
    st.stop()
except Exception as e:
    st.error(f"Configuration Error: {e}")
    st.stop()

# --- 3. THE AGENT ENGINE ---
def call_agent(role_name, avatar, instruction, context):
    """Handles the API call, UI rendering, and data extraction."""
    with st.chat_message(role_name, avatar=avatar):
        placeholder = st.empty()
        placeholder.markdown(f"*{role_name} is evaluating...*")

        prompt = f"""
        INSTRUCTION: {instruction}
        CONTEXT: {context}
        OUTPUT: Return ONLY a valid JSON object with keys:
        'action' (APPROVE/REJECT), 'item', 'reasoning', 'monologue'.
        """
        try:
            response = groq_client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                max_tokens=400,
            )
            data = json.loads(response.choices[0].message.content)

            placeholder.markdown(f"**{data.get('reasoning')}**")
            with st.expander(f"View {role_name}'s Thought Process"):
                st.info(data.get('monologue', 'No internal thoughts recorded.'))
                if data.get('item'):
                    st.caption(f"Targeted Item: {data['item']}")
            return data
        except Exception as e:
            placeholder.error(f"Error parsing agent response: {str(e)}")
            return {"action": "REJECT", "item": "None", "reasoning": "Communication error."}

# --- 4. THE UI & RAG ORCHESTRATION ---
with st.sidebar:
    st.header("⚖️ Compliance Rules")
    sugar_limit = st.slider("Max Sugar (g)", 5, 30, 15)
    chol_limit  = st.slider("Max Cholesterol (mg)", 20, 100, 50)
    st.divider()
    st.caption("Agentic RAG Pipeline v3.0 · Powered by Groq")

user_snack_idea = st.text_input("Child: 'I really want to eat...'", "Chocolate")

if st.button("Start Live Negotiation"):
    if df_nutrition is None:
        st.error("Pantry data not loaded.")
    else:
        st.subheader("Live Negotiation Thread")

        # --- RAG STEP 1: SEMANTIC RETRIEVAL ---
        search_mask    = df_nutrition['Description'].str.contains(user_snack_idea, case=False, na=False, regex=False)
        pantry_matches = df_nutrition[search_mask].head(3)

        if pantry_matches.empty:
            pantry_sample  = df_nutrition.sample(3).to_dict(orient='records')
            retrieval_msg  = f"'{user_snack_idea}' not found. Suggesting alternatives from CSV."
        else:
            pantry_sample  = pantry_matches.to_dict(orient='records')
            retrieval_msg  = f"Found matches for '{user_snack_idea}' in the pantry."

        with st.expander("📦 System: RAG Retrieval Results"):
            st.write(retrieval_msg)
            st.table(pantry_sample)

        # --- AGENT 1: THE CHILD ---
        child_res = call_agent(
            "Child", "👶",
            f"You are a child. User wants {user_snack_idea}. Pick the best match from the pantry list.",
            f"Available Pantry: {pantry_sample}"
        )

        # --- AGENT 2: THE GRANDPARENT (Optional Chaos) ---
        chaos_context = ""
        if random.random() < 0.6:
            rogue_item = df_nutrition.sample(1).iloc[0].to_dict()
            call_agent(
                "Grandparent", "👵",
                "Be a rogue grandparent. Suggest the child has this item instead.",
                f"Rogue Item: {rogue_item['Description']}"
            )
            chaos_context = f"Grandparent is pressuring you to allow {rogue_item['Description']}."

        # --- RAG STEP 2: AUDIT RETRIEVAL ---
        chosen_name = child_res.get('item', 'Unknown')

        if not chosen_name or not isinstance(chosen_name, str) or chosen_name.strip() in ('', 'None', 'Unknown'):
            st.warning("The child couldn't identify a specific item. Try a different snack!")
        else:
            match = df_nutrition[df_nutrition['Description'].str.contains(chosen_name.strip(), case=False, na=False, regex=False)]

            if not match.empty:
                snack_facts  = match.iloc[0].to_dict()
                audit_context = f"""
                REAL DATA FOR {chosen_name}:
                - Sugar: {snack_facts.get('Sugar')}g
                - Cholesterol: {snack_facts.get('Cholesterol')}mg

                SITUATIONAL CONTEXT:
                {chaos_context}
                """

                parent_res = call_agent(
                    "Parent", "👨‍⚖️",
                    f"Auditor. REJECT if Sugar > {sugar_limit} OR Cholesterol > {chol_limit}.",
                    audit_context
                )

                if parent_res.get('action') == "APPROVE":
                    st.balloons()
                    st.success(f"Final Outcome: {chosen_name} is APPROVED.")
                else:
                    st.error(f"Final Outcome: {chosen_name} is DENIED.")
            else:
                st.warning("Parent: 'I don't see that item in the pantry records. Request denied.'")

import streamlit as st
import google.generativeai as genai
import pandas as pd
import random
import json
import time

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(page_title="Snack-Chain: Agentic Negotiation", page_icon="🍪", layout="centered")

# --- 2. DATA & API SETUP ---
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("nutritional_data.csv")
        df['Sugar'] = pd.to_numeric(df['Sugar'], errors='coerce').fillna(0)
        df['Cholesterol'] = pd.to_numeric(df['Cholesterol'], errors='coerce').fillna(0)
        return df
    except FileNotFoundError:
        return None

df_nutrition = load_data()

if df_nutrition is None:
    st.error("🚨 **File Not Found:** Please ensure 'nutritional_data.csv' is in the root folder.")
    st.stop()

if "GEMINI_API_KEY" not in st.secrets:
    st.error("🔑 **Missing API Key:** Check your Secrets dashboard or .streamlit/secrets.toml.")
    st.stop()

genai.configure(api_key=st.secrets["GEMINI_API_KEY"].strip())
model = genai.GenerativeModel('gemini-1.5-flash') 

# --- 3. LOGIC ENGINE ---

def lookup_snack(search_term):
    mask = df_nutrition['Description'].str.contains(search_term, case=False, na=False)
    result = df_nutrition[mask]
    return result.iloc[0].to_dict() if not result.empty else None

def find_tastiest_alternatives(category, max_sugar):
    alts = df_nutrition[
        (df_nutrition['Category'] == category) & 
        (df_nutrition['Sugar'] <= max_sugar)
    ]
    if not alts.empty:
        tastiest = alts.sort_values(by='Sugar', ascending=False).head(3)
        return tastiest.to_dict(orient='records')
    return []

def call_agent(role, persona, context):
    time.sleep(2) # Rate limit protection
    prompt = f"""
    ROLE: {role}
    PERSONA: {persona}
    DATA CONTEXT: {context}
    
    OUTPUT: Return ONLY a valid JSON object with these exact keys:
    "action" (APPROVE or REJECT), "item" (The name of the food), "reasoning" (A sentence), "monologue" (Internal thoughts)
    """
    try:
        response = model.generate_content(prompt)
        # Better JSON cleaning to avoid parser errors
        raw_text = response.text.strip()
        if "```json" in raw_text:
            raw_text = raw_text.split("```json")[1].split("```")[0]
        elif "```" in raw_text:
            raw_text = raw_text.split("```")[1].split("```")[0]
        
        return json.loads(raw_text)
    except Exception as e:
        # Fallback dictionary so the app doesn't crash on KeyError
        return {
            "action": "REJECT", 
            "item": "Unknown Item", 
            "reasoning": "I'm having trouble thinking clearly.", 
            "monologue": str(e)
        }

# --- 4. STREAMLIT UI ---
st.title("👶 Snack-Chain")
st.caption("Agentic Negotiation Engine v2.3 (KeyError Proofed)")

with st.sidebar:
    st.header("⚙️ House Rules")
    sugar_limit = st.slider("Strict Sugar Limit (g)", 5, 30, 15)
    st.divider()
    st.info("The logic now acknowledges the Grandparent's pressure during the final decision.")

user_snack_request = st.text_input("What do you want to eat?", placeholder="e.g. Cookie, Cake, Apple...")

if st.button("Submit Request"):
    if not user_snack_request:
        st.warning("Please type a snack name!")
    else:
        snack = lookup_snack(user_snack_request)
        
        if not snack:
            st.error(f"'{user_snack_request}' not found in pantry.")
        else:
            with st.status("Negotiation in progress...", expanded=True) as status:
                # --- PHASE 1: INITIAL PARENT AUDIT ---
                parent_persona = f"Strict Health Auditor. Limit: {sugar_limit}g sugar."
                parent_context = f"Item: {snack['Description']}, Sugar: {snack['Sugar']}g"
                audit = call_agent("Parent Auditor", parent_persona, parent_context)

                st.chat_message("parent", avatar="👨‍⚖️").write(f"**Verdict:** {audit.get('action')}. {audit.get('reasoning')}")
                
                # --- PHASE 2: IF REJECTED, START NEGOTIATION ---
                if audit.get('action') == "REJECT":
                    
                    #
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
        # Ensure numbers are actual floats/ints
        df['Sugar'] = pd.to_numeric(df['Sugar'], errors='coerce').fillna(0)
        df['Cholesterol'] = pd.to_numeric(df['Cholesterol'], errors='coerce').fillna(0)
        return df
    except FileNotFoundError:
        return None

df_nutrition = load_data()

if df_nutrition is None:
    st.error("🚨 **File Not Found:** Please ensure 'nutritional_data.csv' is in the root folder.")
    st.stop()

# Get API Key from Streamlit Secrets (Cloud or Local)
if "GEMINI_API_KEY" not in st.secrets:
    st.error("🔑 **Missing API Key:** Check your Secrets dashboard or .streamlit/secrets.toml.")
    st.stop()

genai.configure(api_key=st.secrets["GEMINI_API_KEY"].strip())
# Using 1.5 Flash for the Child to save 2.0/2.5 quota if needed
model = genai.GenerativeModel('gemini-1.5-flash') 

# --- 3. LOGIC ENGINE ---

def lookup_snack(search_term):
    """Finds the first match in the CSV."""
    mask = df_nutrition['Description'].str.contains(search_term, case=False, na=False)
    result = df_nutrition[mask]
    return result.iloc[0].to_dict() if not result.empty else None

def find_tastiest_alternatives(category, max_sugar):
    """Finds items in same category closest to the limit (Tastiest)."""
    alts = df_nutrition[
        (df_nutrition['Category'] == category) & 
        (df_nutrition['Sugar'] <= max_sugar)
    ]
    if not alts.empty:
        # Sort by Sugar Descending so the Child picks the highest sugar allowed
        tastiest = alts.sort_values(by='Sugar', ascending=False).head(3)
        return tastiest.to_dict(orient='records')
    return []

def call_agent(role, persona, context):
    """LLM wrapper with a 2-second sleep to avoid Rate Limits (429)."""
    time.sleep(2) # The 'Cool Down' strategy
    
    prompt = f"""
    ROLE: {role}
    PERSONA: {persona}
    DATA CONTEXT: {context}
    
    OUTPUT: Return ONLY a valid JSON object with these keys:
    "action" (APPROVE or REJECT), "item", "reasoning", "monologue"
    """
    try:
        response = model.generate_content(prompt)
        raw_text = response.text.replace("```json", "").replace("```", "").strip()
        return json.loads(raw_text)
    except Exception as e:
        return {"action": "REJECT", "reasoning": f"Quota/API Issue: {str(e)}", "monologue": "Static."}

# --- 4. STREAMLIT UI ---
st.title("👶 Snack-Chain")
st.caption("Agentic Negotiation Engine v2.1 (Data-Grounded)")

with st.sidebar:
    st.header("⚙️ House Rules")
    sugar_limit = st.slider("Strict Sugar Limit (g)", 5, 30, 15)
    st.divider()
    st.info("The Parent uses the CSV. The Child uses persuasion. The Grandparent uses chaos.")

user_snack_request = st.text_input("What do you want to eat?", placeholder="e.g. Cookie, Cake, Apple...")

if st.button("Submit Request"):
    if not user_snack_request:
        st.warning("Please type a snack name!")
    else:
        snack = lookup_snack(user_snack_request)
        
        if not snack:
            st.error(f"'{user_snack_request}' not found in pantry CSV.")
        else:
            # --- PHASE 1: PARENT AUDIT ---
            with st.status("Parent is auditing...", expanded=True) as status:
                parent_persona = f"Strict Health Auditor. Limit is {sugar_limit}g sugar. Reject if over."
                # Minimal context to save tokens
                parent_context = f"Item: {snack['Description']}, Sugar: {snack['Sugar']}g"
                audit = call_agent("Parent Auditor", parent_persona, parent_context)

                st.chat_message("parent", avatar="👨‍⚖️").write(f"**Verdict:** {audit['action']}. {audit['reasoning']}")
                
                # --- PHASE 2: BRANCHING ---
                if audit['action'] == "REJECT":
                    
                    # GRANDPARENT CHANCE (40%)
                    if random.random() < 0.4:
                        gp_response = call_agent("Grandparent", "Chaotic advocate for the child.", f"Parent rejected {snack['Description']}")
                        st.chat_message("grandparent", avatar="👵").write(f"*{gp_response['reasoning']}*")

                    # CHILD PIVOT (Tastiest Alternative)
                    alts = find_tastiest_alternatives(snack['Category'], sugar_limit)
                    
                    if alts:
                        child_persona = "Strategic Child. Pick the highest sugar item from the alternatives. Frame it as being healthy."
                        child_pivot = call_agent("Child Agent", child_persona, f"Pick from: {alts}")
                        
                        st.chat_message("child", avatar="👶").write(f"Fine... can I have **{child_pivot['item']}**? It's under the {sugar_limit}g limit!")
                        
                        # FINAL AUDIT
                        final_snack = lookup_snack(child_pivot['item'])
                        final_audit = call_agent("Parent Auditor", parent_persona, f"Request: {final_snack}")
                        st.chat_message("parent", avatar="👨‍⚖️").write(f"**Final Decision:** {final_audit['action']}. {final_audit['reasoning']}")
                    else:
                        st.error("No healthier options found in this category.")
                else:
                    st.balloons()
                    st.success("Approved! No negotiation needed.")
                
                status.update(label="Negotiation Complete", state="complete")
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
# Using 2.5 Flash as the standard for 2026
model = genai.GenerativeModel('gemini-2.5-flash') 

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
        tastiest = alts.sort_values(by='Sugar', ascending=False).head(3)
        return tastiest.to_dict(orient='records')
    return []

def call_agent(role, persona, context, retries=3):
    """LLM wrapper with a 4-second sleep and retry logic for 429 errors."""
    time.sleep(4) # Increased delay to protect Free Tier RPM
    
    prompt = f"""
    ROLE: {role}
    PERSONA: {persona}
    DATA CONTEXT: {context}
    
    OUTPUT: Return ONLY a valid JSON object. Do not include markdown formatting or prose.
    """
    for i in range(retries):
        try:
            response = model.generate_content(prompt)
            raw_text = response.text.strip()
            # Clean up the JSON string
            if "```json" in raw_text:
                raw_text = raw_text.split("```json")[1].split("```")[0]
            elif "```" in raw_text:
                raw_text = raw_text.split("```")[1].split("```")[0]
            
            return json.loads(raw_text)
            
        except Exception as e:
            if "429" in str(e):
                wait_time = (i + 1) * 10 
                st.warning(f"⚠️ {role} is waiting on the kitchen line... ({wait_time}s)")
                time.sleep(wait_time)
                continue
            else:
                return {"action": "REJECT", "reasoning": f"Error: {str(e)}", "item": "Unknown"}
                
    return {"action": "REJECT", "reasoning": "The kitchen is too busy right now.", "item": "Timeout"}

# --- 4. STREAMLIT UI ---
st.title("👶 Snack-Chain")
st.caption("Agentic Negotiation Engine v2.4 (Batch Optimized)")

with st.sidebar:
    st.header("⚙️ House Rules")
    sugar_limit = st.slider("Strict Sugar Limit (g)", 5, 30, 15)
    st.divider()
    st.info("Consolidated agents reduce API calls and prevent Rate Limit errors.")

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
                
                # --- PHASE 2: BATCHED FAMILY NEGOTIATION ---
                if audit.get('action') == "REJECT":
                    alts = find_tastiest_alternatives(snack['Category'], sugar_limit)
                    
                    if alts:
                        # WE COMBINE THE CHILD AND GRANDPARENT INTO ONE REQUEST
                        family_context = f"Rejected: {snack['Description']}. Options: {alts}."
                        family_persona = f"""
                        You represent a Grandparent and a Child.
                        1. Grandparent: Give a short 'chaotic' plea for a treat.
                        2. Child: Pick the highest sugar item from Options.
                        Return JSON: {{"gp_plea": "...", "child_item": "...", "child_reasoning": "..."}}
                        """
                        
                        neg_team = call_agent("Family Duo", family_persona, family_context)
                        
                        # UI Display
                        gp_plea = neg_team.get('gp_plea', "Let them have a treat!")
                        child_item = neg_team.get('child_item', 'A healthy choice')
                        
                        st.chat_message("grandparent", avatar="👵").write(f"*{gp_plea}*")
                        st.chat_message("child", avatar="👶").write(f"Can I at least have **{child_item}**? {neg_team.get('child_reasoning')}")
                        
                        # --- PHASE 3: FINAL AUDIT ---
                        final_snack = lookup_snack(child_item) or alts[0]
                        final_context = f"New Item: {final_snack}. GP says: {gp_plea}. Limit: {sugar_limit}g."
                        final_parent_persona = "Final Judge. Decide if you yield to family pressure or stay firm. Mention the Grandparent."
                        
                        final_audit = call_agent("Parent Auditor", final_parent_persona, final_context)
                        st.chat_message("parent", avatar="👨‍⚖️").write(f"**Final Decision:** {final_audit.get('action')}. {final_audit.get('reasoning')}")
                    else:
                        st.error("No healthier options available in this category.")
                else:
                    st.balloons()
                    st.success("Approved! No negotiation needed.")
                
                status.update(label="Negotiation Complete", state="complete")
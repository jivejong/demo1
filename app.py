import streamlit as st
import google.generativeai as genai
import pandas as pd
import random
import json
import time

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(page_title="Snack-Chain: RAG Negotiation", page_icon="🍪", layout="centered")

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
    st.error("🚨 **File Not Found:** Ensure 'nutritional_data.csv' is in the root folder.")
    st.stop()

# Get all unique categories for the AI to use as a "Map"
available_categories = df_nutrition['Category'].unique().tolist()

genai.configure(api_key=st.secrets["GEMINI_API_KEY"].strip())
model = genai.GenerativeModel('gemini-2.5-flash') 

# --- 3. THE RAG ENGINE ---

def lookup_snack(search_term):
    """Exact match lookup."""
    mask = df_nutrition['Description'].str.contains(search_term, case=False, na=False)
    result = df_nutrition[mask]
    return result.iloc[0].to_dict() if not result.empty else None

def find_tastiest_alternatives(category, max_sugar):
    """Retrieves real data from the CSV based on a category."""
    alts = df_nutrition[
        (df_nutrition['Category'] == category) & 
        (df_nutrition['Sugar'] <= max_sugar)
    ]
    if not alts.empty:
        tastiest = alts.sort_values(by='Sugar', ascending=False).head(3)
        return tastiest.to_dict(orient='records')
    return []

def call_agent(role, persona, context, retries=3):
    """Standard call with 4s delay for 429 protection."""
    time.sleep(4)
    prompt = f"ROLE: {role}\nPERSONA: {persona}\nDATA: {context}\nReturn ONLY JSON."
    for i in range(retries):
        try:
            response = model.generate_content(prompt)
            raw_text = response.text.strip()
            if "```json" in raw_text:
                raw_text = raw_text.split("```json")[1].split("```")[0]
            return json.loads(raw_text)
        except Exception as e:
            if "429" in str(e):
                time.sleep((i+1)*10)
                continue
            return {"action": "REJECT", "item": "Unknown", "reasoning": "Error"}
    return {"action": "REJECT", "item": "Timeout"}

# --- 4. STREAMLIT UI ---
st.title("👶 Snack-Chain RAG")

with st.sidebar:
    st.header("⚙️ House Rules")
    sugar_limit = st.slider("Strict Sugar Limit (g)", 5, 30, 15)

user_request = st.text_input("What do you want to eat?", placeholder="e.g. Ice Cream, Toast...")

if st.button("Submit Request"):
    with st.status("Analyzing Request...", expanded=True) as status:
        
        # --- RAG STEP 1: SEMANTIC MATCHING ---
        # If not in CSV, we ask the AI to map the request to a real Category
        snack = lookup_snack(user_request)
        
        if not snack:
            st.info(f"'{user_request}' isn't in our pantry. Identifying closest category...")
            mapping_prompt = f"Map the user request '{user_request}' to one of these categories: {available_categories}"
            mapping_res = call_agent("System Auditor", "Return JSON: {'category': '... '}", mapping_prompt)
            target_category = mapping_res.get('category', available_categories[0])
        else:
            target_category = snack['Category']

        # --- PHASE 1: PARENT AUDIT ---
        # If we have the snack, we use its real data. If not, we assume it's "Unknown/High"
        sugar_val = snack['Sugar'] if snack else 999 
        item_desc = snack['Description'] if snack else user_request
        
        parent_persona = f"Health Auditor. Limit: {sugar_limit}g."
        parent_context = f"Item: {item_desc}, Sugar: {sugar_val}g."
        audit = call_agent("Parent", parent_persona, parent_context)

        st.chat_message("parent", avatar="👨‍⚖️").write(f"**Verdict:** {audit.get('action')}. {audit.get('reasoning')}")

        # --- PHASE 2: THE COMPROMISE (The core of RAG) ---
        if audit.get('action') == "REJECT":
            # Retrieve REAL alternatives based on the mapped category
            alts = find_tastiest_alternatives(target_category, sugar_limit)
            
            if alts:
                family_context = f"Original rejected. Category: {target_category}. Options: {alts}."
                family_persona = "Represent Grandparent (plea) and Child (picks best option from list). JSON ONLY."
                neg_team = call_agent("Family", family_persona, family_context)
                
                st.chat_message("grandparent", avatar="👵").write(f"*{neg_team.get('gp_plea')}*")
                st.chat_message("child", avatar="👶").write(f"Can I have **{neg_team.get('child_item')}**? {neg_team.get('child_reasoning')}")
                
                # FINAL AUDIT
                final_snack = lookup_snack(neg_team.get('child_item')) or alts[0]
                final_audit = call_agent("Parent", "Final Judge", f"Item: {final_snack}")
                st.chat_message("parent", avatar="👨‍⚖️").write(f"**Final:** {final_audit.get('action')}")
            else:
                st.error("No alternatives in this category found.")
        else:
            st.success("Approved!")
        
        status.update(label="Complete", state="complete")
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
model = genai.GenerativeModel('gemini-2.5-flash') 

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

def call_agent(role, persona, context, retries=3):
    # Base delay to stay safe
    time.sleep(2) 
    
    prompt = f"ROLE: {role}\nPERSONA: {persona}\nDATA: {context}"
    
    for i in range(retries):
        try:
            response = model.generate_content(prompt)
            # Standard JSON cleaning
            raw_text = response.text.strip()
            if "```json" in raw_text:
                raw_text = raw_text.split("```json")[1].split("```")[0]
            return json.loads(raw_text)
            
        except Exception as e:
            if "429" in str(e):
                # If we hit a 429, wait longer and try again
                wait_time = (i + 1) * 5 
                st.warning(f"⚠️ {role} is thinking too hard... cooling down for {wait_time}s")
                time.sleep(wait_time)
                continue # Try the next loop
            else:
                return {"action": "REJECT", "item": "Error", "reasoning": f"Error: {str(e)}"}
                
    return {"action": "REJECT", "item": "Timeout", "reasoning": "The agents are exhausted. Try again in a minute."}

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
                    
                    # GRANDPARENT INTERVENTION
                    gp_plea = ""
                    if random.random() < 0.4:
                        gp_persona = "Chaotic Grandparent. Persuade the parent to let the child have a treat."
                        gp_response = call_agent("Grandparent", gp_persona, f"Parent rejected {snack['Description']}")
                        gp_plea = gp_response.get('reasoning', "Let them have it!")
                        st.chat_message("grandparent", avatar="👵").write(f"*{gp_plea}*")

                    # CHILD PIVOT
                    alts = find_tastiest_alternatives(snack['Category'], sugar_limit)
                    
                    if alts:
                        child_persona = f"Strategic Child. Request the best item from the list. If a Grandparent spoke up, use that as leverage."
                        child_pivot = call_agent("Child Agent", child_persona, f"Alternatives: {alts}. Grandparent's plea: {gp_plea}")
                        
                        # USES .get() TO PREVENT THE KEYERROR CRASH
                        item_name = child_pivot.get('item', 'Something else')
                        reasoning = child_pivot.get('reasoning', 'I promise to be good!')
                        
                        st.chat_message("child", avatar="👶").write(f"Can I at least have **{item_name}**? {reasoning}")
                        
                        # FINAL AUDIT
                        final_snack = lookup_snack(item_name)
                        if not final_snack: # Fallback if Child hallucinations a name not in CSV
                            final_snack = alts[0]
                            
                        final_context = f"New Item: {final_snack}. Grandparent said: {gp_plea}. Rules: {sugar_limit}g sugar."
                        final_parent_persona = "Final Judge. Acknowledge the Grandparent's pressure in your final verdict."
                        
                        final_audit = call_agent("Parent Auditor", final_parent_persona, final_context)
                        st.chat_message("parent", avatar="👨‍⚖️").write(f"**Final Decision:** {final_audit.get('action')}. {final_audit.get('reasoning')}")
                    else:
                        st.error("No healthier options available.")
                else:
                    st.balloons()
                    st.success("Approved immediately!")
                
                status.update(label="Negotiation Complete", state="complete")
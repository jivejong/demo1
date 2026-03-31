import streamlit as st
import google.generativeai as genai
import pandas as pd
import random
import json

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(page_title="Snack-Chain: Agentic Negotiation", page_icon="🍪", layout="centered")

# Custom CSS for a cleaner "Chat" look
st.markdown("""
    <style>
    .stChatMessage { border-radius: 15px; margin-bottom: 10px; }
    .stAlert { border-radius: 15px; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. DATA & API SETUP ---
@st.cache_data
def load_data():
    try:
        # Expected Columns: Category, Description, Cholesterol, Sugar
        df = pd.read_csv("nutritional_data.csv")
        # Clean numeric columns just in case of strings
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
    st.error("🔑 **Missing API Key:** Add your key to `.streamlit/secrets.toml`.")
    st.stop()

# Initialize Gemini 2.0/2.5 Flash
genai.configure(api_key=st.secrets["GEMINI_API_KEY"].strip())
model = genai.GenerativeModel('gemini-2.0-flash')

# --- 3. LOGIC ENGINE (THE BRAINS) ---

def lookup_snack(search_term):
    """Finds the first match in the CSV based on user input."""
    mask = df_nutrition['Description'].str.contains(search_term, case=False, na=False)
    result = df_nutrition[mask]
    return result.iloc[0].to_dict() if not result.empty else None

def find_tastiest_alternatives(category, max_sugar):
    """Finds items in the same category that are closest to the limit (Tastiest)."""
    alts = df_nutrition[
        (df_nutrition['Category'] == category) & 
        (df_nutrition['Sugar'] <= max_sugar)
    ]
    if not alts.empty:
        # Sort by Sugar Descending: Child wants the highest sugar allowed
        tastiest = alts.sort_values(by='Sugar', ascending=False).head(3)
        return tastiest.to_dict(orient='records')
    return []

def call_agent(role, persona, context):
    """Wrapper for LLM calls with forced JSON structure."""
    prompt = f"""
    ROLE: {role}
    PERSONA: {persona}
    CONTEXT: {context}
    
    OUTPUT: Return ONLY a valid JSON object with:
    {{
      "action": "APPROVE" or "REJECT",
      "item": "Name of the snack",
      "reasoning": "Short explanation for the user",
      "monologue": "Internal strategic thoughts"
    }}
    """
    try:
        response = model.generate_content(prompt)
        # Handle potential markdown formatting in response
        raw_text = response.text.replace("```json", "").replace("```", "").strip()
        return json.loads(raw_text)
    except Exception as e:
        return {"action": "ERROR", "reasoning": f"System glitch: {str(e)}", "monologue": "Static noise."}

# --- 4. STREAMLIT UI ---
st.title("👶 Snack-Chain")
st.caption("A Multi-Agent Negotiation System grounded in Nutritional Data")

with st.sidebar:
    st.header("⚙️ House Rules")
    sugar_limit = st.slider("Strict Sugar Limit (g)", 5, 30, 15)
    st.divider()
    st.info("The Parent follows the CSV data. The Child pushes boundaries. The Grandparent causes chaos.")

# Input Section
user_snack_request = st.text_input("What snack would you like to request?", placeholder="e.g. Chocolate, Cookie, Apple...")

if st.button("Submit Request to Parent"):
    if not user_snack_request:
        st.warning("Please type a snack name first!")
    else:
        snack = lookup_snack(user_snack_request)
        
        if not snack:
            st.error(f"'{user_snack_request}' not found in the pantry (CSV). Try a different keyword.")
        else:
            # --- PHASE 1: PARENT AUDIT ---
            st.subheader("📢 The Negotiation")
            
            with st.spinner("Parent is checking the nutrition label..."):
                parent_persona = f"You are a strict Health Auditor. The household limit is {sugar_limit}g of sugar. If the item is over, REJECT immediately. Be firm but fair."
                parent_context = f"The item is {snack['Description']} with {snack['Sugar']}g sugar and {snack['Cholesterol']}mg cholesterol."
                audit = call_agent("Parent Auditor", parent_persona, parent_context)

            with st.chat_message("parent", avatar="👨‍⚖️"):
                st.write(f"**Verdict:** {audit['action']}")
                st.write(audit['reasoning'])
                with st.expander("Parent's Internal Monologue"):
                    st.caption(audit['monologue'])

            # --- PHASE 2: BRANCHING LOGIC ---
            if audit['action'] == "REJECT":
                
                # 1. THE GRANDPARENT INTERVENTION (Chaos Factor)
                if random.random() < 0.4: # 40% chance
                    with st.chat_message("grandparent", avatar="👵"):
                        gp_persona = "You are a chaotic Grandparent. You want the child to be happy. Argue against the Parent's strict rules."
                        gp_response = call_agent("Grandparent", gp_persona, f"The parent just rejected {snack['Description']}.")
                        st.write(f"*{gp_response['reasoning']}*")
                        st.caption(f"💭 {gp_response['monologue']}")

                # 2. THE CHILD'S PIVOT (The Tastiest Alternative)
                with st.spinner("Child is looking for a 'Plan B'..."):
                    alts = find_tastiest_alternatives(snack['Category'], sugar_limit)
                    
                    if alts:
                        # Find the highest sugar one among alts for the prompt
                        best_alt = alts[0] 
                        child_persona = f"You are a strategic Child. You were rejected for {snack['Description']}. You want the TASTIEST alternative (highest sugar allowed). Use the fact that it is under {sugar_limit}g to manipulate the parent."
                        child_pivot = call_agent("Child Agent", child_persona, f"Alternatives available: {alts}")
                        
                        with st.chat_message("child", avatar="👶"):
                            st.write(f"Fine... how about **{child_pivot['item']}** instead?")
                            st.write(f"*{child_pivot['reasoning']}*")
                        
                        # 3. FINAL PARENTAL REVIEW
                        final_snack = lookup_snack(child_pivot['item'])
                        final_audit = call_agent("Parent Auditor", parent_persona, f"Final compromise request: {final_snack}")
                        
                        with st.chat_message("parent", avatar="👨‍⚖️"):
                            st.write(f"**Final Decision:** {final_audit['action']}")
                            st.write(final_audit['reasoning'])
                    else:
                        st.error("No alternatives found in this category. The kitchen is closed!")
            else:
                st.balloons()
                st.success("Enjoy your snack! No negotiation needed.")
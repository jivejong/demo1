# 🍎 Adversarial Snack Negotiation

A multi-agent agentic AI demo built with Streamlit. Three LLM-powered agents negotiate in real time over whether a child can have a snack — with a rogue grandparent agent that randomly interferes to push junk food.

**Demonstrates:** adversarial agents · RAG pipeline · multi-tier retrieval · pre-screening classifiers · agent handoffs

---

## What It Does

The user enters a snack name. From there, a full multi-agent negotiation runs:

1. **Child Agent** — pleads its case for the requested snack
2. **Grandparent Agent** *(rogue, 35% chance per round)* — randomly hijacks the negotiation and convinces the child to request a high-sugar alternative instead
3. **Parent Agent** — pre-screens the item, looks up nutritional data via the RAG pipeline, and approves or denies based on configurable sugar/fat limits
4. If denied, the **Child Agent** adapts and picks a new snack for the next round

Negotiation continues for a user-selected number of rounds until a snack is approved or all rounds are exhausted.

Each agent's inner monologue is available via a collapsible dropdown.

---

## Agent Architecture

| Agent | Role |
|---|---|
| 👶 Child | Requests snack, pleads case, adapts after denial |
| 👵 Grandparent | Rogue agent — randomly pushes high-sugar alternatives |
| 👨‍⚖️ Parent | Pre-screens item category, retrieves nutrition data, enforces limits |

---

## RAG Pipeline

Nutritional data is retrieved in three tiers:

| Tier | Source | Trigger |
|---|---|---|
| 1 | Vector DB — Chroma + MiniLM embeddings | Semantic search, checked first |
| 2 | Groq web search tool | If no confident vector match |
| 3 | Model knowledge fallback | If web search fails |

**Tier 1 (semantic retrieval).** On first run, every row of `nutritional_data.csv`
(~7,400 items) is embedded with the built-in ONNX `all-MiniLM-L6-v2` model and
stored in a persistent [Chroma](https://www.trychroma.com/) collection (cosine
space). Lookups embed the requested snack and return the nearest item — so
`"choc chip cookie"` matches `COOKIES,CHOCOLATE CHIP` without an exact string
match. A configurable similarity threshold (sidebar → **RAG settings**) decides
when a match is confident enough; weaker matches fall through to web search. The
index is cached to `./chroma_db` and reused across runs.

---

## Pre-Screening Rules

The Parent Agent classifies items before any nutritional check. Items are auto-denied (no further negotiation) if they fall into:

- `non_food` — not a food item (e.g. toy, car)
- `toxic` — poisonous or dangerous (e.g. bleach, glass)
- `adult_only` — alcohol, drugs, tobacco, energy drinks

---

## Tech Stack

| Component | Technology |
|---|---|
| LLM | Groq API · `qwen/qwen3.6-27b` |
| Vector store | Chroma (persistent) · ONNX `all-MiniLM-L6-v2` embeddings |
| UI | Streamlit |
| Data | Pandas · local CSV |
| RAG Tier 2 | Groq web search tool |

---

## Setup

### 1. Clone and install dependencies

```bash
pip install -r requirements.txt   # groq, pandas, streamlit, chromadb
```

### 2. Add your Groq API key

Create `.streamlit/secrets.toml`:

```toml
GROQ_API_KEY = "your-groq-api-key"
# Optional — override the default model without touching the code:
# GROQ_MODEL  = "qwen/qwen3.6-27b"
```

### 3. Add your nutritional data CSV

Place `nutritional_data.csv` in the project root. Expected columns:

```
Category, Description, Cholesterol, Sugar
```

> The dataset has no fat column, so `Cholesterol` is used as the Total-Fat proxy
> for the parent's fat rule.

### 4. Run the app

```bash
streamlit run app.py
```

---

## Configuration

Use the sidebar sliders to set the Parent Agent's house rules:

| Setting | Default | Range |
|---|---|---|
| Max Sugar (g per serving) | 10g | 1–40g |
| Max Fat (g per serving) | 15g | 1–40g |
| Max negotiation rounds | 3 | 1–5 |

---

## Project Structure

```
.
├── app.py                  # Main application
├── nutritional_data.csv    # Local pantry data (source for RAG Tier 1)
├── chroma_db/              # Persisted vector index (auto-built, git-ignored)
└── .streamlit/
    └── secrets.toml        # API keys
```

---

## Key Concepts Demonstrated

- **Adversarial agents** — the Grandparent agent actively works against the Parent agent's goal
- **Vector RAG** — semantic retrieval over an embedded pantry via a Chroma vector DB, with a tunable confidence threshold
- **Multi-tier RAG** — graceful fallback from vector DB → web search → model knowledge
- **Pre-screening classifier** — LLM-as-classifier to gate nutritional lookup
- **Agent memory within session** — child tracks previously attempted snacks to avoid repetition
- **Transparent reasoning** — every agent exposes its inner monologue via expander UI

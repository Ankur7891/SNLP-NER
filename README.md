# Biomedical Text Mining (NER) API using LangChain

An API using **Google Gemini** (via LangChain) to extract biomedical entities and relations (genes, proteins, drugs, diseases, mutations) from text.

---

#### Setup

- Clone the Repo & `cd` into it...
- Install dependencies: `pip install -r requirements.txt`
- Set API key and Gemini model in `.env` as:
  - `GOOGLE_API_KEY`
  - `GOOGLE_GEMINI_MODEL`
- Run the Server: `python main.py`

import streamlit as st
import json
import numpy as np
from openai import OpenAI
import os
from dotenv import load_dotenv

# Φόρτωση ρυθμίσεων
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# 1. Ρύθμιση Σελίδας
st.set_page_config(
    page_title="Vouli-AI: Νομικός Βοηθός",
    page_icon="🏛️",
    layout="centered"
)


# 2. Φόρτωση Δεδομένων
@st.cache_data
def load_knowledge_base():
    chunks = []
    if os.path.exists("chunks.jsonl"):
        with open("chunks.jsonl", "r", encoding="utf-8") as f:
            for line in f:
                chunks.append(json.loads(line))
    return chunks


chunks = load_knowledge_base()

# --- UI ΕΦΑΡΜΟΓΗΣ ---
st.title("🏛️️ Vouli-AI: Ο Ψηφιακός σου Βοηθός Νομοθεσίας")
st.markdown("""
Αυτός ο βοηθός χρησιμοποιεί Τεχνητή Νοημοσύνη για να αναλύει το νομοθετικό έργο της Βουλής. 
Μπορείτε να ρωτήσετε οτιδήποτε για τους νόμους που έχουν καταχωρηθεί στο σύστημα.
""")

with st.sidebar:
    st.header("📌 Πληροφορίες")
    st.write("Ο βοηθός χρησιμοποιεί δεδομένα από το API της Βουλής.")
    if chunks:
        sources = list(set([c['source'] for c in chunks]))
        st.subheader("📚 Ενεργοί Νόμοι:")
        for s in sources:
            st.caption(f"• {s}")

st.divider()


# --- ΛΟΓΙΚΗ ΑΠΑΝΤΗΣΕΩΝ ---
def get_answer(question):
    if not chunks:
        return "Λυπάμαι, αλλά δεν υπάρχουν δεδομένα στη βάση μου."

    try:
        q_emb = client.embeddings.create(
            input=[question],
            model="text-embedding-ada-002"  # Χρησιμοποιούμε το σταθερό
        ).data[0].embedding

        scores = []
        for ch in chunks:
            similarity = np.dot(q_emb, ch["embedding"])
            scores.append((similarity, ch))
        scores.sort(key=lambda x: x[0], reverse=True)

        context = "\n\n".join([f"[Πηγή: {ch['source']}, Σελ: {ch['page']}]: {ch['text']}" for _, ch in scores[:5]])

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system",
                 "content": "Είσαι έμπειρος νομικός βοηθός. Απάντα βασισμένος ΑΠΟΚΛΕΙΣΤΙΚΑ στο κείμενο. Αναφέρε Πηγή και Σελίδα."},
                {"role": "user", "content": f"Κείμενο:\n{context}\n\nΕρώτηση: {question}"}
            ],
            temperature=0.2
        )
        return response.choices[0].message.content
    except Exception as e:
        return "⚠️ Προσωρινό σφάλμα σύνδεσης. Δοκιμάστε ξανά."


# --- CHAT INTERFACE ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Πώς μπορώ να βοηθήσω;"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.markdown(prompt)
    with st.chat_message("assistant"):
        answer = get_answer(prompt)
        st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})
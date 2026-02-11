import streamlit as st
import json
import os
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

# Ρυθμίσεις σελίδας
st.set_page_config(page_title="Vouli-AI Assistant", page_icon="⚖️")
st.title("⚖️ Vouli-AI: Νομικός Βοηθός")
st.markdown("Ρωτήστε με οτιδήποτε για τον **Νόμο 5083/2024** (Επιστολική Ψήφος).")

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Φόρτωση δεδομένων
@st.cache_resource
def load_data():
    with open("chunks.jsonl", "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

chunks = load_data()

def get_answer(question):
    q_emb = client.embeddings.create(input=[question], model="text-embedding-3-small").data[0].embedding
    scores = [(np.dot(q_emb, ch["embedding"]), ch) for ch in chunks]
    scores.sort(key=lambda x: x[0], reverse=True)
    
    context = "\n\n".join([f"[Σελίδα {c[1]['page']}]: {c[1]['text']}" for c in scores[:5]])
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "Είσαι έμπειρος νομικός. Απάντα βασισμένος στο κείμενο. Ανάφερε τις σελίδες."},
            {"role": "user", "content": f"Κείμενο:\n{context}\n\nΕρώτηση: {question}"}
        ]
    )
    return response.choices[0].message.content

# Chat UI
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Πώς μπορώ να βοηθήσω;"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Αναζήτηση στα άρθρα του νόμου..."):
            answer = get_answer(prompt)
            st.markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})

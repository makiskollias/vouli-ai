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
    page_title="Vouli-AI: Βοηθός Νομοθεσίας",
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
st.title("🏛️ Vouli-AI: Ο Ψηφιακός σου Βοηθός Νομοθεσίας")
st.markdown("""
Αυτός ο βοηθός χρησιμοποιεί Τεχνητή Νοημοσύνη για να απαντά **μόνο** με βάση τους νόμους που έχουν καταχωρηθεί στο σύστημα,
παραθέτοντας **Πηγή & Σελίδα**.
""")

with st.sidebar:
    st.header("📌 Πληροφορίες")
    st.write("Ο βοηθός χρησιμοποιεί δεδομένα από το API της Βουλής / επίσημα κείμενα που έχουν εισαχθεί.")
    st.divider()

    top_k = st.slider("🔎 Αποσπάσματα (top-k)", 3, 12, 6, 1)
    min_sim = st.slider("🛡️ Ελάχιστη ομοιότητα (threshold)", 0.10, 0.40, 0.18, 0.01)

    if chunks:
        sources = sorted(list(set([c.get('source', 'Άγνωστη Πηγή') for c in chunks])))
        st.subheader("📚 Ενεργοί Νόμοι/Πηγές:")
        for s in sources:
            st.caption(f"• {s}")

st.divider()

# --- ΒΟΗΘΗΤΙΚΑ ---
def cosine_sim(a, b) -> float:
    a = np.array(a, dtype=np.float32)
    b = np.array(b, dtype=np.float32)
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)

@st.cache_data(show_spinner=False)
def embed_text(text: str):
    # Προτεινόμενο πιο σύγχρονο embedding model:
    # text-embedding-3-small (value) ή text-embedding-3-large (quality)
    return client.embeddings.create(
        input=[text],
        model="text-embedding-3-small"
    ).data[0].embedding

def build_context(scored_chunks):
    # Συμμαζεμένη μορφοποίηση context με σαφείς αναφορές
    parts = []
    for sim, ch in scored_chunks:
        source = ch.get("source", ch.get("source_id", "Άγνωστη Πηγή"))
        page = ch.get("page", "?")
        text = ch.get("text", "")
        parts.append(f"[Πηγή: {source} | Σελίδα: {page}]\n{text}")
    return "\n\n---\n\n".join(parts)

# --- ΛΟΓΙΚΗ ΑΠΑΝΤΗΣΕΩΝ ---
def get_answer(question: str) -> str:
    if not chunks:
        return "Δεν υπάρχουν ακόμη δεδομένα στη βάση. Χρειάζεται εισαγωγή νόμων/εγγράφων."

    try:
        q_emb = embed_text(question)

        scores = []
        for ch in chunks:
            ch_emb = ch.get("embedding")
            if not ch_emb:
                continue
            sim = cosine_sim(q_emb, ch_emb)
            scores.append((sim, ch))

        if not scores:
            return "Δεν βρέθηκαν embeddings στη βάση (λείπει το πεδίο 'embedding' στα chunks)."

        scores.sort(key=lambda x: x[0], reverse=True)
        top = scores[:top_k]

        # Guardrail: αν ακόμα και το καλύτερο αποτέλεσμα είναι χαμηλό, καλύτερα “δεν ξέρω”
        if top[0][0] < min_sim:
            return (
                "Δεν βρήκα επαρκές σχετικό απόσπασμα στα διαθέσιμα κείμενα για να απαντήσω με ασφάλεια.\n\n"
                "Δοκίμασε να:\n"
                "- αναφέρεις αριθμό νόμου/άρθρο/παράγραφο, ή\n"
                "- κάνεις την ερώτηση πιο συγκεκριμένη."
            )

        context = build_context(top)

        system_prompt = (
            "Είσαι βοηθός νομοθεσίας.\n"
            "Κανόνες:\n"
            "1) Απάντα ΜΟΝΟ από το δοσμένο Κείμενο.\n"
            "2) Για ΚΑΘΕ ισχυρισμό/πρόταση που αφορά περιεχόμενο νόμου, βάλε παραπομπή με μορφή: (Πηγή, Σελίδα).\n"
            "3) Αν δεν υπάρχει σαφής πληροφορία στο Κείμενο, πες ξεκάθαρα: «Δεν προκύπτει από τα διαθέσιμα αποσπάσματα».\n"
            "4) Μην προσθέτεις εξωτερική γνώση και μην κάνεις εικασίες.\n"
        )

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Κείμενο:\n{context}\n\nΕρώτηση: {question}"}
            ],
            temperature=0.2
        )
        return response.choices[0].message.content

    except Exception:
        return "⚠️ Προσωρινό σφάλμα. Δοκίμασε ξανά."

# --- CHAT INTERFACE ---

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
        with st.spinner("Σκέφτεται..."):
            answer = get_answer(prompt)
            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
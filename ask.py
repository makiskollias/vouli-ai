import json
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_embedding(text):
    # Μετατρέπει το κείμενο σε διάνυσμα
    return client.embeddings.create(input=[text], model="text-embedding-3-small").data[0].embedding

def cosine_similarity(v1, v2):
    # Απλή μαθηματική σύγκριση ομοιότητας
    sumxx, sumyy, sumxy = 0, 0, 0
    for i in range(len(v1)):
        x = v1[i]; y = v2[i]
        sumxx += x*x; sumyy += y*y; sumxy += x*y
    return sumxy / (sumxx**0.5 * sumyy**0.5)

# 1. Φόρτωση των δεδομένων
print("Φόρτωση δεδομένων...")
with open("chunks.jsonl", "r", encoding="utf-8") as f:
    chunks = [json.loads(line) for line in f]

# Σημείωση: Στο MVP αυτό, θα υπολογίζουμε τα embeddings "on the fly" 
# για τα chunks για να μη σε μπλέξω με βάσεις δεδομένων ακόμα.
# (Αν τα chunks είναι πολλά, κανονικά τα αποθηκεύουμε προ-υπολογισμένα).

def ask_question(question):
    print(f"Αναζήτηση για: {question}")
    q_emb = get_embedding(question)
    
    # 2. Υπολογισμός ομοιότητας (Brute force για το demo)
    for ch in chunks:
        # Αν δεν έχει embedding το chunk, το φτιάχνουμε (μόνο για την πρώτη φορά)
        if "embedding" not in ch:
            ch["embedding"] = get_embedding(ch["text"])
            
    chunks.sort(key=lambda x: cosine_similarity(q_emb, x["embedding"]), reverse=True)
    
    # Παίρνουμε τα 3 καλύτερα
    context = "\n\n".join([f"[Σελίδα {c['page']}]: {c['text']}" for c in chunks[:3]])
    
    # 3. Ερώτηση στο GPT
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "Είσαι ένας έμπειρος νομικός βοηθός. Απάντησε στην ερώτηση χρησιμοποιώντας ΜΟΝΟ το κείμενο που σου παρέχεται. Αν η απάντηση δεν υπάρχει, πες 'Δεν βρέθηκε πληροφορία'. Στο τέλος ανάφερε τις σελίδες."},
            {"role": "user", "content": f"Κείμενο:\n{context}\n\nΕρώτηση: {question}"}
        ]
    )
    return response.choices[0].message.content

if __name__ == "__main__":
    user_q = input("Ρώτα κάτι για το νόμο: ")
    print("\n--- ΑΠΑΝΤΗΣΗ ---")
    print(ask_question(user_q))

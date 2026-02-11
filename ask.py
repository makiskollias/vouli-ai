import json
import os
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Φόρτωση των έτοιμων δεδομένων
with open("chunks.jsonl", "r", encoding="utf-8") as f:
    chunks = [json.loads(line) for line in f]

def ask_question(question):
    # Embedding μόνο για την ερώτηση
    q_emb = client.embeddings.create(input=[question], model="text-embedding-3-small").data[0].embedding
    
    # Γρήγορη αναζήτηση με dot product (αφού τα embeddings είναι normalized)
    scores = []
    for ch in chunks:
        score = np.dot(q_emb, ch["embedding"])
        scores.append((score, ch))
    
    scores.sort(key=lambda x: x[0], reverse=True)
    top_chunks = scores[:3]
    
    context = "\n\n".join([f"[Σελίδα {c[1]['page']}]: {c[1]['text']}" for c in top_chunks])
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "Απάντα αυστηρά βάσει του κειμένου. Ανάφερε τις σελίδες."},
            {"role": "user", "content": f"Κείμενο:\n{context}\n\nΕρώτηση: {question}"}
        ]
    )
    return response.choices[0].message.content

if __name__ == "__main__":
    while True:
        user_q = input("\nΡώτα κάτι (ή 'exit'): ")
        if user_q.lower() == 'exit': break
        print("\n" + ask_question(user_q))

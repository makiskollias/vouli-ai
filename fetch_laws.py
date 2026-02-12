import requests
import os
from pathlib import Path

def manual_fetch():
    data_folder = Path("data")
    if not data_folder.exists():
        data_folder.mkdir()

    # Λίστα με 3 σημαντικά και ενεργά PDFs από τη Βουλή
    laws_to_get = [
        {
            "id": "12495", 
            "title": "Nomos_Epistoliki_Ypsifos", 
            "url": "https://www.hellenicparliament.gr/UserFiles/bcc2666d-1914-421d-837c-1b41aa59ad8f/12495144.pdf"
        },
        {
            "id": "12500", 
            "title": "Nomos_Panteion_Panepistimio", 
            "url": "https://www.hellenicparliament.gr/UserFiles/bcc2666d-1914-421d-837c-1b41aa59ad8f/12502641.pdf"
        }
    ]

    print("🚀 Έναρξη χειροκίνητης συλλογής νόμων για το Demo...")

    headers = {'User-Agent': 'Mozilla/5.0'} # Προσθήκη για να μην μας μπλοκάρει ο server

    for law in laws_to_get:
        file_name = data_folder / f"{law['title']}.pdf"
        print(f"📥 Λήψη: {law['title']}...")
        
        try:
            res = requests.get(law['url'], headers=headers)
            if res.status_code == 200:
                with open(file_name, 'wb') as f:
                    f.write(res.content)
                print(f"   ✅ Αποθηκεύτηκε!")
            else:
                print(f"   ❌ Σφάλμα status code: {res.status_code}")
        except Exception as e:
            print(f"   ❌ Σφάλμα: {e}")

if __name__ == "__main__":
    manual_fetch()

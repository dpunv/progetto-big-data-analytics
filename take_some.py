import json
import random
import os

# --- Configuration ---
TARGET_SIZE = 150000  # Numero di embeddings da estrarre
CHECKPOINT_FILE = 'embeddings_checkpoint.json'
OUTPUT_FILE = 'embeddings.json'

def main():
    # 1. Controlla se embeddings.json esiste già con abbastanza dati
    if os.path.exists(OUTPUT_FILE):
        print(f"Checking existing {OUTPUT_FILE}...")
        try:
            with open(OUTPUT_FILE, 'r') as f:
                existing_data = json.load(f)
            
            if len(existing_data) >= TARGET_SIZE:
                print(f"✅ {OUTPUT_FILE} already exists with {len(existing_data)} embeddings (>= {TARGET_SIZE}).")
                print(f"   Vector size: {len(existing_data[0]['embedding'])} dimensions")
                print("   Skipping extraction. Delete the file to regenerate.")
                return
            else:
                print(f"⚠️  {OUTPUT_FILE} exists but only has {len(existing_data)} embeddings (< {TARGET_SIZE}).")
                print("   Will regenerate with random sampling...")
        except (json.JSONDecodeError, KeyError, IndexError) as e:
            print(f"⚠️  {OUTPUT_FILE} exists but is corrupted: {e}")
            print("   Will regenerate...")
    
    # 2. Carica il checkpoint completo
    print(f"\n📂 Loading {CHECKPOINT_FILE}...")
    if not os.path.exists(CHECKPOINT_FILE):
        print(f"❌ Error: {CHECKPOINT_FILE} not found!")
        print("   Please ensure the checkpoint file exists in the current directory.")
        return
    
    try:
        with open(CHECKPOINT_FILE, 'r') as f:
            full_data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"❌ Error: {CHECKPOINT_FILE} is corrupted: {e}")
        return
    
    total_available = len(full_data)
    print(f"   Total embeddings available: {total_available}")
    
    if total_available == 0:
        print("❌ Error: Checkpoint file is empty!")
        return
    
    # Verifica dimensione vettore
    try:
        vector_size = len(full_data[0]['embedding'])
        print(f"   Vector size: {vector_size} dimensions")
    except (KeyError, IndexError) as e:
        print(f"❌ Error: Invalid data structure in checkpoint: {e}")
        return
    
    # 3. Campiona random
    sample_size = min(TARGET_SIZE, total_available)
    
    if sample_size < total_available:
        print(f"\n🎲 Randomly sampling {sample_size} embeddings from {total_available}...")
        # Usa random.sample per garantire che non ci siano duplicati
        sampled_data = random.sample(full_data, sample_size)
    else:
        print(f"\n⚠️  Requested {TARGET_SIZE} but only {total_available} available.")
        print(f"   Using all {total_available} embeddings (no sampling needed).")
        sampled_data = full_data
    
    # 4. Salva il risultato
    print(f"\n💾 Saving {len(sampled_data)} embeddings to {OUTPUT_FILE}...")
    try:
        with open(OUTPUT_FILE, 'w') as f:
            json.dump(sampled_data, f)
        
        # Verifica dimensione file
        file_size_mb = os.path.getsize(OUTPUT_FILE) / (1024 * 1024)
        print(f"✅ Successfully created {OUTPUT_FILE}")
        print(f"   Embeddings: {len(sampled_data)}")
        print(f"   File size: {file_size_mb:.2f} MB")
        print(f"   Vector dimensions: {vector_size}")
        
    except IOError as e:
        print(f"❌ Error saving file: {e}")

if __name__ == "__main__":
    # Imposta seed per riproducibilità (opzionale)
    # Se vuoi campioni diversi ogni volta, commenta questa riga
    random.seed(42)
    
    main()
import json

data = []
with open('embeddings_mid.json', 'r') as f:
    data = json.load(f)[:50_000]

print(data)

with open('embeddings.json', 'w') as f:
    json.dump(data, f)
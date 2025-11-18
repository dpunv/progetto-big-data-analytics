import json

data = []
with open('embeddings_150k.json', 'r') as f:
    data = json.load(f)[:30_000]

print(data)

with open('embeddings.json', 'w') as f:
    json.dump(data, f)
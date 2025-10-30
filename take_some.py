import json

data = {}
with open('embeddings_checkpoint.json', 'r') as f:
    data = json.load(f)

with open('embeddings.json', 'w') as f:
    json.dump(data[:100000], f)

print(len(data[9999]['embedding']))
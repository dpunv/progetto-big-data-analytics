import json

data = {}
with open('embeddings.json', 'r') as f:
    data = json.load(f)

"""with open('embeddings.json', 'w') as f:
    json.dump(data[:10000], f)
"""
print(len(data[9999]['embedding']))
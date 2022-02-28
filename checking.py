import json

print('hi')
f = open('data_10.json',"r")
data = json.loads(f.read())

print(data['mapping'])
print(len(data['labels']))
print(data['labels'][8000])
print(len(data['spectral_centroid']))
print(len(data['chromagram']))

f.close()
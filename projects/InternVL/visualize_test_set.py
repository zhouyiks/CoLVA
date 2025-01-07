import os
import json

image_folder = './data/DiagrammaticReasoning/'
data_file = './data//DiagrammaticReasoning/test.json'
with open(data_file, 'r') as f:
    data = json.load(f)

type_map = {
    'yangshiguilv': 'Style',
    'shuliangguilv': 'Quantity',
    'weizhiguilv': 'Positional',
    'shuxingguilv': 'Attribute',
    'kongjianguilv': 'Spatial',
    'other': 'Others',
}

type_set = set()
difficulty_set = set()

for data_item in data:
    type_set.add(data_item['type'])
    difficulty_set.add(data_item['difficulty'])

type_set_ = []
for type_name in type_set:
    type_name = type_name.split('/')[0]
    type_name = type_map[type_name]
    type_set_.append(type_name)
print(type_set_)
print(difficulty_set)
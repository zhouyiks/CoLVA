import os

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("folder_path", type=str, )
args = parser.parse_args()

files_names = os.listdir(args.folder_path)
ids = []
for name in files_names:
    if '.txt' not in name:
        pass
    else:
        _id = name.replace('.txt', '')
        ids.append(_id)

final_text = ''
for _id in ids:
    final_text += _id
    final_text += '\n'
with open(os.path.join(args.folder_path, 'id_list.txt'), 'w', encoding='utf-8') as file:
    file.write(final_text)
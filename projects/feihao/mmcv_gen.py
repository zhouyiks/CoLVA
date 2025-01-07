import copy
import os
import json

def get_lines(datas, _str):
    for data in datas:
        if _str in data:
            return data
    raise NotImplementedError

def parse_txt(path, n_image):
    with open(path, 'r') as f:
        datas = f.readlines()
    question = f"There are {n_image} images numbered from 1 to {n_image}. Which of the following images contains the model plane in the center of Image 1?"


    cands = get_lines(datas, "CAND").replace("CAND: [", '').replace(']', '').replace("\n", "")
    cands_ = cands.split(",")
    cands = []
    for cand in cands_:
        if cand != '':
            cands.append(cand.replace("\'", "").strip())

    types = get_lines(datas, "TYPE").replace("TYPE: ", "").replace("\n", "").replace(":", "").replace("\'", "").split(',')
    types_ = []
    for _type in types:
        if _type == '':
            pass
        else:
            types_.append(_type)
    types = types_
    if len(types) == 0:
        return None, None, None, None
    types = [int(type.strip()) for type in types]

    answer = get_lines(datas, "ANSWER").replace("ANSWER: ", '').replace("ANSWER:", "").replace("\n", "").strip()
    return question, cands, types, answer

# create and set the save path
if not os.path.exists('./achieved'):
    os.mkdir('./achieved')
if not os.path.exists('./achieved/images/'):
    os.mkdir('./achieved/images')

save_image_path = './achieved/images'
save_json_path = './achieved/anno.json'

final_json_data = {
    "task": "video object matching",
    "data_source": "MMVM",
    "type": "comprehension",
    "modality": {
        "in": ["image", "text"],
        "out": ["text"]
    },
    "version": 1.0,
}


# reformat the data
_MAX_ITEMS = 50
mmvm_root = './match_bench/'
data_instances = os.listdir(mmvm_root)

category_datas = [[] for i in range(8)]
for i, data_instance in enumerate(data_instances):
    instance_folder_path = os.path.join(mmvm_root, data_instance)
    _files = os.listdir(instance_folder_path)
    n_imgs = 0
    for _file in _files:
        if '.png' in _file or '.jpg' in _file:
            n_imgs += 1
    question, cands, types, answer = parse_txt(os.path.join(instance_folder_path, 'anno.txt'), n_imgs)
    if question is None:
        continue

    if max(types) >= 9 or min(types) < 1:
        print(types, '-----------**************---------------------')
        continue
    print(types)
    _cur_nums = [len(category_datas[_idx-1]) for _idx in types]
    _item_nums = _cur_nums[0]
    _select_type = types[0]
    for _type, _num in zip(types[1:], _cur_nums[1:]):
        if _num < _item_nums:
            _select_type = _type
            _item_nums = _num
    if _item_nums >= _MAX_ITEMS:
        continue

    _id = str(10000+i)[1:]
    _data = {"id": "vm{}".format(_id)}
    # copy
    drt_folder = os.path.join('./achieved/images', _id)
    if not os.path.exists(drt_folder):
        os.mkdir(drt_folder)
    os.system(f"cp -pr {instance_folder_path}/*.png {drt_folder}/")
    os.system(f"cp -pr {instance_folder_path}/*.jpg {drt_folder}/")
    prompt = question + ' Please select one option from the options as the answer: '
    for cand in cands:
        prompt += cand
        prompt += ' '
    prompt = prompt.strip() + '.'

    print(prompt, ' ', answer, '\n')
    _data["input"] = {"video_folder": drt_folder.replace('/achieved', ''), "prompt": prompt}
    _data["output"] = {"answer": answer}

    category_datas[_select_type - 1].append(_data)


print([len(item) for item in category_datas])

for i, data in enumerate(category_datas):
    _save_data = copy.deepcopy(final_json_data)
    _save_data["data"] = data
    with open(os.path.join("./achieved/", f"{i}.json"), "w") as f:
        json.dump(_save_data, f)


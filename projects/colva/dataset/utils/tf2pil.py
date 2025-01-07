import tensorflow as tf
from PIL import Image
import json


save_directories = {
    'general': './images/*',
    'google_apps': 'google_apps/*',
    'install': 'install/*',
    'single': 'single/*',
    'web_shopping': 'web_shopping/*',
}

dataset_directories = {
    'general': 'general/*',
    'google_apps': 'google_apps/*',
    'install': 'install/*',
    'single': 'single/*',
    'web_shopping': 'web_shopping/*',
}

def get_episode(dataset):
    episode = []
    episode_id = None
    for d in dataset:
        ex = tf.train.Example()
        ex.ParseFromString(d)
        ep_id = ex.features.feature['episode_id'].bytes_list.value[0].decode('utf-8')
        if episode_id is None:
            episode_id = ep_id
            episode.append(ex)
        elif ep_id == episode_id:
            episode.append(ex)
        else:
            break
    return episode

def _decode_image(
    example,
    image_height,
    image_width,
    image_channels,
):
    image = tf.io.decode_raw(
        example.features.feature['image/encoded'].bytes_list.value[0],
        out_type=tf.uint8,
    )

    height = tf.cast(image_height, tf.int32)
    width = tf.cast(image_width, tf.int32)
    n_channels = tf.cast(image_channels, tf.int32)

    return tf.reshape(image, (height, width, n_channels))

# general_need_files = []
# json_data = json.load(open('./gpt4v_android_general_detailed_caption_bbox.json'))
# for item in json_data:
#     general_need_files.append(item['image'].split('/')[-1])
# json_data = json.load(open('./gpt4v_android_general_QA_bbox.json'))
# for item in json_data:
#     if item['image'].split('/')[-1] in general_need_files:
#         continue
#     general_need_files.append(item['image'].split('/')[-1])

# google_apps_need_files = []
# json_data = json.load(open('./gpt4v_android_google_apps_detailed_caption_bbox.json'))
# for item in json_data:
#     google_apps_need_files.append(item['image'].split('/')[-1])
# json_data = json.load(open('./gpt4v_android_google_apps_QA_bbox.json'))
# for item in json_data:
#     if item['image'].split('/')[-1] in google_apps_need_files:
#         continue
#     google_apps_need_files.append(item['image'].split('/')[-1])

# install_need_files = []
# json_data = json.load(open('./gpt4v_android_install_detailed_caption_bbox.json'))
# for item in json_data:
#     install_need_files.append(item['image'].split('/')[-1])
# json_data = json.load(open('./gpt4v_android_install_QA_bbox.json'))
# for item in json_data:
#     if item['image'].split('/')[-1] in install_need_files:
#         continue
#     google_apps_need_files.append(item['image'].split('/')[-1])


# single_need_files = []
# json_data = json.load(open('./gpt4v_android_single_detailed_caption_bbox.json'))
# for item in json_data:
#     single_need_files.append(item['image'].split('/')[-1])
# json_data = json.load(open('./gpt4v_android_single_QA_bbox.json'))
# for item in json_data:
#     if item['image'].split('/')[-1] in single_need_files:
#         continue
#     single_need_files.append(item['image'].split('/')[-1])

# web_shopping_need_files = []
# json_data = json.load(open('./gpt4v_android_web_shopping_detailed_caption_bbox.json'))
# for item in json_data:
#     web_shopping_need_files.append(item['image'].split('/')[-1])
# json_data = json.load(open('./gpt4v_android_web_shopping_QA_bbox.json'))
# for item in json_data:
#     if item['image'].split('/')[-1] in web_shopping_need_files:
#         continue
#     web_shopping_need_files.append(item['image'].split('/')[-1])

# need_files = {
#     'general': general_need_files,
#     'google_apps': google_apps_need_files,
#     'install': install_need_files,
#     'single': single_need_files,
#     'web_shopping': web_shopping_need_files,
# }


for dataset_name in [ 'web_shopping']:
    filenames = tf.io.gfile.glob(dataset_directories[dataset_name])
    for filename in filenames:
        raw_dataset = tf.data.TFRecordDataset(filename, compression_type='GZIP').as_numpy_iterator()
        episode = get_episode(raw_dataset)
        for i, example in enumerate(episode):
            image_height = example.features.feature['image/height'].int64_list.value[0]
            image_width = example.features.feature['image/width'].int64_list.value[0]
            image_channels = example.features.feature['image/channels'].int64_list.value[0]
            episode_id = example.features.feature['episode_id'].bytes_list.value[0].decode('utf-8')
            image = _decode_image(example, image_height, image_width, image_channels)
            pil_img = tf.keras.utils.array_to_img(image)
            if 'step_id' in example.features.feature:
                step_id = example.features.feature['step_id'].int64_list.value[0]
                pil_img.save(f'./images/{dataset_name}_{episode_id}_{step_id}.png')
                print('saving ', f'./images/{dataset_name}_{episode_id}_{step_id}.png')
            else:
                pil_img.save(f'./images/{dataset_name}_{episode_id}.png')
                print('saving ', f'./images/{dataset_name}_{episode_id}.png')
print('Done.')




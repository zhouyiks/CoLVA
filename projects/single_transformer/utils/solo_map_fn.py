from xtuner.utils import DEFAULT_IMAGE_TOKEN


def solo_data_llava_map_fn(example):
    messages = example['conversations']
    input = ''
    conversation = []
    while messages and messages[0]['from'] == 'gpt':
        # Skip the first one if it is from gpt
        messages = messages[1:]
    for msg in messages:
        # For SOLO data: Image SFT pairs and language data.
        if 'from' in msg.keys() or 'role' in msg.keys():

            if msg['from'] == 'human' or msg['from'] == 'user' or msg['role'] == 'user':
                if DEFAULT_IMAGE_TOKEN in msg['value']:
                    msg['value'] = msg['value'].replace(DEFAULT_IMAGE_TOKEN,
                                                        '').strip()
                    msg['value'] = DEFAULT_IMAGE_TOKEN + '\n' + msg['value']
                    msg['value'] = msg['value'].strip()
                input += msg['value']

            elif msg['from'] == 'gpt' or msg['from'] == 'model' or \
                    msg['role'] == 'assistant' or msg['from'] == 'assistant' or \
                    msg['from'] == 'assistnat':
                conversation.append({'input': input, 'output': msg['value']})
                input = ''
        else:
            raise NotImplementedError
    return {'conversation': conversation}
from lmdeploy import pipeline, TurbomindEngineConfig
from lmdeploy.vl import load_image

model = 'pretrained/internvl/InternVL2-Llama3-76B-AWQ/'
image = load_image('https://raw.githubusercontent.com/open-mmlab/mmdeploy/main/tests/data/tiger.jpeg')
pipe = pipeline(model, backend_config=TurbomindEngineConfig(session_len=8192))

prompts = [('Please describe this image.', image),
           ('Please describe the tiger.', image),
           ('Please describe the grass.', image)]
response = pipe(prompts)
for rsp in response:
    print('\n\n-------------------*****************************-------------------\n\n')
    print(rsp.text)
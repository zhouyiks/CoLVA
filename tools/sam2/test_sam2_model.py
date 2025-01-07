from projects.llava_sam2.models.sam2 import SAM2
import numpy as np
from PIL import Image
import torch

IMG_PATH = 'assets/view.jpg'
IMG_SIZE = 1024
img_mean=(0.485, 0.456, 0.406)
img_std=(0.229, 0.224, 0.225)


def prepare():
    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True



if __name__ == '__main__':
    prepare()
    model = SAM2()
    model.eval()
    model.to(torch.device('cuda'))


    img_pil = Image.open(IMG_PATH)
    img_np = np.array(img_pil.convert("RGB").resize((IMG_SIZE, IMG_SIZE)))
    if img_np.dtype == np.uint8:  # np.uint8 is expected for JPEG images
        img_np = img_np / 255.0
    else:
        raise NotImplementedError
    img = torch.from_numpy(img_np).permute(2, 0, 1)

    img_mean = torch.tensor(img_mean, dtype=torch.float32)[:, None, None]
    img_std = torch.tensor(img_std, dtype=torch.float32)[:, None, None]
    img -= img_mean
    img /= img_std

    images = img.unsqueeze(0).repeat(5, 1, 1, 1)


    # Start
    language_embd = torch.ones((1, 1, 256), dtype=torch.float32, device=torch.device('cuda'))
    a = model.inject_language_embd(images, language_embd)
    print(1)

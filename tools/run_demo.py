import matplotlib.pyplot as plt
import matplotlib.pylab as pylab

import requests
from io import BytesIO
from PIL import Image
import numpy as np
pylab.rcParams['figure.figsize'] = 20, 12
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.engine.predictor_glip import GLIPDemo
import argparse
import pdb

def load(url_or_path):
    """
    Given an url or a path, this loads the file and
    """
    if url_or_path.startswith("http"):
        response = requests.get(url_or_path)
        pil_image = Image.open(BytesIO(response.content)).convert("RGB")
        # convert to BGR format
        image = np.array(pil_image)[:, :, [2, 1, 0]]
    else:
        image = np.array(Image.open(url_or_path).convert("RGB"))[:, :, [2, 1, 0]]
    return image


parser = argparse.ArgumentParser(description="PyTorch Object Detection Inference")
parser.add_argument("--config", default="configs/pretrain/glip_Swin_T_O365_GoldG.yaml", metavar="FILE", help="path to config file", type=str)
parser.add_argument("--weight", default="OUTPUTS/GLIP_MODEL4/model_0020000.pth", metavar="FILE", help="path to weight file", type=str)
parser.add_argument("--image", default="http://farm4.staticflickr.com/3693/9472793441_b7822c00de_z.jpg", metavar="FILE", help="path to weight file", type=str)
parser.add_argument("--conf", default=0.4, type=float)
parser.add_argument("--caption", default="", type=str)
parser.add_argument("--ground_tokens", default=None, type=str)

args = parser.parse_args()

# update the config options with the config file
# manual override some options
cfg.local_rank = 0
cfg.num_gpus = 1
cfg.merge_from_file(args.config)
cfg.merge_from_list(["MODEL.WEIGHT", args.weight])
cfg.merge_from_list(["MODEL.DEVICE", "cuda"])

glip_demo = GLIPDemo(
    cfg,
    min_image_size=800,
    show_mask_heatmaps=False
)

athetics_params = {
    "skip_name": False, # whether we overlay the phrase over the box
    "override_color": (255, 255, 255), # box color, default is white
    "text_size": 1.0,
    "text_pixel": 3,
    "box_alpha": 1.0,
    "box_pixel": 5,
    "text_offset_original": 8, # distance between text and box
}

image = load(args.image)
specified_tokens = args.ground_tokens.split(";") if args.ground_tokens is not None else None

result, _ = glip_demo.run_on_web_image(
    image,
    args.caption,
    args.conf,
    specified_tokens,
    **athetics_params)

plt.imshow(result[:, :, [2, 1, 0]])
plt.axis("off")
plt.savefig(args.image.replace('.png', "_demo.png").replace('.jpg', "_demo.jpg").replace('.jpeg', "_demo.jpeg"), bbox_inches='tight', pad_inches=0) # save as xxx_demo.xxx

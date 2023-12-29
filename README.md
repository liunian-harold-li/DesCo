# DesCo
This is the code for the paper [DesCo: Learning Object Recognition with Rich Language Descriptions (NeurIPS 2023)](https://arxiv.org/pdf/2306.14060.pdf).

Checkout the huggingface demo at [link](https://huggingface.co/spaces/zdou0830/desco).

## Installation and Setup

***Environment***
This repo requires Pytorch>=1.9 and torchvision. We recommend using docker to setup the environment. You can use this pre-built docker image ``docker pull pengchuanzhang/maskrcnn:ubuntu18-py3.7-cuda10.2-pytorch1.9`` or this one ``docker pull pengchuanzhang/pytorch:ubuntu20.04_torch1.9-cuda11.3-nccl2.9.9`` depending on your GPU.

<!-- ```
docker run --name desco  -it --runtime=nvidia --ipc=host pengchuanzhang/pytorch:ubuntu20.04_torch1.9-cuda11.3-nccl2.9.9˜
``` -->

Then install the following packages:
```
pip install einops shapely timm yacs tensorboardX ftfy prettytable pymongo sentence_transformers fastcluster openai transformers==4.11 wandb protobuf==3.20.1
python setup.py build develop --user
```

## Models

Model | LVIS MiniVal (AP) | OmniLabel (AP) | Config  | Weight
-- | -- | -- | -- | --
DesCo-GLIP (Tiny) | 34.6| 23.8 | [config](configs/pretrain_new/desco_glip.yaml) | [weight](https://huggingface.co/harold/DesCo/blob/main/desco_glip_tiny.pth)
DesCo-FIBER (Base) | [39.5](https://huggingface.co/harold/DesCo/blob/main/desco_fiber_lvisbest.pth) [1] | 29.3 | [config](configs/pretrain_new/desco_glip.yaml) | [weight](https://huggingface.co/harold/DesCo/blob/main/desco_fiber_base.pth)


[1] For DesCo-FIBER, we find it benefitial to early stop for LVIS evaluation. Thus we provide both the best checkpoint for LVIS evaluation and the final checkpoint.


## Quick Start

```
export GPUS=0
export CHECKPOINT=OUTPUTS/GLIP/desco_glip_tiny.pth
export CONFIG=configs/pretrain_new/desco_glip.yaml

CUDA_VISIBLE_DEVICES=$GPUS python tools/run_demo.py --config $CONFIG --weight $CHECKPOINT --image tools/pics/1.png --conf 0.5 --caption "a train besides sidewalk" --ground_tokens "train;sidewalk"
```

Just need a better example now

`ground_tokens` specifies which tokens we wish the model to ground to, separated by `;`; if it is not specified, the script will use NLTK to extract noun phrases automatically.


## Pre-Training
Below we provide scripts to pre-train DesCo-GLIP/FIBER.  We used 8 A6000 GPUs for pre-training; learning rate should be adjusted according to the batch size.

Pre-Training DesCo-GLIP:
```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 \
    tools/train_net.py \
    --config-file configs/pretrain_new/glip.yaml  \
    --skip-test \
    --wandb_name GLIP \
    SOLVER.IMS_PER_BATCH 16 \
    SOLVER.BASE_LR 0.00005 \
    SOLVER.MAX_ITER 300000 \
    SOLVER.MAX_NEG_PER_BATCH 1.0
```

Pre-Training DesCo-FIBER:
```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 \
    tools/train_net.py \
    --config-file configs/pretrain_new/fiber.yaml  \
    --skip-test \
    --wandb_name FIBER \
    SOLVER.IMS_PER_BATCH 16 \
    SOLVER.BASE_LR 0.00002 \
    SOLVER.MAX_ITER 200000 \
    MODEL.WEIGHT MODEL/fiber_coarse_then_fine.pth \
    SOLVER.MAX_NEG_PER_BATCH 1.0
```

Notes:
- In the current config, CC3M is not included. One could add `bing_caption_train_no_coco` to `DATA.TRAIN` to enable training on CC3M. Training without CC3M should give similar performance on OmniLabel and slightly worse performance on LVIS than reported in the paper.
- Checkpoints will be saved to `OUTPUTS/{wandb_name}`; `--use_wandb` can be specified in the arguments to enable logging to [wandb](https://wandb.ai).
- `SOLVER.MAX_NEG_PER_BATCH` needs to be set to `1.0` to enable training with full-negative prompts.

## Evaluation on Benchmarks

### LVIS

```
export GPUS=0,1,2,3,4,5,6,7
export GPU_NUM=8
export CHECKPOINT=OUTPUTS/GLIP/desco_glip_tiny.pth
export MODEL_CONFIG=configs/pretrain_new/glip.yaml

CUDA_VISIBLE_DEVICES=${GPUS}  python -m torch.distributed.launch --nproc_per_node=${GPU_NUM} \
    tools/test_grounding_net.py \
    --config-file ${MODEL_CONFIG} \
    --task_config configs/lvis/val.yaml \
    \
    --weight ${CHECKPOINT} \
    OUTPUT_DIR OUTPUTS/GLIP \
    TEST.EVAL_TASK detection  \
    TEST.CHUNKED_EVALUATION 8 TEST.IMS_PER_BATCH ${GPU_NUM} SOLVER.IMS_PER_BATCH ${GPU_NUM} TEST.MDETR_STYLE_AGGREGATE_CLASS_NUM 3000 MODEL.RETINANET.DETECTIONS_PER_IMG 300 MODEL.FCOS.DETECTIONS_PER_IMG 300 MODEL.ATSS.DETECTIONS_PER_IMG 300 MODEL.ROI_HEADS.DETECTIONS_PER_IMG 300  \
    DATASETS.OD_TO_GROUNDING_VERSION description.gpt.v10.infer.v1 \
    DATASETS.DESCRIPTION_FILE tools/files/lvis_v1.description.v1.json
```

Useful notes:

- `TEST.IMS_PER_BATCH` should be equal to `GPU_NUM`; the current evaluation script only supports inference on 1 image per GPU.
- Since there are over 1000 categories in  `TEST.CHUNKED_EVALUATION` specifies how many categories we put into one prompt. Thus, we need to run the model multiple times for one image with different prompts. We recommend evaluating with 8 GPUs and evaluating on `minival` takes several hours.
- `DATASETS.OD_TO_GROUNDING_VERSION` specifies how we convert the category names into descriptions. It is used in `data/dataset/_od_to_description.py`.
- The default evaluation protocol uses the [fixed AP](https://github.com/achalddave/large-vocab-devil).


### OmniLabel
```
export GPUS=0
export GPU_NUM=1
export CHECKPOINT=OUTPUTS/GLIP/desco_glip_tiny.pth
export MODEL_CONFIG=configs/pretrain_new/glip.yaml

CUDA_VISIBLE_DEVICES=${GPUS} python -m torch.distributed.launch --nproc_per_node=${GPU_NUM} \
    tools/test_net_omnilabel.py \
    --config-file ${MODEL_CONFIG} \
    --weight ${CHECKPOINT} \
    --task_config configs/omnilabel/omnilabel_val_eval.yaml \
    --chunk_size 20 \
    OUTPUT_DIR OUTPUTS/${MODEL_NAME} \
    TEST.IMS_PER_BATCH ${GPU_NUM} \
    DATASETS.TEST "('omnilabel_val_coco',)"
```

- Supported evaluation datasets (set by `DATASETS.TEST`) include `omnilabel_val_o365`, `omnilabel_val_coco`, `omnilabel_val_oi_v5`, and `omnilabel_val`.

###



## Useful Notes

- The core of DesCo is to construct the training prompts and maintain the correspondance between boxes and entities in the prompt; these functionalities are mostly implemented in `data/datasets/_caption_aug.py` and `data/dataset/_od_to_description.py`.

- We have implemented several different versions of the prompt construction process. They are controlled by the `OD_TO_GROUNDING_VERSION` (for detection data such as Objects365), `CAPTION_AUGMENTATION_VERSION` (for gold grounding data such as GoldG and Flickr30K), and `CC_CAPTION_AUGMENTATION_VERSION` (for web data such as CC3M) fields in the config.

- The rest of the code is similar to GLIP/FIBER. We made no changes to the architecture; thus the weights are compatable with GLIP/FIBER repo.

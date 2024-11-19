import argparse
import os
import sys

import numpy as np
import json
import torch
from PIL import Image,ImageColor
import torch.nn.functional as F  

sys.path.append(os.path.join(os.getcwd(), "GroundingDINO"))
sys.path.append(os.path.join(os.getcwd(), "segment_anything"))


# Grounding DINO
import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from transformers import AutoTokenizer
from typing import Any, Dict, List

# segment anything
from segment_anything import (
    sam_model_registry,
    #sam_hq_model_registry,
    SamPredictor
)
import cv2
import numpy as np
import matplotlib.pyplot as plt


def load_image(image_path):
    # load image
    image_pil = Image.open(image_path).convert("RGB")  # load image

    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None)  # 3, h, w
    return image_pil, image


def load_model(model_config_path, model_checkpoint_path, device):
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    print(load_res)
    _ = model.eval()
    return model

def get_phrases_from_positionmap(
    posmap: torch.BoolTensor, tokenized: Dict, tokenizer: AutoTokenizer ,
):  
    if posmap.dim() == 1:
        non_zero_idx = posmap.nonzero(as_tuple=True)[0].tolist()
        token_ids = [tokenized["input_ids"][i] for i in non_zero_idx]
        return tokenizer.decode(token_ids),non_zero_idx
    else:
        raise NotImplementedError("posmap must be 1-dim")

def get_grounding_output(model, image, caption, box_threshold, text_threshold, with_logits=True, device="cpu"):
    caption = caption.lower()
    caption = caption.strip()
    if not caption.endswith("."):
        caption = caption + "."
    model = model.to(device)
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image[None], captions=[caption])
    logits = outputs["pred_logits"].cpu().sigmoid()[0]  
    boxes = outputs["pred_boxes"].cpu()[0] 
    logits.shape[0]

    # filter output
    logits_filt = logits.clone()
    boxes_filt = boxes.clone()
    filt_mask = logits_filt.max(dim=1)[0] > box_threshold
    # filt_mask = logits_filt.max(dim=1)[0] >0
    logits_filt = logits_filt[filt_mask]  
    boxes_filt = boxes_filt[filt_mask]  
    logits_filt.shape[0]

    # get phrase
    tokenlizer = model.tokenizer
    tokenized = tokenlizer(caption)
    len_tokenized = len(tokenized.input_ids)

    # build pred
    pred_phrases = []
    index_list = []
    for logit, box in zip(logits_filt, boxes_filt):
        pred_phrase ,non_zero_idx= get_phrases_from_positionmap(logit > text_threshold, tokenized, tokenlizer)
        if non_zero_idx not in index_list:
            if with_logits:
                pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
            else:
                pred_phrases.append(pred_phrase)
            index_list.append(non_zero_idx)
    len_index = len(index_list)
    boxes_filt=boxes_filt[0:len_index]
    # Use zip to combine list_1 and list_2 into a tuple list 
    combined_lists = list(zip(index_list, boxes_filt))    
    # Use the sorted function and lambda expressions to sort combined_lists by the first element of list_1 sublist
    sorted_combined_lists = sorted(combined_lists, key=lambda x: x[0][0])  
    # Split the sorted list of tuples into two new lists
    index_list, boxes_filt = zip(*sorted_combined_lists)
    index_list = list(index_list)
    index_list.append(len_tokenized)
    boxes_filt = torch.stack(boxes_filt, dim=0)  
    return boxes_filt, pred_phrases ,index_list

def get_new_masks(masks,output_dir):
    new_mask_list=[]
    for i ,mask in enumerate(masks):
        new_mask=torch.zeros_like(masks[0],device=masks.device)
        for j ,m in enumerate(masks):
            if j!=i:
                new_mask+=m
        new_mask=new_mask.bool()
        new_mask = (~new_mask).float()
        invert_mask=new_mask.cpu().numpy().squeeze(0)*255
        save_path=os.path.join(output_dir, "invert_masks")
        os.makedirs(save_path, exist_ok=True)
        cv2.imwrite(save_path+f'/invert_masks_{i}.png', invert_mask)
        new_mask_list.append(new_mask.unsqueeze(0))
    return torch.cat(new_mask_list,dim=0)


def get_masks_and_pred_phrases(config_file,grounded_checkpoint,seg_type,seg_checkpoint,
                               image_path,text_prompt,box_threshold,text_threshold,device):
    
    text_prompt_list = text_prompt.split(",")
    current_working_dir = os.getcwd()
    output_dir = os.path.join(current_working_dir, "outputs")
    # make dir
    os.makedirs(output_dir, exist_ok=True)
    # load image
    image_pil, image = load_image(image_path)
    # load model
    model = load_model(config_file, grounded_checkpoint, device=device)

    # visualize raw image
    image_pil.save(os.path.join(output_dir, "raw_image.jpg"))
    mask_list=[] 
    
    for indx,text_prompts in enumerate(text_prompt_list):
        # run grounding dino model
        boxes_filt, pred_phrases ,index_list= get_grounding_output(
            model, image, text_prompts, box_threshold, text_threshold, device=device
        )

        # initialize SAM
        predictor = SamPredictor(sam_model_registry[seg_type](checkpoint=seg_checkpoint).to(device))
        img = cv2.imread(image_path)
        img= cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        predictor.set_image(img)

        size = image_pil.size
        H, W = size[1], size[0]
        for i in range(boxes_filt.size(0)):
            boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
            boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
            boxes_filt[i][2:] += boxes_filt[i][:2]

        boxes_filt = boxes_filt.cpu()
        transformed_boxes = predictor.transform.apply_boxes_torch(boxes_filt, img.shape[:2]).to(device)
        masks, _, _ = predictor.predict_torch(
            point_coords = None,
            point_labels = None,
            boxes = transformed_boxes.to(device),
            multimask_output = False,
        )
        mask_list.append(masks.float())
        
        # Save mask as image file
        # for i, mask in enumerate(masks):
        #     mask_array = mask.cpu().numpy()[0] * 255 
        #     mask_image = Image.fromarray(mask_array.astype(np.uint8))
        #     file_path = os.path.join(output_dir, f"{text_prompts}_mask_{i}.png")
        #     mask_image.save(file_path)
    mask_list= torch.cat(mask_list,dim=0)
    modify_masks=get_new_masks(mask_list,output_dir)
    return masks,index_list,modify_masks
    
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser("Grounded-Segment-Anything Demo", add_help=True)
    parser.add_argument("--config", type=str,default="" ,required=True, help="path to config file")
    parser.add_argument(
        "--grounded_checkpoint", type=str, required=True, help="path to checkpoint file"
    )
    parser.add_argument(
        "--sam_version", type=str, default="vit_h", required=False, help="SAM ViT version: vit_b / vit_l / vit_h"
    )
    parser.add_argument(
        "--sam_checkpoint", type=str, required=False, help="path to sam checkpoint file"
    )
    parser.add_argument("--input_image", type=str, required=True, help="path to image file")
    parser.add_argument("--text_prompt", type=str, required=True, help="text prompt")
    parser.add_argument(
        "--output_dir", "-o", type=str, default="outputs", required=True, help="output directory"
    )

    parser.add_argument("--box_threshold", type=float, default=0.4, help="box threshold")
    parser.add_argument("--text_threshold", type=float, default=0.25, help="text threshold")

    parser.add_argument("--device", type=str, default="cpu", help="running on cpu only!, default=False")
    args = parser.parse_args()

    # cfg
    config_file = args.config  # change the path of the model config file
    grounded_checkpoint = args.grounded_checkpoint  # change the path of the model
    sam_version = args.sam_version
    sam_checkpoint = args.sam_checkpoint
    image_path = args.input_image
    text_prompt = args.text_prompt
    output_dir = args.output_dir
    box_threshold = args.box_threshold
    text_threshold = args.text_threshold
    device = args.device
    get_masks_and_pred_phrases(config_file,grounded_checkpoint,sam_version,sam_checkpoint,
                               image_path,text_prompt,box_threshold,text_threshold,device)
 
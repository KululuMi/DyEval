import os 
import numpy as np
import argparse
import json
from PIL import Image
import torch
import random
import shutil

import torch
import os
from tqdm import tqdm
import numpy as np
from pathlib import Path
import PIL
import sys
from torchvision import transforms
from torchvision.transforms import ToPILImage
from PIL import Image
import open_clip

to_pil = ToPILImage()

def get_text_img_similarity(init_input, generated_image):
    text = tokenizer([init_input]).to(device)

    gen_batch = [clip_preprocess(i).unsqueeze(0) for i in [generated_image]]
    gen_batch = torch.concatenate(gen_batch).to(device)

    gen_feat = clip_model.encode_image(gen_batch)
    text_feat = clip_model.encode_text(text)
    
    gen_feat = gen_feat / gen_feat.norm(dim=1, keepdim=True)
    text_feat = text_feat / text_feat.norm(dim=1, keepdim=True)

    return (gen_feat @ text_feat.t()).mean().item()

def get_img_similarity(oir_image, generated_image):
    gen_batch = [clip_preprocess(i).unsqueeze(0) for i in [generated_image]]
    gen_batch = torch.concatenate(gen_batch).to(device)
    ori_gen_batch = [clip_preprocess(i).unsqueeze(0) for i in [oir_image]]
    ori_gen_batch = torch.concatenate(ori_gen_batch).to(device)

    gen_feat = clip_model.encode_image(gen_batch)
    ori_gen_feat = clip_model.encode_image(ori_gen_batch)
    
    gen_feat = gen_feat / gen_feat.norm(dim=1, keepdim=True)
    ori_gen_feat = ori_gen_feat / ori_gen_feat.norm(dim=1, keepdim=True)

    return (gen_feat @ ori_gen_feat.t()).mean().item()

def get_direction_similarity(source_prompt, target_prompt, source_image, target_image, direction_loss):
    with torch.no_grad():
        source_text = tokenizer([source_prompt]).to(device)
        source_text_feat = clip_model.encode_text(source_text)
        target_text = tokenizer([target_prompt]).to(device)
        target_text_feat = clip_model.encode_text(target_text)
        text_direction = (target_text_feat - source_text_feat).mean(axis=0, keepdim=True)
        text_direction /= text_direction.norm(dim=-1, keepdim=True)

        source_batch = [clip_preprocess(i).unsqueeze(0) for i in [source_image]]
        source_batch = torch.concatenate(source_batch).to(device)
        target_batch = [clip_preprocess(i).unsqueeze(0) for i in [target_image]]
        target_batch = torch.concatenate(target_batch).to(device)
        source_img_feat = clip_model.encode_image(source_batch)
        target_img_feat = clip_model.encode_image(target_batch)
        img_direction = (target_img_feat - source_img_feat)
        
        if img_direction.sum() == 0:
            target_img_feat = clip_model.encode_image(target_batch + 1e-6)
            img_direction = (target_img_feat - source_img_feat)
        img_direction /= (img_direction.clone().norm(dim=-1, keepdim=True))

        # text_direction和img_direction
    
    return 1 - direction_loss(text_direction, img_direction).mean().item()

class DirectionLoss(torch.nn.Module):

    def __init__(self, loss_type='mse'):
        super(DirectionLoss, self).__init__()

        self.loss_type = loss_type

        self.loss_func = {
            'mse':    torch.nn.MSELoss,
            'cosine': torch.nn.CosineSimilarity,
            'mae':    torch.nn.L1Loss
        }[loss_type]()

    def forward(self, x, y):
        if self.loss_type == "cosine":
            return 1. - self.loss_func(x, y)
        
        return self.loss_func(x, y)

direction_loss = DirectionLoss(loss_type="cosine")

def get_k_img(img_list, score_list, n_k):
    # 获取分数最高的两个句子及其分数
    highest_scores = sorted(zip(img_list, score_list), key=lambda x: x[1], reverse=True)[:n_k]
    highest_img, highest_scores = zip(*highest_scores)
    highest_img = list(highest_img)
    highest_scores = list(highest_scores)

    return highest_img, highest_scores

device = torch.device("cuda")

parser = argparse.ArgumentParser()
parser.add_argument('--ori_path', type=str) # 训练数据位置
parser.add_argument('--image_path', type=str) # 待筛选的数据
parser.add_argument('--prompt_src', type=str)
parser.add_argument('--n_k', type=int)
args = parser.parse_args()

# 加载clip
clip_model_name = "ViT-L-14"
clip_pretrain = "laion2b_s32b_b82k"
clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(clip_model_name, pretrained=clip_pretrain, device=device, jit=True)
tokenizer = open_clip.get_tokenizer(clip_model_name)

path = f"{args.image_path}"  # 待筛数据路径
new_path = f"{args.image_path}_filtered"
if os.path.exists(new_path):
    shutil.rmtree(new_path)  # 先删除
os.makedirs(new_path)  # 再新建（重新筛选）

# 加载图像
ori_image_names = []
ori_image_list = []
for i in os.listdir(args.ori_path):
    img = os.path.join(args.ori_path, i)
    ori_image_names.append(img)
    ori_image = Image.open(img).convert('RGB')
    ori_image_list.append(ori_image)

score_list = []
img_list = []
# import pdb;pdb.set_trace()
for i in os.listdir(path):
    # prompt_tar = i.split('__')[-1].split('.')[0]
    # prompt_tar = prompt_tar.replace("-", " ")
    img_path = os.path.join(path, i)  # 图片路径
    img = Image.open(img_path).convert('RGB')
    name = i.split('.')[0]  # 对应编辑目标
    # 计算编辑匹配度，挑最高的几张图片
    # score = get_text_img_similarity(prompt_tar, img)# - get_img_similarity(ori_image, img)
    score = get_img_similarity(ori_image, img)
    # score = 0
    # for ori_image in ori_image_list:
    #     score += get_direction_similarity(args.prompt_src, prompt_tar, ori_image, img, direction_loss)
    # score /= len(ori_image_list)

    img_list.append(img_path)
    score_list.append(score)
# import pdb;pdb.set_trace()
# 排序
highest_img, highest_scores = get_k_img(img_list, score_list, args.n_k)
print(highest_scores)
# 转存
for img_path in highest_img:
    target_img_path = f"{new_path}/{img_path.split('/')[-1]}"
    shutil.copy(img_path, target_img_path)
# # 复制原图到filtered中，保证效果
# for num in range(len(ori_image_names)):
#     ori_image_name = ori_image_names[num]
#     target_img_path = f"{new_path}/{num}__{args.prompt_src.replace(' ', '-')}.png"
#     shutil.copy(ori_image_name, target_img_path)
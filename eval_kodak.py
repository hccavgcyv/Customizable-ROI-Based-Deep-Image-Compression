import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import numpy as np
from tqdm import tqdm
import torchvision
from timm.scheduler.plateau_lr import PlateauLRScheduler
import torch.nn.functional as F
from PIL import Image
from pytorch_msssim import ms_ssim
import argparse
import math
import pandas as pd
import shutil
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms
from compressai.zoo.image import bmshj2018_factorized,cheng2020_attn
from compressai.datasets import ImageFolder
from compressai.zoo import models
# import xlwt
import clip
import compressai
# from model import Clip_Codec
from model_tic import *
# from compressai.utils.eval_model.__main__ import compute_metrics,compute_padding
from COCO_dataset import COCODataset_train,COCODataset_test
from COCO_dataset import COCODataset_train,COCODataset_test
from compressai.zoo.image import bmshj2018_factorized,cheng2020_attn,mbt2018

from thop import profile


def compute_padding(in_h: int, in_w: int, *, out_h=None, out_w=None, min_div=1):
    """Returns tuples for padding and unpadding.

    Args:
        in_h: Input height.
        in_w: Input width.
        out_h: Output height.
        out_w: Output width.
        min_div: Length that output dimensions should be divisible by.
    """
    if out_h is None:
        out_h = (in_h + min_div - 1) // min_div * min_div
    if out_w is None:
        out_w = (in_w + min_div - 1) // min_div * min_div

    if out_h % min_div != 0 or out_w % min_div != 0:
        raise ValueError(
            f"Padded output height and width are not divisible by min_div={min_div}."
        )

    left = (out_w - in_w) // 2
    right = out_w - in_w - left
    top = (out_h - in_h) // 2
    bottom = out_h - in_h - top

    pad = (left, right, top, bottom)
    unpad = (-left, -right, -top, -bottom)

    return pad, unpad


class AverageMeter:
    """Compute running average."""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count



def listdir(path, list_name):
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        list_name.append(file_path)

def main(argv):



    test_dataset = COCODataset_test()


    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)

    net_lseg = LSegNet(
            backbone="clip_vitl16_384",
            features=256,
            crop_size=256,
            arch_option=0,
            block_depth=0,
            activation="lrelu",
        )

    net_lseg.load_state_dict(torch.load("/path/to/lseg/checkpoint/lseg_minimal_e200.ckpt"))
    net_lseg.eval()
    net_lseg.to(device)


    for qp in [1]:
        model = TIC()

        checkpoint = torch.load("/path/to/checkpoint/best_loss.pth.tar", map_location=device)

        model_dict = model.state_dict()
        checkpoint_dict = checkpoint['state_dict']
        compatible_dict = {k: v for k, v in checkpoint_dict.items() if k in model_dict}
        model_dict.update(compatible_dict)
        model.load_state_dict(model_dict)
        model = model.cuda()
        model.eval()
        model.update(force=True)
        bpp = AverageMeter()
        psnr = AverageMeter()


        with torch.no_grad():
            with tqdm(total=len(test_dataloader.dataset), unit='img', ncols=200) as pbar:
                for i,batch  in enumerate(test_dataloader):

                    image = batch['image']
                    labels = batch['labels']
                    name = batch['name']
                    labels = list(itertools.chain(*labels))

                    name = ''.join(name)

                    original_file = f"{name}"
                    image = Image.open('/path/to/dataset/' + original_file).convert('RGB')
                    image = np.array(image)
                    image = image.astype(np.float32) / 255.0
                    image = np.expand_dims(image, axis=0)

                    image = np.transpose(image, (0, 3, 1, 2))



                    res = inference_clip(model, image,labels,net_lseg,name)

                    psnr.update(res['psnr'])
                    bpp.update(res['bpp'])

                    pbar.set_postfix(
                                     PSNR='{:.6f}'.format(res['psnr']),
                                      Bpp_loss='{:.6f}'.format(res['bpp']),
                                     )
                    pbar.update(image.shape[0])
        print(
            f"  Bpp: {bpp.avg:.6f} |"
            f"  PSNR: {psnr.avg:.6f} \n"

        )

@torch.no_grad()
def inference_clip(model, x, labels, net_lseg, name):

    name = ''.join(name)
    name_save = name[:-4]
    labels_save = labels[0]

    x = torch.tensor(x)
    x = x.cuda()
    h, w = x.size(2), x.size(3)
    pad, unpad = compute_padding(h, w, min_div=256)  # pad to allow 6 strides of 2

    x_padded = F.pad(x, pad, mode="constant", value=0)


    img_feat = net_lseg.forward(x_padded)
    img_feat_norm = torch.nn.functional.normalize(img_feat, dim=1)
    #
    prompt = clip.tokenize(labels).cuda()
    text_feat = net_lseg.clip_pretrained.encode_text(prompt)  # 1, 512
    text_feat_norm = torch.nn.functional.normalize(text_feat, dim=1)
    #
    cosine_similarity = torch.nn.CosineSimilarity(dim=1)
    similarity = cosine_similarity(
        img_feat_norm, text_feat_norm.unsqueeze(-1).unsqueeze(-1)
    )
    similarity = similarity.unsqueeze(0)





    out_enc = model(x_padded,similarity)

    # out_dec = model.decompress_1(out_enc["strings"], out_enc["shape"])

    out_enc["x_hat"] = F.pad(out_enc["x_hat"], unpad)





    # input images are 8bit RGB for now
    x = torch.round(torch.clamp(x * 255, 0, 255)).float()
    x_hat = torch.round(torch.clamp((out_enc["x_hat"]) * 255, 0, 255)).float()
    v_mse = torch.mean((x - x_hat) ** 2, [0, 1, 2, 3])
    v_psnr = torch.mean(20 * torch.log10(255 / torch.sqrt(v_mse)), 0)
    num_pixels = x.size(0) * x.size(2) * x.size(3)
    # bpp = (sum(len(s[0]) for s in out_enc["strings"])) * 8.0 / num_pixels
    bpp = sum(
        (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
        for likelihoods in out_enc["likelihoods"].values()
    )




    reconstructed_image = torchvision.transforms.ToPILImage(mode='RGB')(x_hat.to('cpu')[0] / 255)
    reconstructed_image.save("/path/to/image/{}_{}_{:.6f}_{:.6f}.png".format(name_save,labels_save,bpp,v_psnr))




    return {
        "psnr":v_psnr,
        "bpp": bpp,
    }

if __name__ == "__main__":
    main(sys.argv[1:])
# Copyright (c) 2021-2022, InterDigital Communications, Inc
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted (subject to the limitations in the disclaimer
# below) provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# * Neither the name of InterDigital Communications, Inc nor the names of its
#   contributors may be used to endorse or promote products derived from this
#   software without specific prior written permission.

# NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY
# THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
# CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT
# NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
# PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
# OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import argparse
import math
import numpy as np
import random
import shutil
import sys
import os
import time
import logging
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim

import torchvision

from torchvision import transforms

from compressai.datasets import ImageFolder
from compressai.zoo import image_models
from model_tic import *
# from test_token import MyDataset_train, MyDataset_test
from pytorch_msssim import ms_ssim as compute_ms_ssim
from torch.utils.data import DataLoader
# from ADE20K_label import ADE20KDataset
from COCO_dataset import COCODataset_train,COCODataset_test
from lseg import LSegNet
# from compressai.utils.eval_model.__main__ import compute_metrics,compute_padding


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


class RateDistortionLoss(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self, lmbda=0.0932):
        super().__init__()
        self.mse = nn.MSELoss()
        self.lmbda = lmbda

    def forward(self, output, target):
        N, _, H, W = target.size()
        out = {}
        num_pixels = N * H * W

        out["bpp_loss"] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output["likelihoods"].values()
        )
        out["mse_loss"] =torch.mean(((output["x_hat"] - target) * output["similarity"]) ** 2)

        out["loss"] = self.lmbda * 255 ** 2 * out["mse_loss"] + out["bpp_loss"]

        return out


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


class CustomDataParallel(nn.DataParallel):
    """Custom DataParallel to access the module methods."""

    def __getattr__(self, key):
        try:
            return super().__getattr__(key)
        except AttributeError:
            return getattr(self.module, key)


def init(args):
    base_dir = f'./pretrained/{args.model}/{args.quality_level}/'
    os.makedirs(base_dir, exist_ok=True)

    return base_dir


def setup_logger(log_dir):
    log_formatter = logging.Formatter("%(asctime)s [%(levelname)-5.5s]  %(message)s")
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    log_file_handler = logging.FileHandler(log_dir, encoding='utf-8')
    log_file_handler.setFormatter(log_formatter)
    root_logger.addHandler(log_file_handler)

    log_stream_handler = logging.StreamHandler(sys.stdout)
    log_stream_handler.setFormatter(log_formatter)
    root_logger.addHandler(log_stream_handler)

    logging.info('Logging file is %s' % log_dir)


def configure_optimizers(net, args):
    """Separate parameters for the main optimizer and the auxiliary optimizer.
    Return two optimizers"""

    parameters = {
        n
        for n, p in net.named_parameters()
        if not n.endswith(".quantiles") and p.requires_grad
    }
    aux_parameters = {
        n
        for n, p in net.named_parameters()
        if n.endswith(".quantiles") and p.requires_grad
    }

    # Make sure we don't have an intersection of parameters
    params_dict = dict(net.named_parameters())
    inter_params = parameters & aux_parameters
    union_params = parameters | aux_parameters

    assert len(inter_params) == 0
    # assert len(union_params) - len(params_dict.keys()) == 0

    optimizer = optim.Adam(
        (params_dict[n] for n in sorted(parameters)),
        lr=args.learning_rate,
    )
    aux_optimizer = optim.Adam(
        (params_dict[n] for n in sorted(aux_parameters)),
        lr=args.aux_learning_rate,
    )
    return optimizer, aux_optimizer


def train_one_epoch(
        model, net_lseg,criterion, train_dataloader, optimizer, aux_optimizer, epoch, clip_max_norm
):
    model.train()
    device = next(model.parameters()).device

    for i, batch in enumerate(train_dataloader):

        image = batch['image']
        labels = batch['labels']
        image = image.permute(0, 3, 1, 2)
        image = image.to(device)
        prompt = list(itertools.chain(*labels))

        cosine_similarity = torch.nn.CosineSimilarity(dim=1)
        batch_size = 8



        img_feat = net_lseg.forward(image)
        img_feat_norm = torch.nn.functional.normalize(img_feat, dim=1)

        '''prompt
        '''
        prompt = clip.tokenize(prompt).to(device)
        text_feat = net_lseg.clip_pretrained.encode_text(prompt)  # 1, 512
        text_feat_norm = torch.nn.functional.normalize(text_feat, dim=1)

        similarities=[]
        for _i in range(image.shape[0]):
            similarity = cosine_similarity(
                img_feat_norm[_i:_i+1,:,:,:], text_feat_norm[_i:_i+1,:].unsqueeze(-1).unsqueeze(-1)
            )
            similarities.append(similarity)
        similarity = torch.stack(similarities, dim=0)


        optimizer.zero_grad()
        aux_optimizer.zero_grad()
        out_net = model(image,similarity)

        out_criterion = criterion(out_net, image)
        out_criterion["loss"].backward()


        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
        optimizer.step()

        aux_loss = model.aux_loss()
        aux_loss.backward()
        aux_optimizer.step()


        if i * len(image) % 500 == 0:
            logging.info(
                f'[{i * len(image)}/{len(train_dataloader.dataset)}] | '
                f'Loss: {out_criterion["loss"].item():.5f} | '
                f'MSE loss: {out_criterion["mse_loss"].item():.5f} | '
                f'Bpp loss: {out_criterion["bpp_loss"].item():.4f} | '
                f"Aux loss: {aux_loss.item():.2f}"
            )




def test_epoch(epoch, net_lseg, test_dataloader, model, criterion):
    model.eval()
    device = next(model.parameters()).device

    loss = AverageMeter()
    bpp_loss = AverageMeter()
    mse_loss = AverageMeter()
    aux_loss = AverageMeter()
    mse_real = AverageMeter()
    psnr = AverageMeter()

    with torch.no_grad():
        for i, batch in enumerate(test_dataloader):
            image = batch['image']
            labels = batch['labels']
            name = batch['name']
            name = ''.join(name)
            name = name[:-4]

            image = image.permute(0, 3, 1, 2)
            image = image.to(device)
            prompt = list(itertools.chain(*labels))
            cosine_similarity = torch.nn.CosineSimilarity(dim=1)



            h, w = image.size(2), image.size(3)
            pad, unpad = compute_padding(h, w, min_div= 256)  # pad to allow 6 strides of 2
            image = F.pad(image, pad, mode="constant", value=0)

            img_feat = net_lseg.forward(image)
            img_feat_norm = torch.nn.functional.normalize(img_feat, dim=1)

            '''prompt
            '''
            prompt = clip.tokenize(prompt).to(device)
            text_feat = net_lseg.clip_pretrained.encode_text(prompt)  # 1, 512
            text_feat_norm = torch.nn.functional.normalize(text_feat, dim=1)

            similarity = cosine_similarity(
                img_feat_norm, text_feat_norm.unsqueeze(-1).unsqueeze(-1)
            )
            #
            similarity = similarity.unsqueeze(0)

            '''COCO GT mask
            '''
            # image_name = name
            # mask_file = f"{image_name}.png"
            # mask_path = os.path.join('/path/to/dataset/', mask_file)
            # if os.path.exists(mask_path):
            #     mask = Image.open(mask_path).convert('L')
            #     mask = np.array(mask)
            #     mask = np.where(mask != 0, 1, 0).astype(np.uint8)
            #     mask = torch.tensor(mask)
            #     mask = F.pad(mask, pad, mode="constant", value=0)
            #     similarity = mask.unsqueeze(0)
            #     similarity = similarity.unsqueeze(0)
            # else:
            #     similarity = torch.ones((1,1,image.shape[2], image.shape[3]))



            similarity = similarity.cuda()



            out_net = model(image, similarity)
            out_criterion = criterion(out_net, image)


            aux_loss.update(model.aux_loss())
            bpp_loss.update(out_criterion["bpp_loss"])
            loss.update(out_criterion["loss"])
            mse_loss.update(out_criterion["mse_loss"])

            out_net["x_hat"] = F.pad(out_net["x_hat"], unpad)
            x = F.pad(image, unpad)
            x = torch.round(torch.clamp((x) * 255, 0, 255)).float()

            x_hat = torch.round(torch.clamp((out_net["x_hat"]) * 255, 0, 255)).float()
            v_mse = torch.mean((x - x_hat) ** 2, [0, 1, 2, 3])
            v_psnr = torch.mean(20 * torch.log10(255 / torch.sqrt(v_mse)), 0)
            mse_real.update(v_mse)
            psnr.update(v_psnr)


    logging.info(
        f"Test epoch {epoch}: Average losses: "
        f"Loss: {loss.avg:.3f} | "
        f"Bpp loss: {bpp_loss.avg:.4f} | "
        f"MSE loss: {mse_loss.avg:.5f} | "
        f"PSNR: {psnr.avg:.6f} |"
        f"Aux loss: {aux_loss.avg:.2f}\n"
    )

    return loss.avg


def save_checkpoint(state, is_best, base_dir, filename="checkpoint.pth.tar"):
    torch.save(state, base_dir + filename)
    if is_best:
        shutil.copyfile(base_dir + filename, base_dir + "best_loss_checkpoint.pth.tar")




def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example training script.")
    parser.add_argument(
        "-m",
        "--model",
        default="tic",
        choices=image_models.keys(),
        help="Model architecture (default: %(default)s)",
    )
    parser.add_argument(
        "-d",
        "--dataset",
        default="/path/to/dataset/",
        type=str,
        required=True,
        help="Training dataset"
    )
    parser.add_argument(
        "-e",
        "--epochs",
        default=200,
        type=int,
        help="Number of epochs (default: %(default)s)",
    )
    parser.add_argument(
        "-lr",
        "--learning-rate",
        default=1e-4,
        type=float,
        help="Learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "-n",
        "--num-workers",
        type=int,
        default=4,
        help="Dataloaders threads (default: %(default)s)",
    )
    parser.add_argument(
        "-q",
        "--quality-level",
        type=int,
        default=3,
        help="Quality level (default: %(default)s)",
    )
    parser.add_argument(
        "--lambda",
        dest="lmbda",
        type=float,
        default=0.0483,
        help="Bit-rate distortion parameter (default: %(default)s)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size (default: %(default)s)"
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1,
        help="Test batch size (default: %(default)s)",
    )
    parser.add_argument(
        "--aux-learning-rate",
        default=1e-3,
        help="Auxiliary loss learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        nargs=2,
        default=(256, 256),
        help="Size of the patches to be cropped (default: %(default)s)",
    )
    parser.add_argument(
        "--cuda",
        action="store_true",
        help="Use cuda"
    )
    parser.add_argument(
        "--gpu-id",
        type=str,
        default='0',
        help="GPU ids (default: %(default)s)",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        default=True,
        help="Save model to disk"
    )
    parser.add_argument(
        "--seed",
        type=float,
        help="Set random seed for reproducibility"
    )
    parser.add_argument(
        "--clip_max_norm",
        default=1.0,
        type=float,
        help="gradient clipping max norm (default: %(default)s",
    )
    parser.add_argument(
        '--name',
        default=datetime.now().strftime('%Y-%m-%d_%H_%M_%S'),
        type=str,
        help='Result dir name',
    )
    parser.add_argument(
        "--checkpoint",
        default="/path/to/checkpoint/checkpoint.pth.tar",
        type=str,
        help="Path to a checkpoint"
    )
    args = parser.parse_args(argv)
    return args


def main(argv):
    args = parse_args(argv)
    base_dir = init(args)

    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)

    setup_logger(base_dir + '/' + time.strftime('%Y%m%d_%H%M%S') + '.log')
    msg = f'======================= {args.name} ======================='
    logging.info(msg)
    for k in args.__dict__:
        logging.info(k + ':' + str(args.__dict__[k]))
    logging.info('=' * len(msg))


    train_dataset = COCODataset_train()
    test_dataset = COCODataset_test()



    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)


    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"


    net = TIC()
    net = net.to(device)

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



    optimizer, aux_optimizer = configure_optimizers(net, args)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, verbose=True)
    criterion = RateDistortionLoss(lmbda=args.lmbda)

    last_epoch = 0
    if args.checkpoint:  # load from previous checkpoint
        logging.info("Loading", args.checkpoint)
        checkpoint = torch.load(args.checkpoint, map_location=device)
        last_epoch = checkpoint["epoch"] + 1
        net.load_state_dict(checkpoint["state_dict"], strict=False)
        # optimizer.load_state_dict(checkpoint["optimizer"])
        aux_optimizer.load_state_dict(checkpoint["aux_optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])

        total_params = sum(p.numel() for p in net.parameters())
        total_params += sum(p.numel() for p in net.buffers())
        print(f'{total_params:,} total parameters.')
        print(f'{total_params / (1024 * 1024):.2f}M total parameters.')
        total_trainable_params = sum(
            p.numel() for p in net.parameters() if p.requires_grad)
        print(f'{total_trainable_params:,} training parameters.')
        print(f'{total_trainable_params / (1024 * 1024):.2f}M training parameters.')

    best_loss = float("inf")
    for epoch in range(last_epoch, args.epochs):
        logging.info('======Current epoch %s ======' % epoch)
        train_one_epoch(
            net,
            net_lseg,
            criterion,
            train_dataloader,
            optimizer,
            aux_optimizer,
            epoch,
            args.clip_max_norm,
        )
        loss = test_epoch(epoch, net_lseg, test_dataloader, net, criterion)
        lr_scheduler.step(loss)

        is_best = loss < best_loss
        best_loss = min(loss, best_loss)

        if args.save:
            save_checkpoint(
                {
                    "epoch": epoch,
                    "state_dict": net.state_dict(),
                    "loss": loss,
                    "optimizer": optimizer.state_dict(),
                    "aux_optimizer": aux_optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                },
                is_best,
                base_dir
            )


if __name__ == "__main__":
    main(sys.argv[1:])

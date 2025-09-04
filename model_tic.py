import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from RSTB import RSTB, CausalAttentionModule
from compressai.ans import BufferedRansEncoder, RansDecoder
from timm.models.layers import trunc_normal_
from compressai.models.utils import conv, deconv, update_registered_buffers
from compressai.layers import AttentionBlock
import clip
from PIL import Image
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from lseg.lseg_net import LSegNet
# import cv2
# import random
import itertools


# device = "cuda" if torch.cuda.is_available() else "cpu"

# From Balle's tensorflow compression examples
SCALES_MIN = 0.11
SCALES_MAX = 256
SCALES_LEVELS = 64

device = "cuda"

def get_scale_table(min=SCALES_MIN, max=SCALES_MAX, levels=SCALES_LEVELS):
    return torch.exp(torch.linspace(math.log(min), math.log(max), levels))


class Binarizer(torch.autograd.Function):
    """
    An elementwise function that bins values
    to 0 or 1 depending on a threshold of
    0.5

    Input: a tensor with values in range(0,1)

    Returns: a tensor with binary values: 0 or 1
    based on a threshold of 0.5

    Equation(1) in paper
    """
    @staticmethod
    def forward(ctx, i):
        result = torch.where(i > 0.9, torch.tensor(1.0), torch.tensor(0.2))
        return result

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

def bin_values(x):
    return Binarizer.apply(x)




class TIC(nn.Module):
    """Neural image compression framework from
    Lu Ming and Guo, Peiyao and Shi, Huiqing and Cao, Chuntong and Ma, Zhan:
    `"Transformer-based Image Compression" <https://arxiv.org/abs/2111.06707>`, (DCC 2022).

    Args:
        N (int): Number of channels
        M (int): Number of channels in the expansion layers (last layer of the
            encoder and last layer of the hyperprior decoder)
        input_resolution (int): Just used for window partition decision
    """

    def __init__(self,  N=192, M=192):
        super().__init__()

        depths = [1, 2, 3, 1, 1]
        num_heads = [4, 8, 16, 16, 16]
        window_size = 8
        mlp_ratio = 4.
        qkv_bias = True
        qk_scale = None
        drop_rate = 0.
        attn_drop_rate = 0.
        drop_path_rate = 0.2
        norm_layer = nn.LayerNorm
        use_checkpoint = False

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.align_corners = True


        self.g_a0 = conv(3, N, kernel_size=5, stride=2)
        self.g_a1 = RSTB(dim=N,
                         input_resolution=(128, 128),
                         depth=depths[0],
                         num_heads=num_heads[0],
                         window_size=window_size,
                         mlp_ratio=mlp_ratio,
                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                         drop=drop_rate, attn_drop=attn_drop_rate,
                         drop_path=dpr[sum(depths[:0]):sum(depths[:1])],
                         norm_layer=norm_layer,
                         use_checkpoint=use_checkpoint,
                         )
        self.g_a2 = conv(N, N, kernel_size=3, stride=2)
        self.g_a3 = RSTB(dim=N,
                         input_resolution=(64, 64),
                         depth=depths[1],
                         num_heads=num_heads[1],
                         window_size=window_size,
                         mlp_ratio=mlp_ratio,
                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                         drop=drop_rate, attn_drop=attn_drop_rate,
                         drop_path=dpr[sum(depths[:1]):sum(depths[:2])],
                         norm_layer=norm_layer,
                         use_checkpoint=use_checkpoint,
                         )
        self.g_a4 = conv(N, N, kernel_size=3, stride=2)
        self.g_a5 = RSTB(dim=N,
                         input_resolution=(32, 32),
                         depth=depths[2],
                         num_heads=num_heads[2],
                         window_size=window_size,
                         mlp_ratio=mlp_ratio,
                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                         drop=drop_rate, attn_drop=attn_drop_rate,
                         drop_path=dpr[sum(depths[:2]):sum(depths[:3])],
                         norm_layer=norm_layer,
                         use_checkpoint=use_checkpoint,
                         )

        self.g_a6 = conv(N, M, kernel_size=3, stride=2)

        self.h_a0 = conv(M, N, kernel_size=3, stride=1)
        self.h_a1 = RSTB(dim=N,
                         input_resolution=(16, 16),
                         depth=depths[3],
                         num_heads=num_heads[3],
                         window_size=window_size // 2,
                         mlp_ratio=mlp_ratio,
                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                         drop=drop_rate, attn_drop=attn_drop_rate,
                         drop_path=dpr[sum(depths[:3]):sum(depths[:4])],
                         norm_layer=norm_layer,
                         use_checkpoint=use_checkpoint,
                         )
        self.h_a2 = conv(N, N, kernel_size=3, stride=2)
        self.h_a3 = RSTB(dim=N,
                         input_resolution=(8, 8),
                         depth=depths[4],
                         num_heads=num_heads[4],
                         window_size=window_size // 2,
                         mlp_ratio=mlp_ratio,
                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                         drop=drop_rate, attn_drop=attn_drop_rate,
                         drop_path=dpr[sum(depths[:4]):sum(depths[:5])],
                         norm_layer=norm_layer,
                         use_checkpoint=use_checkpoint,
                         )
        self.h_a4 = conv(N, N, kernel_size=3, stride=2)

        depths = depths[::-1]
        num_heads = num_heads[::-1]
        self.h_s0 = deconv(N, N, kernel_size=3, stride=2)
        self.h_s1 = RSTB(dim=N,
                         input_resolution=(8, 8),
                         depth=depths[0],
                         num_heads=num_heads[0],
                         window_size=window_size // 2,
                         mlp_ratio=mlp_ratio,
                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                         drop=drop_rate, attn_drop=attn_drop_rate,
                         drop_path=dpr[sum(depths[:0]):sum(depths[:1])],
                         norm_layer=norm_layer,
                         use_checkpoint=use_checkpoint,
                         )
        self.h_s2 = deconv(N, N, kernel_size=3, stride=2)
        self.h_s3 = RSTB(dim=N,
                         input_resolution=(16, 16),
                         depth=depths[1],
                         num_heads=num_heads[1],
                         window_size=window_size // 2,
                         mlp_ratio=mlp_ratio,
                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                         drop=drop_rate, attn_drop=attn_drop_rate,
                         drop_path=dpr[sum(depths[:1]):sum(depths[:2])],
                         norm_layer=norm_layer,
                         use_checkpoint=use_checkpoint,
                         )
        self.h_s4 = conv(N, M * 2, kernel_size=3, stride=1)

        self.g_s0 = deconv(M, N, kernel_size=3, stride=2)
        self.g_s1 = RSTB(dim=N,
                         input_resolution=(32, 32),
                         depth=depths[2],
                         num_heads=num_heads[2],
                         window_size=window_size,
                         mlp_ratio=mlp_ratio,
                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                         drop=drop_rate, attn_drop=attn_drop_rate,
                         drop_path=dpr[sum(depths[:2]):sum(depths[:3])],
                         norm_layer=norm_layer,
                         use_checkpoint=use_checkpoint,
                         )
        self.g_s2 = deconv(N, N, kernel_size=3, stride=2)
        self.g_s3 = RSTB(dim=N,
                         input_resolution=(64, 64),
                         depth=depths[3],
                         num_heads=num_heads[3],
                         window_size=window_size,
                         mlp_ratio=mlp_ratio,
                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                         drop=drop_rate, attn_drop=attn_drop_rate,
                         drop_path=dpr[sum(depths[:3]):sum(depths[:4])],
                         norm_layer=norm_layer,
                         use_checkpoint=use_checkpoint,
                         )
        self.g_s4 = deconv(N, N, kernel_size=3, stride=2)
        self.g_s5 = RSTB(dim=N,
                         input_resolution=(128, 128),
                         depth=depths[4],
                         num_heads=num_heads[4],
                         window_size=window_size,
                         mlp_ratio=mlp_ratio,
                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                         drop=drop_rate, attn_drop=attn_drop_rate,
                         drop_path=dpr[sum(depths[:4]):sum(depths[:5])],
                         norm_layer=norm_layer,
                         use_checkpoint=use_checkpoint,
                         )
        self.g_s6 = deconv(N, 3, kernel_size=5, stride=2)

        self.entropy_bottleneck = EntropyBottleneck(N)
        self.gaussian_conditional = GaussianConditional(None)
        self.context_prediction = CausalAttentionModule(M, M * 2)
        # self.attetionmap = AttentionBlock(M)

        self.entropy_parameters = nn.Sequential(
            nn.Conv2d(M * 12 // 3, M * 10 // 3, 1),
            nn.GELU(),
            nn.Conv2d(M * 10 // 3, M * 8 // 3, 1),
            nn.GELU(),
            nn.Conv2d(M * 8 // 3, M * 6 // 3, 1),
        )

        self.sub_net_leaky = nn.Sequential(
            conv(N,N,kernel_size=3,stride=2),
            nn.LeakyReLU()
        )

        self.sub_net0 = nn.Sequential(
            conv(N,64,kernel_size=1,stride=1),
            nn.ReLU()
        )
        self.sub_net1 = nn.Sequential(
            conv(64,64,kernel_size=3,stride=1),
            nn.ReLU()
        )
        self.sub_net2 = conv(64,N,kernel_size=1,stride=1)

        self.sub_net_channel = conv(N,M,kernel_size=1,stride=1)

        self.simi_net = nn.Sequential(
            conv(1,64,kernel_size=3,stride=2),
            nn.ReLU(),
            conv(64,128,kernel_size=3,stride=2),
            nn.ReLU(),
            conv(128, M, kernel_size=3, stride=2),
        )

        self.net_lseg = LSegNet(
            backbone="clip_vitl16_384",
            features=256,
            crop_size=256,
            arch_option=0,
            block_depth=0,
            activation="lrelu",
        )

        self.cosine_similarity = torch.nn.CosineSimilarity(dim=1)




        # self.con3_3 = conv(192,192,kernel_size=3,stride=1)

        self.tanh = nn.Tanh()
        self.softsign = nn.Softsign()
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()


        self.apply(self._init_weights)



    def g_a(self, x, x_size=None):
        if x_size is None:
            x_size = x.shape[2:4]
        x = self.g_a0(x)
        x = self.g_a1(x, (x_size[0] // 2, x_size[1] // 2))
        x = self.g_a2(x)
        x = self.g_a3(x, (x_size[0] // 4, x_size[1] // 4))
        x = self.g_a4(x)
        x = self.g_a5(x, (x_size[0] // 8, x_size[1] // 8))
        # x = self.g_a6(x)
        return x

    def g_s(self, x, x_size=None):
        if x_size is None:
            x_size = (x.shape[2] * 16, x.shape[3] * 16)
        x = self.g_s0(x)
        x = self.g_s1(x, (x_size[0] // 8, x_size[1] // 8))
        x = self.g_s2(x)
        x = self.g_s3(x, (x_size[0] // 4, x_size[1] // 4))
        x = self.g_s4(x)
        x = self.g_s5(x, (x_size[0] // 2, x_size[1] // 2))
        x = self.g_s6(x)
        return x

    def h_a(self, x, x_size=None):
        if x_size is None:
            x_size = (x.shape[2] * 16, x.shape[3] * 16)
        x = self.h_a0(x)
        x = self.h_a1(x, (x_size[0] // 16, x_size[1] // 16))
        x = self.h_a2(x)
        x = self.h_a3(x, (x_size[0] // 32, x_size[1] // 32))
        x = self.h_a4(x)
        return x

    def h_s(self, x, x_size=None):
        if x_size is None:
            x_size = (x.shape[2] * 64, x.shape[3] * 64)
        x = self.h_s0(x)
        x = self.h_s1(x, (x_size[0] // 32, x_size[1] // 32))
        x = self.h_s2(x)
        x = self.h_s3(x, (x_size[0] // 16, x_size[1] // 16))
        x = self.h_s4(x)
        return x


    def sub_impor_net(self,x):   # important map
        x1 = self.sub_net_leaky(x)

        x2 = self.sub_net0(x1)
        x2 = self.sub_net1(x2)
        x2 = self.sub_net2(x2)

        x2 = x1 + x2
        x3 = self.sub_net0(x2)
        x3 = self.sub_net1(x3)
        x3 = self.sub_net2(x3)

        x3 = x2 + x3
        x4 = self.sub_net0(x3)
        x4 = self.sub_net1(x4)
        x4 = self.sub_net2(x4)

        x_out = x4 + x3
        x_out = self.sub_net_channel(x_out)

        return x_out



    def aux_loss(self):
        """Return the aggregated loss over the auxiliary entropy bottleneck
        module(s).
        """
        aux_loss = sum(
            m.loss() for m in self.modules() if isinstance(m, EntropyBottleneck)
        )
        return aux_loss

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward(self, x, similarity):

        x_size = (x.shape[2], x.shape[3])

        h, w = x.size(2), x.size(3)

        similarity_loss = torch.where(similarity > 0.85, torch.tensor(1.0), torch.tensor(0.01))
        similarity_imp = torch.where(similarity > 0.85, torch.tensor(1.0), torch.tensor(0.01))

        similarity_up = F.interpolate(similarity_loss, scale_factor=2, mode='bilinear')
        similarity_up_repeated = similarity_up.repeat(1, 3, 1, 1)

        similarities_channel = self.simi_net(similarity_imp)
        similarities_sigmoid = torch.sigmoid(similarities_channel)


        y_codec = self.g_a(x, x_size) # y
        y_codec_a6 = self.g_a6(y_codec)


        y_import = self.sub_impor_net(y_codec)
        y_tanh = self.tanh(y_import)
        y_soft = self.softsign(y_tanh)



        y_imp = y_soft + similarities_sigmoid
        y = y_codec_a6 * y_imp


        z = self.h_a(y, x_size)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        params = self.h_s(z_hat, x_size)

        y_hat = self.gaussian_conditional.quantize(
            y, "noise" if self.training else "dequantize"
        )
        ctx_params = self.context_prediction(y_hat)
        gaussian_params = self.entropy_parameters(
            torch.cat((params, ctx_params), dim=1)
        )
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        _, y_likelihoods = self.gaussian_conditional(y, scales_hat, means=means_hat)
        x_hat = self.g_s(y_hat, x_size)

        return {
            "y_hat": y_hat,
            "y": y,
            "similarity":similarity_up_repeated,
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }


    def update(self, scale_table=None, force=False):
        """Updates the entropy bottleneck(s) CDF values.

        Needs to be called once after training to be able to later perform the
        evaluation with an actual entropy coder.

        Args:
            scale_table (bool): (default: None)  
            force (bool): overwrite previous values (default: False)

        Returns:
            updated (bool): True if one of the EntropyBottlenecks was updated.

        """
        if scale_table is None:
            scale_table = get_scale_table()
        self.gaussian_conditional.update_scale_table(scale_table, force=force)

        updated = False
        for m in self.children():
            if not isinstance(m, EntropyBottleneck):
                continue
            rv = m.update(force=force)
            updated |= rv
        return updated

    def load_state_dict(self, state_dict, strict=True):
        # Dynamically update the entropy bottleneck buffers related to the CDFs
        update_registered_buffers(
            self.entropy_bottleneck,
            "entropy_bottleneck",
            ["_quantized_cdf", "_offset", "_cdf_length"],
            state_dict,
        )
        update_registered_buffers(
            self.gaussian_conditional,
            "gaussian_conditional",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
            state_dict,
        )
        super().load_state_dict(state_dict, strict=strict)

    @classmethod
    def from_state_dict(cls, state_dict):
        """Return a new model instance from `state_dict`."""
        N = state_dict["g_a0.weight"].size(0)
        M = state_dict["g_a6.weight"].size(0)
        net = cls(N, M)
        net.load_state_dict(state_dict)
        return net

    # def compress(self, x,similarity):
    def compress(self, x):
        x = x.cuda()
        # similarity = similarity.to(device)
        x_size = (x.shape[2], x.shape[3])

    #     start_1 = time.time()
    #
    #     img_feat = self.net_lseg.forward(x)
    #     img_feat_norm = torch.nn.functional.normalize(img_feat, dim=1)
    # #
    #     prompt = clip.tokenize(similarity).cuda()
    #     text_feat = self.net_lseg.clip_pretrained.encode_text(prompt)  # 1, 512
    #     text_feat_norm = torch.nn.functional.normalize(text_feat, dim=1)
    # #
    #     similarity = self.cosine_similarity(
    #         img_feat_norm, text_feat_norm.unsqueeze(-1).unsqueeze(-1)
    #     )
    #     similarity = similarity.unsqueeze(0)
    #
    #     torch.cuda.synchronize()
    #
    #     inf_time = time.time() - start_1
    #
    #     print(inf_time)

        # #####在这里

        start = time.time()
        # similarity_down_1 = torch.where(similarity > 0.9, torch.tensor(1.0), torch.tensor(1.0))
        # similarities_repeated = self.simi_net(similarity_down_1)
        # similarities_repeated = torch.sigmoid(similarities_repeated)





        y_codec = self.g_a(x, x_size) # y

        # y_import = self.sub_impor_net(y_codec)
        # y_tanh = self.tanh(y_import)
        #
        # y_soft = self.softsign(y_tanh)


        y_codec_a6 = self.g_a6(y_codec)


        # y_imp = y_soft + similarities_repeated # 相似度* important map
        # y = y_codec_a6 * y_imp
        y = y_codec_a6
        # y = y_imp * y_codec_a6
        # y = self.sub_net_channel(y)

        # y = y_codec_a6 * similarities_repeated

        z = self.h_a(y)

        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        params = self.h_s(z_hat)

        s = 4  # scaling factor between z and y
        kernel_size = 5  # context prediction kernel size
        padding = (kernel_size - 1) // 2

        y_height = z_hat.size(2) * s
        y_width = z_hat.size(3) * s

        y_hat = F.pad(y, (padding, padding, padding, padding))

        # pylint: disable=protected-access
        cdf = self.gaussian_conditional._quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional._cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional._offset.reshape(-1).int().tolist()
        # pylint: enable=protected-access
        # print(cdf, cdf_lengths, offsets)
        y_strings = []
        for i in range(y.size(0)):
            encoder = BufferedRansEncoder()
            # Warning, this is slow...
            # TODO: profile the calls to the bindings...
            symbols_list = []
            indexes_list = []
            y_q_ = torch.zeros_like(y)
            indexes_ = torch.zeros_like(y)
            for h in range(y_height):
                for w in range(y_width):
                    y_crop = y_hat[
                             i: i + 1, :, h: h + kernel_size, w: w + kernel_size
                             ]
                    ctx_p = self.context_prediction(y_crop)
                    # 1x1 conv for the entropy parameters prediction network, so
                    # we only keep the elements in the "center"
                    p = params[i: i + 1, :, h: h + 1, w: w + 1]
                    gaussian_params = self.entropy_parameters(
                        torch.cat((p, ctx_p[i: i + 1, :, 2: 3, 2: 3]), dim=1)
                    )
                    scales_hat, means_hat = gaussian_params.chunk(2, 1)

                    indexes = self.gaussian_conditional.build_indexes(scales_hat)
                    y_q = torch.round(y_crop - means_hat)
                    y_hat[i, :, h + padding, w + padding] = (y_q + means_hat)[
                                                            i, :, padding, padding
                                                            ]
                    y_q_[i,:, h, w] = y_q[i, :, padding, padding]
                    indexes_[i,:, h, w] = indexes[i, :,0,0]

            flag = np.array(np.zeros(y_q_.shape[1]))
            for idx in range(y_q_.shape[1]):
                if torch.sum(torch.abs(y_q_[:, idx, :, :])) > 0:  # 全部大于0就设置标志位是1
                    flag[idx] = 1
            y_q_ = y_q_[:,np.nonzero(flag),...].squeeze()
            indexes_ = indexes_[:,np.nonzero(flag),...].squeeze()
            for h in range(y_height):
                for w in range(y_width):
                    # encoder.encode_with_indexes(
                    #     y_q_[:,np.nonzero(flag),h,w].squeeze().int().tolist(),
                    #     indexes_[:,np.nonzero(flag),h,w].squeeze().int().tolist(), cdf, cdf_lengths, offsets
                    # )
                    symbols_list.extend(y_q_[:,h,w].int().tolist())
                    indexes_list.extend(indexes_[:,h,w].squeeze().int().tolist())
            encoder.encode_with_indexes(
                symbols_list, indexes_list, cdf, cdf_lengths, offsets
            )
            string = encoder.flush()
            y_strings.append(string)
            print(flag.sum())

        torch.cuda.synchronize()  # 确保 model2 真正跑完
        t2 = time.time() - start
        # print(t2)

        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:],"flag":flag}

        # return {"test":similarity}

    def compress_1(self, x,similarity):
    # def compress_1(self, x):
        x = x.cuda()
        x_size = (x.shape[2], x.shape[3])

        similarity = similarity.cuda()
        # # #
        similarity_down_1 = torch.where(similarity == 0, torch.tensor(1e-4), torch.tensor(1.0))
        #
        #


        # similarity_down_1 = torch.where(similarity > 0.9, torch.tensor(1.0), torch.tensor(1e-4))
        #
        similarity_down_1 = F.interpolate(similarity_down_1, scale_factor=0.5, mode='bilinear')
        similarities_repeated = self.simi_net(similarity_down_1)
        similarities_repeated = torch.sigmoid(similarities_repeated)


        y_codec = self.g_a(x, x_size) # y

        y_import = self.sub_impor_net(y_codec)
        y_tanh = self.tanh(y_import)
        y_soft = self.softsign(y_tanh) # important2
        # y_soft = self.sigmoid(y_soft)

        y_codec_a6 = self.g_a6(y_codec)
        # y_codec_a6 = self.attetionmap(y_codec_a6)


        y_imp = similarities_repeated + y_soft # 相似度* important map


        y = y_codec_a6 * y_imp
        #
        # y= y_codec_a6 * y_tanh

        # cmap = ListedColormap(['yellow'])
        #
        # similarity_image = torch.where(similarity > 0.9, torch.tensor(1.0), torch.tensor(0.1))
        # similarity_image = F.interpolate(similarity_image, scale_factor=2, mode='bilinear')
        # abs = torch.abs(similarity_image)
        # mean = torch.mean(abs, axis=1, keepdims=True)
        # viz = mean.detach().cpu().numpy()
        # viz = viz[0]
        # viz = viz.squeeze()
        # plt.imshow(viz)
        # # # 保存图像
        # plt.imsave('/mnt/disk10T/xfx/CLIP/bird.png', viz)

        z = self.h_a(y)

        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        params = self.h_s(z_hat)

        s = 4  # scaling factor between z and y
        kernel_size = 5  # context prediction kernel size
        padding = (kernel_size - 1) // 2

        y_height = z_hat.size(2) * s
        y_width = z_hat.size(3) * s

        y_hat = F.pad(y, (padding, padding, padding, padding))

        # pylint: disable=protected-access
        cdf = self.gaussian_conditional._quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional._cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional._offset.reshape(-1).int().tolist()
        # pylint: enable=protected-access
        # print(cdf, cdf_lengths, offsets)

        y_strings = []
        for i in range(y.size(0)):
            encoder = BufferedRansEncoder()
            # Warning, this is slow...
            # TODO: profile the calls to the bindings...
            symbols_list = []
            indexes_list = []
            for h in range(y_height):
                for w in range(y_width):
                    y_crop = y_hat[
                             i: i + 1, :, h: h + kernel_size, w: w + kernel_size
                             ]
                    ctx_p = self.context_prediction(y_crop)
                    # 1x1 conv for the entropy parameters prediction network, so
                    # we only keep the elements in the "center"
                    p = params[i: i + 1, :, h: h + 1, w: w + 1]
                    gaussian_params = self.entropy_parameters(
                        torch.cat((p, ctx_p[i: i + 1, :, 2: 3, 2: 3]), dim=1)
                    )
                    scales_hat, means_hat = gaussian_params.chunk(2, 1)

                    indexes = self.gaussian_conditional.build_indexes(scales_hat)
                    y_q = torch.round(y_crop - means_hat)
                    y_hat[i, :, h + padding, w + padding] = (y_q + means_hat)[
                                                            i, :, padding, padding
                                                            ]

                    symbols_list.extend(y_q[i, :, padding, padding].int().tolist())
                    indexes_list.extend(indexes[i, :].squeeze().int().tolist())

            encoder.encode_with_indexes(
                symbols_list, indexes_list, cdf, cdf_lengths, offsets
            )

            string = encoder.flush()
            y_strings.append(string)

        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}

    def compress_2(self, x,similarity):
        # def compress_1(self, x):
        x = x.cuda()
        x_size = (x.shape[2], x.shape[3])

        #####在这里
        similarity_down_1 = torch.where(similarity > 0.9, torch.tensor(1.0), torch.tensor(0.1))
        similarities_repeated = self.simi_net(similarity_down_1)
        similarities_repeated = torch.sigmoid(similarities_repeated)

        y_codec = self.g_a(x, x_size)  # y

        y_import = self.sub_impor_net(y_codec)
        y_tanh = self.tanh(y_import)

        y_codec_a6 = self.g_a6(y_codec)



        y_imp = similarities_repeated + y_tanh # 相似度* important map

        y = y_codec_a6 * y_imp

        z = self.h_a(y)

        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        params = self.h_s(z_hat)

        s = 4  # scaling factor between z and y
        kernel_size = 5  # context prediction kernel size
        padding = (kernel_size - 1) // 2

        y_height = z_hat.size(2) * s
        y_width = z_hat.size(3) * s

        y_hat = F.pad(y, (padding, padding, padding, padding))

        # pylint: disable=protected-access
        cdf = self.gaussian_conditional._quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional._cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional._offset.reshape(-1).int().tolist()
        # pylint: enable=protected-access
        # print(cdf, cdf_lengths, offsets)

        y_strings = []
        for i in range(y.size(0)):
            encoder = BufferedRansEncoder()
            # Warning, this is slow...
            # TODO: profile the calls to the bindings...
            symbols_list = []
            indexes_list = []
            for h in range(y_height):
                for w in range(y_width):
                    y_crop = y_hat[
                             i: i + 1, :, h: h + kernel_size, w: w + kernel_size
                             ]
                    ctx_p = self.context_prediction(y_crop)
                    # 1x1 conv for the entropy parameters prediction network, so
                    # we only keep the elements in the "center"
                    p = params[i: i + 1, :, h: h + 1, w: w + 1]
                    gaussian_params = self.entropy_parameters(
                        torch.cat((p, ctx_p[i: i + 1, :, 2: 3, 2: 3]), dim=1)
                    )
                    scales_hat, means_hat = gaussian_params.chunk(2, 1)

                    indexes = self.gaussian_conditional.build_indexes(scales_hat)
                    y_q = torch.round(y_crop - means_hat)
                    y_hat[i, :, h + padding, w + padding] = (y_q + means_hat)[
                                                            i, :, padding, padding
                                                            ]

                    symbols_list.extend(y_q[i, :, padding, padding].int().tolist())
                    indexes_list.extend(indexes[i, :].squeeze().int().tolist())

            encoder.encode_with_indexes(
                symbols_list, indexes_list, cdf, cdf_lengths, offsets
            )

            string = encoder.flush()
            y_strings.append(string)

        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}

    def decompress(self, strings, shape, flag):
    # def decompress(self, strings, shape):
        flag = np.nonzero(flag)
        assert isinstance(strings, list) and len(strings) == 2
        # FIXME: we don't respect the default entropy coder and directly call the
        # range ANS decoder

        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        params = self.h_s(z_hat)

        s = 4  # scaling factor between z and y
        kernel_size = 5  # context prediction kernel size
        padding = (kernel_size - 1) // 2

        y_height = z_hat.size(2) * s
        y_width = z_hat.size(3) * s

        # initialize y_hat to zeros, and pad it so we can directly work with
        # sub-tensors of size (N, C, kernel size, kernel_size)
        y_hat = torch.zeros(
            (z_hat.size(0), 192, y_height + 2 * padding, y_width + 2 * padding),
            device=z_hat.device,
        )
        decoder = RansDecoder()

        # pylint: disable=protected-access
        cdf = self.gaussian_conditional._quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional._cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional._offset.reshape(-1).int().tolist()

        # Warning: this is slow due to the auto-regressive nature of the
        # decoding... See more recent publication where they use an
        # auto-regressive module on chunks of channels for faster decoding...
        for i, y_string in enumerate(strings[0]):
            decoder.set_stream(y_string)

            for h in range(y_height):
                for w in range(y_width):
                    # only perform the 5x5 convolution on a cropped tensor
                    # centered in (h, w)
                    y_crop = y_hat[
                             i: i + 1, :, h: h + kernel_size, w: w + kernel_size
                             ]
                    ctx_p = self.context_prediction(y_crop)
                    # 1x1 conv for the entropy parameters prediction network, so
                    # we only keep the elements in the "center"
                    p = params[i: i + 1, :, h: h + 1, w: w + 1]
                    gaussian_params = self.entropy_parameters(
                        torch.cat((p, ctx_p[i: i + 1, :, 2: 3, 2: 3]), dim=1)
                    )
                    scales_hat, means_hat = gaussian_params.chunk(2, 1)

                    indexes = self.gaussian_conditional.build_indexes(scales_hat)
                    rv = decoder.decode_stream(
                        indexes[i, flag].squeeze().int().tolist(),
                        # indexes[i, :].squeeze().int().tolist(),
                        cdf,
                        cdf_lengths,
                        offsets,
                    )
                    # rv = torch.Tensor(rv).reshape(1, -1, 1, 1)
                    rv = torch.Tensor(rv).reshape(1, -1, 1, 1)
                    tmp = torch.zeros((1, 192, 1, 1))
                    tmp[:, flag, ...] = rv
                    rv = self.gaussian_conditional._dequantize(tmp, means_hat)
                    # rv = self.gaussian_conditional._dequantize(rv, means_hat)

                    y_hat[
                    i,
                    :,
                    h + padding: h + padding + 1,
                    w + padding: w + padding + 1,
                    ] = rv


        y_hat = y_hat[:, :, padding:-padding, padding:-padding]
        # pylint: enable=protected-access

        x_hat = self.g_s(y_hat).clamp_(0, 1)
        return {"x_hat": x_hat,}

    def decompress_1(self, strings, shape):
        assert isinstance(strings, list) and len(strings) == 2
        # FIXME: we don't respect the default entropy coder and directly call the
        # range ANS decoder

        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        params = self.h_s(z_hat)

        s = 4  # scaling factor between z and y
        kernel_size = 5  # context prediction kernel size
        padding = (kernel_size - 1) // 2

        y_height = z_hat.size(2) * s
        y_width = z_hat.size(3) * s

        # initialize y_hat to zeros, and pad it so we can directly work with
        # sub-tensors of size (N, C, kernel size, kernel_size)
        y_hat = torch.zeros(
            (z_hat.size(0), 192, y_height + 2 * padding, y_width + 2 * padding),
            device=z_hat.device,
        )
        decoder = RansDecoder()

        # pylint: disable=protected-access
        cdf = self.gaussian_conditional._quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional._cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional._offset.reshape(-1).int().tolist()

        # Warning: this is slow due to the auto-regressive nature of the
        # decoding... See more recent publication where they use an
        # auto-regressive module on chunks of channels for faster decoding...
        for i, y_string in enumerate(strings[0]):
            decoder.set_stream(y_string)

            for h in range(y_height):
                for w in range(y_width):
                    # only perform the 5x5 convolution on a cropped tensor
                    # centered in (h, w)
                    y_crop = y_hat[
                             i: i + 1, :, h: h + kernel_size, w: w + kernel_size
                             ]
                    ctx_p = self.context_prediction(y_crop)
                    # 1x1 conv for the entropy parameters prediction network, so
                    # we only keep the elements in the "center"
                    p = params[i: i + 1, :, h: h + 1, w: w + 1]
                    gaussian_params = self.entropy_parameters(
                        torch.cat((p, ctx_p[i: i + 1, :, 2: 3, 2: 3]), dim=1)
                    )
                    scales_hat, means_hat = gaussian_params.chunk(2, 1)

                    indexes = self.gaussian_conditional.build_indexes(scales_hat)

                    rv = decoder.decode_stream(
                        indexes[i, :].squeeze().int().tolist(),
                        cdf,
                        cdf_lengths,
                        offsets,
                    )
                    rv = torch.Tensor(rv).reshape(1, -1, 1, 1)

                    rv = self.gaussian_conditional._dequantize(rv, means_hat)

                    y_hat[
                    i,
                    :,
                    h + padding: h + padding + 1,
                    w + padding: w + padding + 1,
                    ] = rv
        y_hat = y_hat[:, :, padding:-padding, padding:-padding]
        # pylint: enable=protected-access

        x_hat = self.g_s(y_hat).clamp_(0, 1)
        return {"x_hat": x_hat}
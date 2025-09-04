# coding=utf-8
# Copyright 2023 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Core code of VCT: the temporal entropy model."""

from __future__ import annotations

from collections.abc import Sequence
import itertools
from typing import NamedTuple, Optional
from feature_patch import extract_patches, compute_padding
from compressai.entropy_models import GaussianConditional
# from absl import logging
import numpy as np
from compressai.ops.ops import quantize_ste
# import auxiliary_layers
# import bottlenecks
# import metric_collection
# import patcher
import transformer_layers
import torch.nn as nn
import torch
import compressai


_LATENT_NORM_FAC: float = 35.0


def _unbatch(t, dims):
    """Reshapes first dimension, i.e. (b, ...) becomes (b', *dims, ...)."""
    b_in, *other_dims = t.shape
    b_out = b_in // np.prod(dims)
    return t.reshape(b_out, *dims, *other_dims)


class VCTEntropyModel(nn.Module):
    """Temporal Entropy model."""

    def __init__(self, num_channels, context_len=2, window_size_enc=8, window_size_dec=4,
        num_layers_encoder_sep=3, num_layers_encoder_joint=2, num_layers_decoder=5,
        d_model=768, num_head=16, mlp_expansion=4, drop_out_enc=0.0, drop_out_dec=0.0):
        """Initializes model.

        Args:
          num_channels: Number of channels of the latents, i.e., symbols per token.
          context_len: How many previous latents to expect.
          window_size_enc: Window size in encoder.
          window_size_dec: Window size in decoder.
          num_layers_encoder_sep: Sepearte layer count.
          num_layers_encoder_joint: Joint layer count.
          num_layers_decoder: Number of decoder layers.
          d_model: Feature dimensionality inside the model.
          num_head: Number of attention heads per Multi-Head Attention layer.
          mlp_expansion: Expansion factor in feature dimensionality for each MLP.
          drop_out_enc: Sets the drop_out probability for various places in the
            encoder.
          drop_out_dec: Sets the drop_out probability for various places in the
            decoder.
        """
        if window_size_enc < window_size_dec:
            raise ValueError("Invalid config.")
        if num_channels < 0:
            raise ValueError("Invalid config.")
        super().__init__()
        self.num_channels = num_channels
        self.d_model = d_model
        self.gaussian_conditional = GaussianConditional(None)
        self.range_bottleneck = None

        self.context_len = context_len
        self.encoder_sep = transformer_layers.Transformer(
            is_decoder=False,
            drop_out=drop_out_enc,
            num_layers=num_layers_encoder_sep,
            d_model=d_model,
            num_head=num_head,
            mlp_expansion=mlp_expansion)
        self.encoder_joint = transformer_layers.Transformer(
            is_decoder=False,
            drop_out=drop_out_enc,
            num_layers=num_layers_encoder_joint,
            d_model=d_model,
            num_head=num_head,
            mlp_expansion=mlp_expansion)

        # self.patcher = patcher.Patcher(window_size_dec, "REFLECT")
        # self.learned_zero = auxiliary_layers.StartSym(num_channels)
        self.learned_zero = nn.Parameter(torch.randn((num_channels,)))

        self.window_size_enc = window_size_enc
        self.window_size_dec = window_size_dec
        self.enc_position_sep = nn.Parameter(torch.randn((window_size_enc**2, d_model),dtype=torch.float32))

        self.enc_position_joint = nn.Parameter(torch.randn((window_size_enc**2 * context_len, d_model),dtype=torch.float32))

        self.dec_position = nn.Parameter(torch.randn((window_size_dec**2, d_model),dtype=torch.float32))


        self.seq_len_dec = (window_size_dec ** 2)

        self.post_embedding_norm = nn.LayerNorm(d_model,eps=1e-5)
        self.encoder_embedding = nn.Linear(num_channels,d_model)
        self.decoder_embedding = nn.Linear(num_channels,d_model)

        self.decoder = transformer_layers.Transformer(
            is_decoder=True,
            num_layers=num_layers_decoder,
            d_model=d_model,
            seq_len=self.seq_len_dec,
            num_head=num_head,
            mlp_expansion=mlp_expansion,
            drop_out=drop_out_dec,
        )

        self.mean = nn.Sequential(
            nn.Linear(d_model,d_model),
            nn.LeakyReLU(),
            nn.Linear(d_model,d_model),
            nn.LeakyReLU(),
            nn.Linear(d_model, num_channels),
        )

        self.scale = nn.Sequential(
            nn.Linear(d_model,d_model),
            nn.LeakyReLU(),
            nn.Linear(d_model,d_model),
            nn.LeakyReLU(),
            nn.Linear(d_model, num_channels),
        )


    def process_previous_latent_q(self, previous_latent_quantized):
        """Processes the previous via the encoder, see baseclass docstring.

        This can be used if previous latents go through an expensive transform
        before being fed to the entropy model, and will be stored in the `processed`
        field of the `PreviousLatent` tuple.

        The output of this function applied to all quantized latents should
        be fed to `__call__`. Note that this is used to improve efficiency,
        as it avoids calling expensive processing of previous latents at
        each time step. For an example, see beachcomber/models.py.

        Args:
          previous_latent_quantized: The previous latent to process.
          training: Whether we are training.

        Returns:
          Processed previous latent as a `PreviousLatent` tuple.
        """
        # (b', seq_len, num_channels)
        latent_patches, _ = self.patcher(previous_latent_quantized, self.window_size_enc)
        patches = latent_patches / _LATENT_NORM_FAC

        # (b', seq_len, d_model)
        patches = self.encoder_embedding(patches)
        patches = self.post_embedding_norm(patches)
        # patches = self.enc_position_sep(patches)
        patches = self.enc_position_sep + patches
        patches = self.encoder_sep(patches,None)

        return {"quantized":previous_latent_quantized, "processed":patches}

    def _embed_latent_q_patched(self, latent_q_patched):
        """Embeds current latent for decoder."""
        # (b', seq_len, num_channels)
        latent_q_patched = latent_q_patched / _LATENT_NORM_FAC
        # (b', seq_len, d_model)
        latent_q_patched = self.decoder_embedding(latent_q_patched)
        latent_q_patched = self.post_embedding_norm(latent_q_patched)
        return self.dec_position+latent_q_patched

    def _get_transformer_output(self, encoded_patched, latent_q_patched):
        """Calculates transformer distribution prediction."""
        if encoded_patched.shape[-1] != self.d_model:
          raise ValueError(f"Context must have final dim {self.d_model}, "
                           f"got shape={encoded_patched.shape}. "
                           "Did you run `process_previous_latent_q`?")

        # latent_q_patched_shifted = self.learned_zero(latent_q_patched)
        latent_q_patched_shifted = torch.cat([latent_q_patched[...,:-1,:], self.learned_zero * torch.ones((latent_q_patched.shape[0], 1, latent_q_patched.shape[-1]))],dim=-2)
        latent_q_patched_emb_shifted = self._embed_latent_q_patched(
            latent_q_patched_shifted)
        del latent_q_patched  # Should not use after this line.

        # encoded_patched = self.enc_position_joint(encoded_patched)
        encoded_patched = self.enc_position_joint + encoded_patched
        encoded_patched = self.encoder_joint(encoded_patched, None)

        dec_output = self.decoder(
            latent=latent_q_patched_emb_shifted,
            enc_output=encoded_patched)

        mean = self.mean(dec_output)
        scale = self.scale(dec_output)

        return mean, scale, dec_output

    def _get_encoded_seqs(self, previous_latents, latent_shape):
        encoded_seqs = [p["processed"] for p in previous_latents]
        if len(encoded_seqs) < self.context_len:
            if self.context_len == 2:
                return [encoded_seqs[0], encoded_seqs[0]]
            if self.context_len == 3:
                if len(encoded_seqs) == 1:
                    return [encoded_seqs[0], encoded_seqs[0], encoded_seqs[0]]
                else:
                    assert len(encoded_seqs) == 2  # Sanity.
                    return [encoded_seqs[0], encoded_seqs[0], encoded_seqs[1]]
            raise ValueError(f"Unsupported: {self.context_len}")
        return encoded_seqs

    def patcher(self, t, patch_size):
        # t_padded, n_h, n_w = self._pad(t, patch_size)
        b, c, h, w = t.shape
        pad, unpad = compute_padding(h, w, min_div=self.window_size_dec)
        patches = extract_patches(t, patch_size, self.window_size_dec, pad=pad)

        # patches = extract_patches(t_padded, patch_size, self.stride)
        # `extract_patches` returns (b, n_h, n_w, seq_len * d), we reshape this
        # to (..., seq_len, d).
        b, n_hp, n_wp, _ = patches.shape
        patches = patches.reshape(b * n_hp * n_wp, patch_size ** 2, _//patch_size ** 2)
        return patches,(n_hp, n_wp)

    def unpatch(self, t, n_h, n_w,stride, crop):
        """Goes back to (b, h, w, d)."""
        _, seq_len, d = t.shape
        assert seq_len == stride ** 2
        t = t.reshape(-1, n_h, n_w, stride, stride, d).permute(0,1,3,2,4,5)
        t = t.reshape(-1, n_h * stride, n_w * stride, d).permute(0,3,1,2)
        if crop:
            h, w = crop
            return t[:, :h, :w, :]
        else:
            return t

    def forward(self, latent_unquantized, previous_latents):

        b, _, h, w = latent_unquantized.shape
        encoded_seqs = self._get_encoded_seqs(previous_latents, (h, w))
        b_enc, _, d_enc = encoded_seqs[0].shape
        if d_enc != self.d_model:
            raise ValueError(encoded_seqs[0].shape)
        # latent_q = tfc.round_st(latent_unquantized)
        latent_q = quantize_ste(latent_unquantized)
        latent_q_patched, (n_h, n_w) = self.patcher(latent_q, self.window_size_dec)

        # (b * n_h * n_w, self.window_size_dec ** 2, d)
        b_dec, _, d_dec = latent_q_patched.shape
        if d_dec != self.num_channels:
            raise ValueError(latent_q_patched.shape)
        if b_dec != b_enc:
            raise ValueError(
                f"Expected matching batch dimes, got {b_enc} != {b_dec}!")

        mean, scale, dec_output = self._get_transformer_output(
            # Fuse all in the sequence dimension.
            encoded_patched=torch.cat(encoded_seqs,dim=-2),
            latent_q_patched=latent_q_patched)
        assert mean.shape == latent_q_patched.shape
        decoder_features = self.unpatch(dec_output, n_h, n_w,self.window_size_dec, crop=(h, w)) # t[:, :h, :w, :]

        # Each tensor here is (b', seq_len, num_channels).
        # latent_unquantized_patched, _ = self.patcher(latent_unquantized, self.window_size_dec)
        scale = self.unpatch(scale, n_h, n_w, self.window_size_dec, crop=(h, w))  # t[:, :h, :w, :]
        mean = self.unpatch(mean, n_h, n_w, self.window_size_dec, crop=(h, w))  # t[:, :h, :w, :]

        output, latent_likelihoods = self.gaussian_conditional(latent_unquantized,scales=scale,means=mean)
        # (b, h, w, num_channels).
        # output = self.unpatch(output, n_h, n_w,self.window_size_dec, crop=(h, w))

        # (b,)
        # bits_per_batch = torch.sum(_unbatch(latent_likelihoods, (n_h, n_w)), (1, 2, 3, 4))
        return {'perturbed_latent':output,'likelihoods':latent_likelihoods,'features':decoder_features}


    def _get_mean_scale_jitted(self, *, encoded_patched, latent_q_patched, true_batch_size,):
        """Jitted version of `_get_transformer_output`, for `validate_causal`."""
        mean, scale, dec_output = self._get_transformer_output(
            encoded_patched=encoded_patched,
            latent_q_patched=latent_q_patched,
            training=False
            )
        return mean, scale, dec_output

    def _encode(self, latent_patched, encoded):
        bytestrings = []
        current_inp = np.full_like(latent_patched, fill_value=100.)
        autoreg_means = np.full_like(latent_patched, fill_value=100.)
        autoreg_scales = np.full_like(latent_patched, fill_value=100.)
        dec_output_shape = (*latent_patched.shape[:-1], self.d_model)
        dec_output = np.full(dec_output_shape, fill_value=100., dtype=np.float32)
        prev_mean = None
        prev_scale = None

        # We add 0 to code the very last symbol, it will become 0 - 1 == -1.
        for i in itertools.chain(range(self.seq_len_dec), [0]):
            # On the very first pass, we have no `prev_mean`, since we will feed
            # the zero symbol first to get an initial distribution.
            if prev_mean is not None:
                latent_i = latent_patched[:, i - 1, :]
                quantized_i, bytestring = self.range_bottleneck.compress(
                    latent_i, prev_mean, prev_scale)
                assert bytestring.shape == ()  # pylint: disable=g-explicit-bool-comparison
                bytestrings.append(bytestring)
                current_inp[:, i - 1, :] = quantized_i
                if i == 0:
                    break
        mean_i, scale_i, dec_output_i = self._get_mean_scale_jitted(
              encoded_patched=encoded,
              latent_q_patched=current_inp,
              true_batch_size=1)
        prev_mean = autoreg_means[:, i, :] = mean_i[:, i, :]
        prev_scale = autoreg_scales[:, i, :] = scale_i[:, i, :]
        dec_output[:, i, :] = dec_output_i[:, i, :]

        return bytestrings, autoreg_means, autoreg_scales, dec_output

    def _decode(self, bytestrings, encoded, shape,
              encode_means, encode_scales):
        h, w, c = shape
        fake_patched, (n_h, n_w) = self.patcher(
            torch.zeros((1, h, w, c)), self.window_size_dec)
        current_inp = np.full_like(fake_patched, fill_value=10.)
        prev_mean = None
        prev_scale = None
        for i in itertools.chain(range(self.seq_len_dec), [0]):
            if prev_mean is not None:
                decompressed_i = self.range_bottleneck.decompress(
                bytestrings.pop(0), prev_mean, prev_scale)
            current_inp[:, i - 1, :] = decompressed_i
            if i == 0:
                break
            mean_i, scale_i, _ = self._get_mean_scale_jitted(
              encoded_patched=encoded,
              latent_q_patched=current_inp,
              true_batch_size=1)
            # NOTE: We use the means from encoding, and log errors. Non-determinism
            # makes some outputs blow up ever so rarely (once every 100 or so
            # symbols).
            target_mean = encode_means[:, i, :]
            target_scale = encode_scales[:, i, :]
            actual_mean = mean_i[:, i, :]
            actual_scale = scale_i[:, i, :]

            error_mean = torch.sum(torch.abs(actual_mean - target_mean))
            error_scale = torch.sum(torch.abs(actual_scale - target_scale))


            prev_mean = target_mean
            prev_scale = target_scale
        assert not bytestrings
        return self.unpatch(current_inp, n_h, n_w, crop=(h, w))

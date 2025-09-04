# coding=utf-8

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

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


def window_partition(features, window_size, pad):
  # b, h, w, c = features.shape
  features = F.pad(features, pad, mode="reflect")
  assert features is not None  # pytype
  b, c, h, w = features.shape
  features = features.reshape((b, c, h // window_size, window_size, w // window_size, window_size)).permute(0,2,4,3,5,1)
  features = features.reshape((b, h // window_size, w // window_size, window_size ** 2, c))
  return features


def unwindow(features, window_size, unpad=0):
  b, nh, nw, _, c = features.shape
  features = features.reshape(b, nh, nw, window_size, window_size, c).transpose(2,3)
  # features = tf.einsum("bhwijc->bhiwjc", features)
  b, nh, _, nw, _, c = features.shape
  features = features.reshape(b, nh * window_size, nw * window_size, c).permute(0,3,1,2)
  features = F.pad(features, unpad)
  return features


def extract_patches_conv2d(image,
                           size,
                           stride = 1):
  channels = image.shape[1]
  kernel = torch.eye((size * size * channels)).reshape((channels * size * size,channels,size, size))
  return F.conv2d(image,weight=kernel,stride=stride,padding=(size-1)//2).permute(0,2,3,1)

def extract_patches_nonoverlapping(
    features,
    window_size,
    pad = None,
):
  """Wrapper around `window_partition` that returns same shape as other."""
  # Go from [B, nH, nW, wSize*wSize, c] to [b, n_H, n_W, size*size*c]
  patches = window_partition(features, window_size, pad=pad)
  _, n_h, n_w, seq_len, d = patches.shape
  return patches.reshape(-1, n_h, n_w, seq_len * d)


def extract_patches(
    image,
    size,
    stride = 1,
    pad = None
):
  if size == stride:
    # This function is reshape + transpose based and is always the fastest, but
    # of course only works if size == stride.
    return extract_patches_nonoverlapping(image, size, pad=pad)
  return extract_patches_conv2d(image, size, stride) #可能有问题


if __name__ == '__main__':

    image = torch.randn((2, 4, 16, 14))
    b, c, h, w = image.shape
    pad, unpad = compute_padding(h, w, min_div=4)
    tmp = extract_patches_nonoverlapping(image,4,pad=pad)

    # patches = window_partition(image, 4, pad=pad)
    # unpatched = unwindow(patches, 4, unpad=unpad)
    # if (image!=unpatched).sum()>0:
    #   print('error')
    tmp = extract_patches_conv2d(image,4)

    # image = tf.random.normal((2, 14, 14, 4))
    # patches = extract_patches.window_partition(image, 4, pad=True)
    # unpatched = extract_patches.unwindow(patches, 4, unpad=(14, 14))
    # self.assertAllEqual(image, unpatched)
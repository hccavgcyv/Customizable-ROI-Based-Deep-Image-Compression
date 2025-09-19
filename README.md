# Customizable-ROI-Based-Deep-Image-Compression
Pytorch Implementation of "Customizable-ROI-Based-Deep-Image-Compression" [arXiv](https://arxiv.org/pdf/2507.00373), TCSVT,2025


# Pretrained Models
Below, we provide the model weights to replicate the results of our paper ($sigma; =0.01).
| Lambda | Channels | Link                  |
|--------|--------|-----------------------|
| 0.0035 | 128    | [BaiduDisk](https://pan.baidu.com/s/19SlRJZiczF-BJhGGKzNIlQ)  |
| 0.0013 | 128    | [BaiduDisk](https://pan.baidu.com/s/1R1-UYFk9496Bsc-D1yxQEA)  |
| 0.025 | 192    | [BaiduDisk](https://pan.baidu.com/s/1vrkolgIImEB7OFhgr5BV3A)  |
| 0.0483 | 192    | [BaiduDisk](https://pan.baidu.com/s/12KJnd2xVw8WC0OS2eJQbgg)  |
| 0.0932 | 192    | [BaiduDisk](https://pan.baidu.com/s/1Hd_qN9thxxPbiDb-lMvadw)  |

code: jooy

# Training
Run the script:

`python train_tic.py -d /path/to/dataset/COCO/ --cuda`

# Testing
The testing code is provided in `eval_kodak.py`. 

By default, it supports ROI localization using textual input, which fully corresponds to the experimental setup and details described in our paper.

If you would like to evaluate the method using Ground-Truth masks (GT masks) instead, please modify the following section in `eval_kodak.py` with your GT-mask loading implementation:

```python
img_feat = net_lseg.forward(x_padded)
img_feat_norm = torch.nn.functional.normalize(img_feat, dim=1)
    
prompt = clip.tokenize(labels).cuda()
text_feat = net_lseg.clip_pretrained.encode_text(prompt)
text_feat_norm = torch.nn.functional.normalize(text_feat, dim=1)

cosine_similarity = torch.nn.CosineSimilarity(dim=1)
similarity = cosine_similarity(
    img_feat_norm, text_feat_norm.unsqueeze(-1).unsqueeze(-1)
)
similarity = similarity.unsqueeze(0)

```



# Acknowledgement
The framework is based on [TIC](https://github.com/lumingzzz/TIC), the lseg part of the code and checkpoint comes from [lseg-minimal](https://github.com/krrish94/lseg-minimal).

# Citation  
If you find this work useful for your research, please cite:  

```bibtex
@article{jin2025customizable,
  title={Customizable ROI-Based Deep Image Compression},
  author={Jin, Jian and Xia, Fanxin and Ding, Feng and Zhang, Xinfeng and Liu, Meiqin and Zhao, Yao and Lin, Weisi and Meng, Lili},
  journal={IEEE Transactions on Circuits and Systems for Video Technology},
  year={2025},
  publisher={IEEE}
}


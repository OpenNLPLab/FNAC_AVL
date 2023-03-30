# FNAC_AVL

Official Pytorch implementation of FNAC. 

Our paper is accpeted to CVPR 2023:

[**Learning Audio-Visual Source Localization via False Negative Aware Contrastive Learning**](https://arxiv.org/abs/2303.11302)

## Environment

Setup  environment:

```
pip install -r requirements.txt
```

## Data preparation
###  Flickr-SoundNet

Data can be downloaded from [Learning to localize sound sources](https://github.com/ardasnck/learning_to_localize_sound_source)

###  VGG-Sound Source

Data can be downloaded from [Localizing Visual Sounds the Hard Way](https://github.com/hche11/Localizing-Visual-Sounds-the-Hard-Way)

###  VGG-SS Unheard & Heard Test Data 

Data can be downloaded from [Unheard](https://github.com/stoneMo/EZ-VSL/blob/main/metadata/vggss_unheard_test.csv) and [Heard](https://github.com/stoneMo/EZ-VSL/blob/main/metadata/vggss_heard_test.csv)

## Training and Evaluation

For training FNAC on Flickr:
set the dataset path, train set ('flickr_10k' or 'flickr_144k'), and experiment name accordingly and run:
```
bash train_flickr.sh
```
For evaluation:
set the dataset path, test set, and experiment name accordingly and run:
```
bash test_flickr.sh
```

You can follow the same implementation to train and test on VGGSS and heard&unheard


## Weights
coming soon

## Todo


## Citation

If you find our work useful, please cite our paper:
```
@article{sun2023learning,
  title={Learning Audio-Visual Source Localization via False Negative Aware Contrastive Learning},
  author={Sun, Weixuan and Zhang, Jiayi and Wang, Jianyuan and Liu, Zheyuan and Zhong, Yiran and Feng, Tianpeng and Guo, Yandong and Zhang, Yanhao and Barnes, Nick},
  journal={arXiv preprint arXiv:2303.11302},
  year={2023}
}
```

## Acknowledgement
We thank [EZ-VSL](https://github.com/stoneMo/EZ-VSL) for their great codebase.

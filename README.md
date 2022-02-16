# Medical Matting: A New Perspective on Medical Segmentation with Uncertainty, [arxiv](https://arxiv.org/abs/2106.09887)

This is a PyTorch implementation of our paper. We introduce matting as a soft segmentation method and a new perspective to deal with and represent uncertain regions into medical scenes. 


## Reference
```
@inproceedings{wang2021medical,
  title={Medical matting: a new perspective on medical segmentation with uncertainty},
  author={Wang, Lin and Ju, Lie and Zhang, Donghao and Wang, Xin and He, Wanji and Huang, Yelin and Yang, Zhiwen and Yao, Xuan and Zhao, Xin and Ye, Xiufen and others},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={573--583},
  year={2021},
  organization={Springer}
}
```
A jounal extension version can be achieved at [arxiv](https://arxiv.org/abs/2106.09887v3).

## 1. Requirements

torch>=1.8.0; torchvision>=0.9.0; matplotlib; numpy; opencv_python; pandas; Pillow; PyYAML; scikit_image; scikit_learn; scipy; skimage; tensorboardX; tqdm; pickle;

data prepare: unpack the zip files in the dataset folder first.

Directory structure in this repo:
```
│MedicalMatting/
│   config.py
│   evaluate.py
│   params_brain.yaml
│   params_isic.yaml
│   params_lidc.yaml
│   README.md
│   train.py
+---dataloader
│       data_loader.py
│       data_spliter.py
│       transform.py
│       utils.py
+---dataset
│       brain_growth_alpha.pkl
│       isic_attributes.pkl
│       lidc_attributes.pkl
+---model
│   │   loss_functions.py
│   │   loss_strategy.py
│   │   medical_matting.py
│   │   utils.py
│   +---matting_network
│   │       cbam.py
│   │       matting_net.py
│   │       resnet_block.py
│   +---metrics
│   │       compute_connectivity_error.py
│   │       compute_gradient_loss.py
│   │       compute_mse_loss.py
│   │       compute_sad_loss.py
│   │       dice_accuracy.py
│   │       generalised_energy_distance.py
│   │       utils.py
│   +---probabilistic_unet
│   │       axis_aligned_conv_gaussian.py
│   │       encoder.py
│   │       fcomb.py
│   │       prob_unet.py
│   \---unet
│           unet.py
│           unet_blocks.py
+---models
\---utils
        logger.py
        utils.py
```

## 2. Train

- LIDC-IDRI
```bash
CUDA_VISIBLE_DEVICES=0 python train.py --config /path/to/params_lidc.yaml
```

- ISIC
```bash
CUDA_VISIBLE_DEVICES=0 python train.py --config /path/to/params_isic.yaml
```

- Brain-growth
```bash
CUDA_VISIBLE_DEVICES=0 python train.py --config /path/to/params_brain.yaml
```

## 3. Evaluation
```
CUDA_VISIBLE_DEVICES=0. python evaluate.py --config /path/to/params_**task.yaml \
  --save_path /path/to/your/model/dir
```

## Acknowledgements
The following code is referenced in this repo.
- A PyTorch implementation of [Probablistic UNet](https://github.com/stefanknegt/Probabilistic-Unet-Pytorch).
- [CBAM](https://github.com/Jongchan/attention-module)
- We reimplement the evaluation metrics in Matlab [DIM_evaluate_code](https://sites.google.com/view/deepimagematting) with PyTorch, more details please refer to [DIM_evaluation_code_python](https://github.com/wangsssky/DIM_evaluation_code_python).

Datasets:

The datasets in this paper were constructed based on the LIDC-IDRI, ISIC, and Brain-growth dataset, and the rights to the images used are owned by the original datasets. Please refer to the requirements of the original datasets for any use of the original images.
- [LIDC-IDRI](https://wiki.cancerimagingarchive.net/display/Public/LIDC-IDRI) 
    - The authors acknowledge the National Cancer Institute and the Foundation for the National Institutes of Health, and their critical role in the creation of the free publicly available LIDC/IDRI Database used in this study.
    - The LIDC-IDRI dataset used in this paper was obtained based on further annotation of these [patches](https://github.com/stefanknegt/Probabilistic-Unet-Pytorch).
- [ISIC](https://www.isic-archive.com/#!/topWithHeader/wideContentTop/main)
- [Brain-growth](https://qubiq21.grand-challenge.org/QUBIQ2021/)

## LICENSE

The codes of this repo is under the GPL license. For commercial use, please contact with the authors.

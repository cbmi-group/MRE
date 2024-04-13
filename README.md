# Accurate Segmentation for Intracellular Organelle Networks based on Low-level features and Primary Structure Invariance

*Jiaxing Huang, Yaoru Luo, Yuanhao Guo, Wenjing Li, Zichen Wang, Guole Liu and Ge Yang*

J. Huang, Y. Luo and  Y. Guo contributed equally to this work. 

Architecture of the Multi-resolution Encoder (MRE). 

<img src="./figures/architecture.png" style="width: 80%; height: auto;">

Contact: *Jiaxing Huang* huangjiaxing2021@ia.ac.cn.

If you have any questions, please feel free to contact us!

## Dataset

To develop deep learning segmentation models, we construct two custom image datasets: ER and MITO (openly accessible at IEEE DataPort (https://dx.doi.org/10.21227/t2he-zn97) ), for the ER network and the mitochondrial network, respectively. To test the generalization capability of our deep learning models, we also use two public datasets of the retinal blood vessel network: DRIVE and STARE.

### Preparing Dataset

Please generate ".txt" files for train data and test data separately. In the ".txt" file, each line consists of the path of image and mask corresponding to the image.

For example:

```
/datasets/directory/train_mito.txt
	└── .../MITO/train/images/20_h384_w384.tif .../MITO/train/masks/20_h384_w384.tif
/datasets/directory/test_mito.txt
	└── .../MITO/test/images/20_h384_w384.tif .../MITO/test/masks/20_h384_w384.tif
```

## Setup

Setting up for this project.

### Installing dependencies

To install all the dependencies, please run the following:

```
pip install -r requirements.txt or conda install --yes --file requirements.txt
```

## Running Models with MRE

### Training

In this project, we used nine basic models UNet, UNet++, nnUNet, HRNet, SAM, TransUNet, MedT, UT-Net, and PVT (tiny) as baseline. We apply our Multi-resolution Encoder in U-Net and PVT (tiny) in the experimental phase to valid the effectiveness of our method. To facilitate the training of the U-Net&MRE and PVT&MRE, we provide the training files.

Below lines will run the training code with default setting in the *train_MRE.py* file.

```
Set up
 # the model type
 model_choice = ['PVT_MRE', 'UNet_MRE']
 # the dataset type
 dataset_list = ['er', 'retina', 'mito', 'stare']
 # the encoder type
 encoder_choice = ['unet','pvt_tiny', 'pvt_small', 'pvt_base', 'pvt_large']
 # the loss type
 loss_choice = ['single', 'up_sampling', 'multi_layer', 'hierarchical_fusing']
 txt_choice = ['train_drive.txt', 'train_mito.txt',  'train_er.txt','train_stare.txt']
Run 
 python train_MRE.py
```

### Inference

In order to obtain segmentation results and evaluate model's performance under different thresholds, you can run the following line: 

```
Set up
 model_choice = ['PVT_MRE', 'UNet_MRE']
 encoder_choice = ['unet','pvt_tiny', 'pvt_small', 'pvt_base', 'pvt_large']
 loss_choice = ['single', 'up_sampling', 'multi_layer', 'hierarchical_fusing']
 dataset_list = ['er', 'retina', 'mito', 'stare']
 txt_choice = ['test_drive.txt', 'test_mito.txt', 'test_er.txt', 'test_stare.txt']
Run 
 python inference.py
```
## Results
This video showcases the results of our algorithm specifically tailored for the visualization of the endoplasmic reticulum (ER) network within cellular microscopy footage. Each frame of the video has been meticulously processed by the algorithm to segment and highlight the ER, allowing viewers to observe the network's intricate details in real-time. Notably, the algorithm distinguishes and accentuates the ER's tubules and nodes, offering a clear view of the cellular structure. This visualization not only serves as a vivid demonstration of the algorithm's capabilities in capturing dynamic cellular processes but also provides an essential resource for researchers to analyze the ER's morphology and function with unprecedented clarity and precision.

<video id="video" controls="" preload="none" poster="封面">
      <source id="mp4" src="./figures/seg_results.mp4" type="video/mp4">
</videos>

## Citation

If you find MRE useful in your research, please consider citing:

```
TODO
```

## Acknowledgements

This work was supported in part by the National Natural Science Foundation of China (grant 91954201 under the major research program “Organellar interactome for cellular homeostasis” and grant 31971289) and the Strategic Priority Research Program of the Chinese Academy of Sciences (grant XDB37040402). (*Corresponding author: Ge Yang*). 


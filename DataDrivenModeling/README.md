# PeRCNN
Physics-embedded recurrent convolutional neural network

Paper link: [[ArXiv](https://arxiv.org/pdf/2106.04781.pdf)] (We will update the final version later...)

By [Chengping Rao](https://scholar.google.com/citations?user=29DpfrEAAAAJ&hl=en), [Pu Ren](https://scholar.google.com/citations?user=7FxlSHEAAAAJ&hl=en), [Yang Liu](https://coe.northeastern.edu/people/liu-yang/), [Hao Sun](https://gsai.ruc.edu.cn/addons/teacher/index/info.html?user_id=0&ruccode=20210163&ln=en)


## Highlights
- Propose a physics-embedded recurrent-convolutional neural network (PeRCNN), which forcibly embeds the physics structure to facilitate learning for data-driven modeling of nonlinear systems
- The physics-embedding mechanism guarantees the model to rigorously obey the given physics based on our prior knowledge
- Present the recurrent π-Block to achieve nonlinear approximation via element-wise product among the feature maps
- Design the spatial information learned by either convolutional or predefined finite-differencebased filters
- Model the temporal evolution with forward Euler time marching scheme


### Training and extrapolation results
We show the reconstruction and extrapolation performance of our PeCRNN on 2D Gray-Scott equation below:
<p align="center">
  <img src="https://user-images.githubusercontent.com/55661641/141688143-b7fc69b0-13a7-476a-ae1a-b10c8378858d.gif" width="1024">
</p>


## Datasets
Due to the file size limit, we attach the google drive [[link](https://drive.google.com/drive/u/3/folders/1uZd7_OtbWuBJ0j-guJ0Ddva2V7nQ2VTM)] to download the datasets.


## Models
- PeRCNN model is provided under folders for each dataset
- misc/2d_burgers_ablation contains (part of) models for ablation study
- misc/xx_baselines contains baselines (ConvLSTM, DHPM, ResNet)


## Requirements
- pytorch>=1.6 is recommended
- plotly is needed to plot isosurface for 3D case
- TF 1.0 is required for DHPM


## Citation
Please consider citing us if you find our research helpful :D
```
@article{rao2021embedding,
  title={Embedding Physics to Learn Spatiotemporal Dynamics from Sparse Data},
  author={Rao, Chengping and Sun, Hao and Liu, Yang},
  journal={arXiv preprint arXiv:2106.04781},
  year={2021}
}
```





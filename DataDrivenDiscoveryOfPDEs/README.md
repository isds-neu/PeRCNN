## Data-driven discovery

PeRCNN for discovery of PDEs from scarce and noisy measurement data

## Prerequisite

Please download the data using the link provided in the ``data/readme4data.txt`` file. 


## Methodology overview

![](https://github.com/Raocp/Discover-PDE-with-Noisy-Scarce-Data/blob/main/Gallery/Slide2.JPG)

> Three stages of governing equation discovery process


![](https://github.com/Raocp/Discover-PDE-with-Noisy-Scarce-Data/blob/main/Gallery/Slide3.JPG)

> Two types of physics-encoded recurrent network (a. partial physics known; c. physics completely known.)

### Stage-1: data reconstruction

In Stage-1, we use a physics-encoded recurrent network to reconstruct the high-fidelity data. This step uses the same routine of https://github.com/Raocp/PeRCNN. 

### Stage-2: sparse regression

The sparse regression for two equation of u (`PDE_FIND_u.py` ), v (`PDE_FIND_v.py` ) is performed separately. We recommend you to run the sparse regression via IPython or Jupyter Notebook.

### Stage-3: coefficient finetuning

Based on the result from Stage-2, we perform coefficient finetuning using a physics-based recurrent network (i.e., the recurrent block mimics finite difference discretization of a governing PDE). Note that the finetuning is performed on the orginal sparse data (please refer the paper for explanation). 

## Tips

Set `restart=True` (it is a bad arg name...) when invoking train() function to read from checkpoint and continue training. 

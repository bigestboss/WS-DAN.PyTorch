# WS-DAN.PyTorch
A PyTorch implementation of WS-DAN (Hu et al., ["See Better Before Looking Closer: Weakly Supervised Data Augmentation
Network for Fine-Grained Visual Classification"](https://arxiv.org/abs/1901.09891v2), arXiv:1901.09891)


## Attention Cropping and Attention Dropping
![Fig1](./images/Fig1.png)

The framework introduce an attention based method for extracting more detailed features and more object's parts by Attention Cropping and Attention Dropping, see Fig 1. 

## Training Process and  Testing Process 
![Fig2a](./images/Fig2a.png)

![Fig2b](./images/Fig2b.png)

## Bilinear Attention Pooling (BAP)

![Fig3](./images/Fig3.png)

## Usage
1. ``` git clone``` this repo.
2. Prepare image data and rewrite ```dataset.py``` for your CustomDataset.
3. ```$ nohup python3 train_wsdan.py -j <num_workers> -b <batch_size> --sd <save_ckpt_directory> (etc.) 1>log.txt 2>&1 &``` (see ```train_wsdan.py``` for more training options)
4. ```$ tail -f log.txt``` for logging information.


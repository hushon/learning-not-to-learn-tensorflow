# learning-not-to-learn-tensorflow

This repo is a TensorFlow implementation of the following paper presented at CVPR 2019 -- [*Learning Not to Learn: Training Deep Neural Networks with Biased Data*](https://arxiv.org/abs/1812.10352)  
Please note that this is an unofficial repo and may not fully reproduce the original results. If you are looking for the official PyTorch source from the authors, redirect to [github.com/feidfoe/learning-not-to-learn](https://github.com/feidfoe/learning-not-to-learn)

## Introduction

![learning-not-to-learn-figure](./figure1.png)

This code demonstrates unlearning of bias from a classification model, particularly under a modified version of MNIST dataset named the Colored-MNIST. Here, the training set is artifically injected with bias such that class categories have strong correlation to color, meanwhile the test set is not contaminated with such bias. In this setting, colors work as false discriminative signal against baseline training methods, thus leading to low test accuracy of the classifier.  
The authors propose a novel training method such that the classifier model learns from the training data but *unlearns* from bias. The key ideas are: adoption of bias prediction model, and a regularizing loss function based on mutual information between feature embedding and bias. The bias predictor is trained to predict bias label from features extracted from the classifier. The classifer is trained adversarially against the bias predictor so that extracted feature lose correlation to bias.

## Setup

- Python 3
- TensorFlow 2
- Pillow

## Download Dataset

![Colored-MNIST](./colored-mnist-example.png)

[Download Colored-MNIST dataset](https://drive.google.com/file/d/1NSv4RCSHjcHois3dXjYw_PaLIoVlLgXu/view?usp=sharing)  
Please refer to the paper for more information on the dataset.  

## Train model

``` bash
python main.py --phase=train\
               --max_epoch=100\
               --batch_size=128\
               --lr=0.001\
               --loss_lambda=0.01
```

To train model by baseline method, add `--train_baseline` argument.  
Once you begin, you can launch TensorBoard on `./logs/` directory to monitor training.

``` bash
tensorboard --logdir=./logs/
```

## Test model

``` bash
python main.py --phase=test\
               --batch_size=128
```

## Reference

- Byungju Kim, Hyunwoo Kim, Kyungsu Kim, Sungjin Kim, Junmo Kim, "Learning Not to Learn: Training Deep Neural Networks with Biased Data", in CVPR, 2019

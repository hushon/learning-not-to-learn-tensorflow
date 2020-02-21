# learning-not-to-learn-tensorflow

This repo is an unofficial TensorFlow implementation of -- [*Learning Not to Learn: 
Training Deep Neural Networks with Biased Data (CVPR 2019)*](https://arxiv.org/abs/1812.10352)  
If you are looking for the official PyTorch source from the authors, redirect to [github.com/feidfoe/learning-not-to-learn](https://github.com/feidfoe/learning-not-to-learn)

## Introduction

![learning-not-to-learn-figure](./figure1.png)

This code demonstrates unlearning of bias from an classification model, particularly under a modified version of MNIST dataset named the Colored-MNIST. Here, the training set is artifically injected with bias such that a class category has strong correlation to color, meanwhile the test set is not contaminated such bias. This effects as a false discriminative signal against baseline training methods, thus result in low test accuracy.  
The authors suggest a novel training method such that a classifier model learns from the training set but *unlearns* from bias. The key ideas are: adoption of a bias prediction model, and a novel regularizing loss function based on mutual information between feature embedding and bias.

## Setup
- Python 3
- TensorFlow 1.14
- tqdm

## Download Dataset
![Colored-MNIST](./colored-mnist-example.png)

[Download Colored-MNIST dataset](https://drive.google.com/file/d/1NSv4RCSHjcHois3dXjYw_PaLIoVlLgXu/view?usp=sharing)  
Please refer to the paper for more information on the dataset.  

## Train model
```
python main.py --phase=train\
               --max_epoch=100\
               --batch_size=200\
               --lr=1e-3\
               --loss_lambda=0.1\
               --loss_mu=0.1
```
Once you begin, you can launch TensorBoard on `./logs/` directory to monitor training.
```
tensorboard --logdir=./logs/
```

## Test model
```
python main.py --phase=test\
               --batch_size=200\
```

## Reference

- Byungju Kim, Hyunwoo Kim, Kyungsu Kim, Sungjin Kim, Junmo Kim, "Learning Not to Learn: Training Deep Neural Networks with Biased Data", in CVPR, 2019
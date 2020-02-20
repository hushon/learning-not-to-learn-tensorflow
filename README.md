# learning-not-to-learn-tensorflow

This repo is an unofficial TensorFlow implementation of -- [*Learning Not to Learn: 
Training Deep Neural Networks with Biased Data (CVPR 2019)*](https://arxiv.org/abs/1812.10352)  
If you are looking for the official PyTorch source from the authors, you may redirect to [github.com/feidfoe/learning-not-to-learn](https://github.com/feidfoe/learning-not-to-learn)

## Introduction

![learning-not-to-learn-figure](./figure1.png)

This code demonstrates unlearning of bias from an MNIST classification model.  
Here, the training set is artifically injected with bias such that the class category has direct correlation against its color, meanwhile the test set does not has such bias. This effectly serves as false signals during training of the classifier, thus result in low test accuracy with baseline methods.  
The authors suggest a novel training method so that the model learns from the training set but unlearns from the bias. The key ideas are: adoption of an additional bias prediction model, and a novel regularizing loss function based on mutual information between feature embeddings and bias.

## Setup
- Python 3
- TensorFlow 1.14
- tqdm

## Download Dataset
![Colored-MNIST](./colored-mnist-example.png)

[Colored-MNIST dataset](https://drive.google.com/file/d/11K-GmFD5cg3_KTtyBRkj9VBEnHl-hx_Q/view?usp=sharing)  
Find more about the dataset in the authors' paper.  

## Train model
```
python main.py --phase=train\
               --max_epoch=100\
               --batch_size=200\
               --lr=1e-3\
               --loss_lambda=0.01\
               --loss_mu=1.0
```
Once you begin, `./logs/` directory is generated so you can launch TensorBoard to monitor training.

## Test model
```
python main.py --phase=test\
               --batch_size=200\
```

## Reference

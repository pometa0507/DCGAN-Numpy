# DCDGAN-Numpy
DCGAN in Numpy

# License
This software is released under the MIT License, see license.txt.

# Introduction

DCGAN (Deep Convolutional Generative Adversarial Network) is one of the generative models.
<br>
This code is implemented DCGAN using Numpy without using a deep learning framework.


# Python Library

To run the source code, the following software is required.

* Python 3.x
* NumPy
* Matplotlib
* tqdm
* (cupy)


# Usage

```console
python dcgan_numpy.py
```



# DCGANの概要

DCGAN(Deep Convolutional GAN)は生成モデルのひとつです。


DCGANをディープラーニングのフレームワークを使わずにNumpyを使って実装しました。

![DCGAN_Overview](https://github.com/pometa0507/DCDGAN-Numpy/blob/master/images/appendix/DCGAN-Overview.png)



# ネットワーク構成

## Generator Network

![Generator-Network](https://github.com/pometa0507/DCDGAN-Numpy/blob/master/images/appendix/Generator-Network.png)


## Generator Network

![Discriminator-Network](https://github.com/pometa0507/DCDGAN-Numpy/blob/master/images/appendix/Discriminator-Network.png)


<br>

# 学習

DVGANの学習は、GeneratorとDiscriminatorでそれぞれ学習することになります。
<br>


## Generator Training

![Generator-Training](https://github.com/pometa0507/DCDGAN-Numpy/blob/master/images/appendix/Generator-Training.png)

<br>


## Generator Training

![Discriminator-Training](https://github.com/pometa0507/DCDGAN-Numpy/blob/master/images/appendix/Discriminator-Training.png)

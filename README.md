# DCDGAN-Numpy
DCGAN in Numpy

# License
This software is released under the MIT License, see license.txt.

# Introduction

DCGAN (Deep Convolutional Generative Adversarial Networks) is one of the generative models.
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



# GANの概要

In Japanese from the following.

GANの概要について説明します。
<br>
GANはGenerator(生成器)とDiscriminator(識別器)という2つのネットワークに分かれています。

Generatorはノイズデータを入力として、本物そっくりの偽物画像を生成します。
一方、Discriminatorは画像を入力として、その画像が偽物か本物かを識別します。

GeneratorとDiscriminatorは敵対的な関係にあることから、Generative Adversarial Nets(GAN)と呼ばれています。



# DCGANの概要
DCGAN(Deep Convolutional GAN)はGANのひとつの手法です。

GANでは、ネットワークに全結合層を使っていますが、DCGANでは、畳み込み層または転置畳み込み層を使用してます。

![DCGAN_Overview](https://github.com/pometa0507/DCDGAN-Numpy/blob/master/images/appendix/DCGAN-Overview.png)



# ネットワーク構成

DCGANをディープラーニングのフレームワークを使わずにNumpyを使って実装しています。


## Generator Network

Generatorの入力は100次元の潜在変数zです。
この潜在変数から、偽物の画像(3chanel,64x64)を生成します。

画像データのピクセル値の範囲は、-1～+1となっています。
そのため、出力層の活性化関数にTanhを使用して-1～+1を出力しています。

![Generator-Network](https://github.com/pometa0507/DCDGAN-Numpy/blob/master/images/appendix/Generator-Network.png)


## Discriminator Network

Discriminatorの入力は画像(3chanel,64x64)です。
入力された画像が、偽物(0)か本物(1)かを識別します。

出力は0か1なので2クラス分類となります。
そのため、出力層の活性化関数にSigmoidを使用して0～1を出力しています。

![Discriminator-Network](https://github.com/pometa0507/DCDGAN-Numpy/blob/master/images/appendix/Discriminator-Network.png)


<br>

# 学習

DVGANの学習は、GeneratorとDiscriminatorでそれぞれ学習することになります。
<br>


## Generator Training

![Generator-Training](https://github.com/pometa0507/DCDGAN-Numpy/blob/master/images/appendix/Generator-Training.png)

<br>


## Discriminator Training

![Discriminator-Training](https://github.com/pometa0507/DCDGAN-Numpy/blob/master/images/appendix/Discriminator-Training.png)

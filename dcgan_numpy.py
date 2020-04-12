# -*- coding: utf-8 -*-

"""# モジュールインポート"""
import os
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import OrderedDict
import numpy as npy
try:
    import cupy as np
    print("cupy mode")
except ImportError:
    import numpy as np
    print("numpy mode")

from common import *


"""# NetworkModel

## Generator
"""

class Generator:
    def __init__(self, latent_dim=100):

        self.latent_dim = latent_dim   #潜在空間

        # レイヤの生成
        self.layers = OrderedDict()
        self.layers['ConvTrans1'] = ConvTrans2D(in_channels=latent_dim, out_channels=256, kernel_size=4, stride=1, pad=0, bias=False)
        self.layers['BN1'] = BatchNormalization(np.ones(256),np.zeros(256))
        self.layers['Relu1'] = Relu()

        self.layers['ConvTrans2'] = ConvTrans2D(in_channels=256, out_channels=128, kernel_size=4, stride=2, pad=1, bias=False)
        self.layers['BN2'] = BatchNormalization(np.ones(128),np.zeros(128))
        self.layers['Relu2'] = Relu()
        
        self.layers['ConvTrans3'] = ConvTrans2D(in_channels=128, out_channels=64, kernel_size=4, stride=2, pad=1, bias=False)
        self.layers['BN3'] = BatchNormalization(np.ones(64),np.zeros(64))
        self.layers['Relu3'] = Relu()
        
        self.layers['ConvTrans4'] = ConvTrans2D(in_channels=64, out_channels=32, kernel_size=4, stride=2, pad=1, bias=False)
        self.layers['BN4'] = BatchNormalization(np.ones(32),np.zeros(32))
        self.layers['Relu4'] = Relu()
        
        self.layers['ConvTrans5'] = ConvTrans2D(in_channels=32, out_channels=3, kernel_size=4, stride=2, pad=1, bias=False)
        self.layers['Tanh'] = Tanh()
        
        self.dout = None
        
        # 重みの初期化
        self.params = {}
        self.params['W1'] = self.layers['ConvTrans1'].W
        self.params['gamma1'] = self.layers['BN1'].gamma
        self.params['beta1'] = self.layers['BN1'].beta
        self.params['W2'] = self.layers['ConvTrans2'].W
        self.params['gamma2'] = self.layers['BN2'].gamma
        self.params['beta2'] = self.layers['BN2'].beta
        self.params['W3'] = self.layers['ConvTrans3'].W
        self.params['gamma3'] = self.layers['BN3'].gamma
        self.params['beta3'] = self.layers['BN3'].beta
        self.params['W4'] = self.layers['ConvTrans4'].W
        self.params['gamma4'] = self.layers['BN4'].gamma
        self.params['beta4'] = self.layers['BN4'].beta
        self.params['W5'] = self.layers['ConvTrans5'].W

    def forward(self, x, train_flag=True):
        x = self.layers['ConvTrans1'].forward(x)
        x = self.layers['BN1'].forward(x, train_flag)
        x = self.layers['Relu1'].forward(x)
        x = self.layers['ConvTrans2'].forward(x)
        x = self.layers['BN2'].forward(x, train_flag)
        x = self.layers['Relu2'].forward(x)
        x = self.layers['ConvTrans3'].forward(x)
        x = self.layers['BN3'].forward(x, train_flag)
        x = self.layers['Relu3'].forward(x)
        x = self.layers['ConvTrans4'].forward(x)
        x = self.layers['BN4'].forward(x, train_flag)
        x = self.layers['Relu4'].forward(x)
        x = self.layers['ConvTrans5'].forward(x)
        x = self.layers['Tanh'].forward(x)
        return x

    def loss(self, z, t, skip_forward=False):
        pass
        
    def backward(self, dout_d):
        """勾配を求める（誤差逆伝播法）
        Parameters
        ----------
        dout_d : Discriminatorから逆伝播された勾配
        """

        # backward
        self.dout = dout_d           #Discriminatorからの勾配
        
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            self.dout = layer.backward(self.dout)
        
        # 設定
        grads = {}
        grads['W1']= self.layers['ConvTrans1'].dW
        grads['gamma1'], grads['beta1'] = self.layers['BN1'].dgamma, self.layers['BN1'].dbeta
        grads['W2']= self.layers['ConvTrans2'].dW
        grads['gamma2'], grads['beta2'] = self.layers['BN2'].dgamma, self.layers['BN2'].dbeta
        grads['W3']= self.layers['ConvTrans3'].dW
        grads['gamma3'], grads['beta3'] = self.layers['BN3'].dgamma, self.layers['BN3'].dbeta
        grads['W4']= self.layers['ConvTrans4'].dW
        grads['gamma4'], grads['beta4'] = self.layers['BN4'].dgamma, self.layers['BN4'].dbeta
        grads['W5']= self.layers['ConvTrans5'].dW

        return grads

"""## Discriminator"""

class Discriminator:
    def __init__(self):

        # レイヤの生成
        self.layers = OrderedDict()
        self.layers['Conv1'] = Convolution(3,32,4,4,stride=2, pad=1, bias=False)
        self.layers['LRelu1'] = LRelu(0.2)

        self.layers['Conv2'] = Convolution(32,64,4,4,stride=2, pad=1, bias=False)
        self.layers['BN1'] = BatchNormalization(np.ones(64),np.zeros(64))
        self.layers['LRelu2'] = LRelu(0.2)
        
        self.layers['Conv3'] = Convolution(64,128,4,4,stride=2, pad=1, bias=False)
        self.layers['BN2'] = BatchNormalization(np.ones(128),np.zeros(128))
        self.layers['LRelu3'] = LRelu(0.2)
        
        self.layers['Conv4'] = Convolution(128,256,4,4,stride=2, pad=1, bias=False)
        self.layers['BN3'] = BatchNormalization(np.ones(256),np.zeros(256))
        self.layers['LRelu4'] = LRelu(0.2)
        
        self.layers['Conv5'] = Convolution(256,1,4,4,stride=1, pad=0, bias=False)

        #Binary Cross Entropy With Sigmoid
        self.last_layer = SigmoidWithLoss()
        
        self.dout = None
                
        # 重みの初期化
        self.params = {}
        self.params['W1'] = self.layers['Conv1'].W
        self.params['W2'] = self.layers['Conv2'].W
        self.params['gamma1'] = self.layers['BN1'].gamma
        self.params['beta1'] = self.layers['BN1'].beta
        self.params['W3'] = self.layers['Conv3'].W
        self.params['gamma2'] = self.layers['BN2'].gamma
        self.params['beta2'] = self.layers['BN2'].beta
        self.params['W4'] = self.layers['Conv4'].W
        self.params['gamma3'] = self.layers['BN3'].gamma
        self.params['beta3'] = self.layers['BN3'].beta
        self.params['W5'] = self.layers['Conv5'].W

    def forward(self, x, train_flag=True):
        x = self.layers['Conv1'].forward(x)
        x = self.layers['LRelu1'].forward(x)
        x = self.layers['Conv2'].forward(x)
        x = self.layers['BN1'].forward(x, train_flag)
        x = self.layers['LRelu2'].forward(x)
        x = self.layers['Conv3'].forward(x)
        x = self.layers['BN2'].forward(x, train_flag)
        x = self.layers['LRelu3'].forward(x)
        x = self.layers['Conv4'].forward(x)
        x = self.layers['BN3'].forward(x, train_flag)
        x = self.layers['LRelu4'].forward(x)
        x = self.layers['Conv5'].forward(x)       
        return x

    def loss(self, z, t):
        """損失関数を求める
        Parameters
        ----------
        z : 入力データ
        t : 教師ラベル
        """
        y = self.forward(z)
        return self.last_layer.forward(y, t)
        
    def backward(self, z, t):
        """勾配を求める（誤差逆伝播法）
        Parameters
        ----------
        z : 入力データ
        t : 教師ラベル
        """
        # forward
        self.loss_D = self.loss(z, t)

        # backward
        self.dout = 1
        self.dout = self.last_layer.backward(self.dout)   #勾配

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            self.dout = layer.backward(self.dout)
        
        # 設定
        grads = {}
        grads['W1']= self.layers['Conv1'].dW
        grads['W2']= self.layers['Conv2'].dW
        grads['gamma1'], grads['beta1'] = self.layers['BN1'].dgamma, self.layers['BN1'].dbeta
        grads['W3']= self.layers['Conv3'].dW
        grads['gamma2'], grads['beta2'] = self.layers['BN2'].dgamma, self.layers['BN2'].dbeta
        grads['W4']= self.layers['Conv4'].dW
        grads['gamma3'], grads['beta3'] = self.layers['BN3'].dgamma, self.layers['BN3'].dbeta
        grads['W5']= self.layers['Conv5'].dW

        return grads

"""## DCGAN"""

class DCGAN:
    """Convolutional GAN"""
    def __init__(self, latent_dim=100):
        #潜在変数の次元
        self.latent_dim = latent_dim
        
        #Model
        self.model_G = Generator(self.latent_dim)
        self.model_D = Discriminator()

        #Optimizer
        self.optimizer_G = Adam(lr=0.0002, beta1=0.5, beta2=0.999)
        self.optimizer_D = Adam(lr=0.0002, beta1=0.5, beta2=0.999)

    def train(self, x_train, savedir, epochs, batch_size=64):
        """学習"""

        print("epochs={} , batch_size={}".format(epochs,batch_size))

        loss_G_array = np.array([])
        loss_D_real_array = np.array([])
        loss_D_fake_array = np.array([])
        
        #訓練データ　シャッフル
        np.random.shuffle(x_train)

        for ep in range(epochs):

            #イテレーション処理（tqdmでプログレスバーを表示）
            itr = np.arange(0, len(x_train), batch_size)
            with tqdm(itr) as pbar:
                for idx in pbar:
                    #tqdm プログレスバー表示
                    pbar.set_description("[Epoch %d]" % (ep))
                    
                    #訓練データ(batch)
                    x = x_train[idx:idx+batch_size]
                    
                    #実際のbatchサイズ
                    batch = x.shape[0]
                    
                    #教師データ 0 or 1
                    t_ones = np.ones((batch,1,1,1))
                    t_zeros = np.zeros((batch,1,1,1))
                    
                    ########################################
                    ####   Generator  backpropagation   ####
                    ########################################

                    z = np.random.randn(batch, self.latent_dim, 1, 1)        #潜在変数 z
                    fake_img = self.model_G.forward(z)                       #Generatorが生成したfake_img

                    self.model_D.backward(fake_img, t_ones)                  #Dは勾配のみ求める。このとき教師は1とする
                    Loss_G = self.model_D.loss_D                             #Dの損失関数(ログ用)

                    #Discriminatorの勾配 model_D.dout
                    dout_D = self.model_D.dout
                    grads_G = self.model_G.backward(dout_D)                  #Gの勾配
                    self.optimizer_G.update(self.model_G.params, grads_G)
                    

                    ########################################
                    #### Discriminator  backpropagation ####
                    ########################################

                    real_img = x                                             #Real_img

                    #それぞれの勾配を計算
                    grads_D_real = self.model_D.backward(real_img, t_ones)   #Real_imgからDの勾配を求める。このとき教師は1とする
                    Loss_D_real = self.model_D.loss_D                        #Dの損失関数(ログ用)
                    
                    grads_D_fake = self.model_D.backward(fake_img, t_zeros)  #Fake_imgからDの勾配を求める。このとき教師は0とする
                    Loss_D_fake = self.model_D.loss_D                        #Dの損失関数(ログ用)

                    #勾配を合算して最適化を実行
                    grads_D = {}
                    for key in grads_D_real.keys():       
                        grads_D[key] = grads_D_real[key] + grads_D_fake[key]

                    self.optimizer_D.update(self.model_D.params, grads_D)


                    #損失関数のログを格納
                    loss_G_array = np.concatenate((loss_G_array, np.atleast_1d(Loss_G)))
                    loss_D_real_array = np.concatenate((loss_D_real_array, np.atleast_1d(Loss_D_real)))
                    loss_D_fake_array = np.concatenate((loss_D_fake_array, np.atleast_1d(Loss_D_fake)))

                    #tqdm プログレスバー表示
                    pbar.set_postfix(OrderedDict(Loss_G=Loss_G, Loss_D_real=Loss_D_real, Loss_D_fake=Loss_D_fake))
                

            #10epochsごとに64枚(8*8)の画像を保存
            if ep % 10 == 0:
                z = np.random.randn(64, self.latent_dim, 1, 1)               #潜在変数 z
                fake_img = self.model_G.forward(z, train_flag=False)         #train_flagはFalse          
                savename = os.path.join(savedir, 'Generate_{}.png'.format(str(ep)))
                self.save_generate_imgs(fake_img, savename)
        
        history = [loss_G_array, loss_D_real_array, loss_D_fake_array]
        return history

    def save_generate_imgs(self, imgs, savename):        
        genarate_imgs = imgs.transpose(0,2,3,1) #N,H,W,C
        genarate_imgs = (genarate_imgs + 1) / 2            #0～1に正規化
        
        #cupy対応
        if np.__name__=='cupy':        #cupyの場合はnumpyのndarrayに変換
            genarate_imgs = np.asnumpy(genarate_imgs)

        #64枚をひとつのイメージに変換
        genarate_imgs = genarate_imgs.reshape(-1, 64, 3)   # 縦64枚に並べる
        for i in range(8):                                 # 縦8枚 横8枚に並べる
            if i == 0:
                img_tile = genarate_imgs[0:64*8]
            else:
                img_tile = npy.concatenate( [img_tile, genarate_imgs[64*8*i:64*8*(i+1)] ], axis=1 )

        plt.imshow(img_tile)
        plt.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False) #目盛非表示
        plt.savefig(savename)


if __name__ == '__main__':
    
    train_show_mode = False     #データセットの画像表示モード

    dirname = os.path.dirname(__file__)
    
    train_data_path = os.path.join(dirname, 'cats_datasets_npy.pickle')
    savedir = os.path.join(dirname, 'images/generate')

    """# 画像データセット読込 npy"""
    #画像データセット(Picle)読み込み
    f = open(train_data_path,'rb')
    x_train = pickle.load(f)
    
    #cupy対応
    if np.__name__=='cupy':  #numpy ⇒ cupy配列へ変換 
        x_train = np.asarray(x_train)
        
    #画像ピクセル値変換 0～255 -> -1.0～+1.0
    x_train = (x_train / 127.5) - 1

    
    """# データセットの画像表示"""
    if train_show_mode:
            train_imgs = x_train[:64].transpose(0,2,3,1) #N,H,W,C
            train_imgs = (train_imgs + 1) / 2            #0～1に正規化
            
            #64枚をひとつのイメージに変換
            train_imgs = train_imgs.reshape(-1, 64, 3)   #縦64枚に並べる
            for i in range(8):                           #縦8枚、横8枚に並べる
                if i == 0:
                    img_tile = train_imgs[0:64*8]
                else:
                    img_tile = npy.concatenate( [img_tile, train_imgs[64*8*i:64*8*(i+1)] ], axis=1 )
            
            #cupy対応
            if np.__name__=='cupy':                      #cupyの場合はnumpyのndarrayに変換
                img_tile = np.asnumpy(img_tile)
            
            fig = plt.figure(figsize=(10,10))
            plt.imshow(img_tile)
            plt.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False) #目盛非表示
            plt.show()
            

    """# 学習"""
    
    gan = DCGAN(latent_dim=100)
    
    print("DCGAN Training Start.")
    history = gan.train(x_train, savedir=savedir, batch_size=64, epochs=300)
    

# -*- coding: utf-8 -*-

"""# モジュールインポート"""
try:
    import cupy as np
    #print("cupy")
except ImportError:
    import numpy as np
    #print("numpy")


"""# Common

## Optimizer
"""

class Adam:
    """Adam (http://arxiv.org/abs/1412.6980v8)"""

    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.iter = 0
        self.m = None
        self.v = None
        
    def update(self, params, grads):
        if self.m is None:
            self.m, self.v = {}, {}
            for key, val in params.items():
                self.m[key] = np.zeros_like(val)
                self.v[key] = np.zeros_like(val)
        
        self.iter += 1
        lr_t  = self.lr * np.sqrt(1.0 - self.beta2**self.iter) / (1.0 - self.beta1**self.iter)         
        
        for key in params.keys():
            #self.m[key] = self.beta1*self.m[key] + (1-self.beta1)*grads[key]
            #self.v[key] = self.beta2*self.v[key] + (1-self.beta2)*(grads[key]**2)
            self.m[key] += (1 - self.beta1) * (grads[key] - self.m[key])
            self.v[key] += (1 - self.beta2) * (grads[key]**2 - self.v[key])
            
            params[key] -= lr_t * self.m[key] / (np.sqrt(self.v[key]) + 1e-7)
            
            #unbias_m += (1 - self.beta1) * (grads[key] - self.m[key]) # correct bias
            #unbisa_b += (1 - self.beta2) * (grads[key]*grads[key] - self.v[key]) # correct bias
            #params[key] += self.lr * unbias_m / (np.sqrt(unbisa_b) + 1e-7)

"""## layer"""

class Tanh:
    def __init__(self):
        self.out = None

    def forward(self, x):
        out = np.tanh(x)
        self.out = out
        return out
    
    def backward(self, dout):
        dx = dout * (1.0 - self.out**2)
        return dx

class Relu:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0
        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout
        return dx

class LRelu:
    """LeakyReLU"""
    def __init__(self, alpha=0.01):
        self.out = None
        self.alpha = alpha

    def forward(self, x):
        self.out = x.copy()
        return np.maximum(x, x*self.alpha)

    def backward(self, dout):  
        dx = np.where(self.out <= 0, self.alpha * dout, dout)
        return dx

class Sigmoid:
    def __init__(self):
        self.out = None

    def forward(self, x):
        out = sigmoid(x)
        self.out = out
        return out

    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out
        return dx

def sigmoid(x):
    return 1 / (1 + np.exp(-x))    

def binary_cross_entropy_error(y, t):
    """バイナリー交差エントロピー誤差"""
    #y.shape (N,C,H,W)
    delta = 1e-7
    return -np.mean(t*np.log(y + delta) + (1-t)*np.log(1-y + delta))

class SigmoidWithLoss:
    def __init__(self):
        self.loss = None
        self.y = None # sigmoidの出力
        self.t = None # 教師データ

    def forward(self, x, t):
        self.t = t
        self.y = sigmoid(x)
        self.loss = binary_cross_entropy_error(self.y, self.t)
        return self.loss

    def backward(self, dout=1):
        dx = dout * (self.y - self.t) / self.y.size
        return dx

def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    """
    Parameters
    ----------
    input_data : (データ数, チャンネル, 高さ, 幅)の4次元配列からなる入力データ
    filter_h : フィルターの高さ
    filter_w : フィルターの幅
    stride : ストライド
    pad : パディング
    Returns
    -------
    col : 2次元配列
    """
    N, C, H, W = input_data.shape
    out_h = (H + 2*pad - filter_h)//stride + 1
    out_w = (W + 2*pad - filter_w)//stride + 1

    img = np.pad(input_data, [(0,0), (0,0), (pad, pad), (pad, pad)], 'constant')
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))

    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N*out_h*out_w, -1)
    return col


def col2im(col, input_shape, filter_h, filter_w, stride=1, pad=0):
    """
    Parameters
    ----------
    col :
    input_shape : 入力データの形状（例：(10, 1, 28, 28)）
    filter_h :
    filter_w
    stride
    pad
    Returns
    -------
    """
    N, C, H, W = input_shape
    out_h = (H + 2*pad - filter_h)//stride + 1
    out_w = (W + 2*pad - filter_w)//stride + 1
    col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(0, 3, 4, 5, 1, 2)

    img = np.zeros((N, C, H + 2*pad + stride - 1, W + 2*pad + stride - 1))
    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]

    return img[:, :, pad:H + pad, pad:W + pad]

class Convolution:
    def __init__(self, input_channel, output_channel, kernel_h=5, kernel_w=5, stride=1, pad=0, bias=True):
        self.bias = bias

        k = 1/ (output_channel * kernel_h*kernel_w)
        limit = np.sqrt(k)
        self.W = np.random.rand(output_channel, input_channel, kernel_h, kernel_w) * 2*limit - limit
        if self.bias: self.b = np.zeros(output_channel)
        self.stride = stride
        self.pad = pad
        
        # 中間データ（backward時に使用）
        self.x = None   
        self.col = None
        self.col_W = None
        
        # 重み・バイアスパラメータの勾配
        self.dW = None
        self.db = None
        
        
    def forward(self, x):
        FN, C, FH, FW = self.W.shape
        N, C, H, W = x.shape
        out_h = 1 + int((H + 2*self.pad - FH) / self.stride)
        out_w = 1 + int((W + 2*self.pad - FW) / self.stride)

        col = im2col(x, FH, FW, self.stride, self.pad)
        col_W = self.W.reshape(FN, -1).T

        out = np.dot(col, col_W)
        if self.bias: out = out + self.b
        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)

        self.x = x
        self.col = col
        self.col_W = col_W

        return out

    def backward(self, dout):
        FN, C, FH, FW = self.W.shape
        dout = dout.transpose(0,2,3,1).reshape(-1, FN)

        if self.bias: self.db = np.sum(dout, axis=0)
        self.dW = np.dot(self.col.T, dout)
        self.dW = self.dW.transpose(1, 0).reshape(FN, C, FH, FW)

        dcol = np.dot(dout, self.col_W.T)
        dx = col2im(dcol, self.x.shape, FH, FW, self.stride, self.pad)

        return dx


class Pooling:
    def __init__(self, pool_h, pool_w, stride=1, pad=0):
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride
        self.pad = pad
        
        self.x = None
        self.arg_max = None

    def forward(self, x):
        N, C, H, W = x.shape
        out_h = int(1 + (H - self.pool_h) / self.stride)
        out_w = int(1 + (W - self.pool_w) / self.stride)

        col = im2col(x, self.pool_h, self.pool_w, self.stride, self.pad)
        col = col.reshape(-1, self.pool_h*self.pool_w)

        arg_max = np.argmax(col, axis=1)
        out = np.max(col, axis=1)
        out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)

        self.x = x
        self.arg_max = arg_max

        return out

    def backward(self, dout):
        dout = dout.transpose(0, 2, 3, 1)
        
        pool_size = self.pool_h * self.pool_w
        dmax = np.zeros((dout.size, pool_size))
        dmax[np.arange(self.arg_max.size), self.arg_max.flatten()] = dout.flatten()
        dmax = dmax.reshape(dout.shape + (pool_size,)) 
        
        dcol = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)
        dx = col2im(dcol, self.x.shape, self.pool_h, self.pool_w, self.stride, self.pad)
        
        return dx

class ConvTrans2D:
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, pad=0, out_pad=0, bias=True):
        
        #重みを乱数で初期化
        k = 1/ (in_channels * kernel_size**2)
        limit = np.sqrt(k)
        self.W = np.random.rand(in_channels, out_channels, kernel_size, kernel_size) * 2*limit - limit
        self.bias = bias
        
        if self.bias:
            self.b = np.random.rand(out_channels) * limit - limit
        
        self.stride = stride
        self.pad = pad
        self.out_pad = out_pad
        
        # 中間データ（backward時に使用）
        self.x = None
        
        # 重み・バイアスパラメータの勾配
        self.dW = None
        
        if self.bias:
            self.db = None

    def forward(self, x):
        
        self.x = x
        N, in_channels, H, W = self.x.shape
        in_channels, out_channels, FH, FW = self.W.shape

        #出力サイズ
        #out_shape = (W - 1)*stride - 2*padding + (FW - 1) + output_padding + 1

        out_h = (H - 1)*self.stride + (FH - 1) + self.out_pad + 1  #padはあとで処理。（paddingで削るまえのサイズ）
        out_w = (W - 1)*self.stride + (FW - 1) + self.out_pad + 1  #padはあとで処理。（paddingで削るまえのサイズ）

        out = np.zeros((N, out_channels, out_h, out_w))   # N, C, H, W

        W_tmp = self.W.reshape(in_channels, -1)           # in_channels, out_channels * FH * FW

        for i_y in range(H):
            out_y = i_y * self.stride
            out_y_max = out_y + FH

            for i_x in range(W):
                out_x = i_x * self.stride
                out_x_max = out_x + FW

                out[:, :, out_y:out_y_max, out_x:out_x_max] += np.dot(x[:,:,i_y,i_x], W_tmp).reshape(N, out_channels, FH, FW)
        
        #padがあれば、pad分をスライスで除外
        if self.pad >= 1 : out = out[:, :, self.pad : -self.pad, self.pad : -self.pad]
        
        if self.bias:
            out += self.b.reshape(1, -1, 1, 1)  #次元を合わせる　N, out_C, H, W

        return out

    def backward(self, dout):
        N, in_channels, H, W = self.x.shape
        in_channels, out_channels, FH, FW = self.W.shape
        
        #dout_shape =  N, out_C, H, W
        
        if self.bias:
            #dbはout_chanelごとにsumして求める
            dout_b = dout.transpose(0,2,3,1).reshape(-1, out_channels)  # shape = (N*H*W,out_C)
            self.db = np.sum(dout_b, axis=0)

        #dx, self.dWを初期化
        dx = np.zeros_like(self.x)
        self.dW = np.zeros_like(self.W)
        
        self.dW = self.dW.reshape(in_channels, -1)
                
        self.W_tmp = self.W.reshape(in_channels, -1)
        
        #padがあればdoutを拡張(padding)
        dout = np.pad(dout,[(0,0),(0,0),(self.pad,self.pad),(self.pad,self.pad)],"constant")
        
        for i_y in range(H):
            out_y = i_y * self.stride
            out_y_max = out_y + FH
            
            for i_x in range(W):
                out_x = i_x * self.stride
                out_x_max = out_x + FW

                x_tmp = self.x[:,:,i_y,i_x]

                dout_tmp = dout[:, :, out_y:out_y_max, out_x:out_x_max].reshape(dout.shape[0],-1)  # N, out_C * H * W

                self.dW += np.dot(x_tmp.T, dout_tmp)         #dWを足しこんでいく

                dx[:,:,i_y,i_x]  = np.dot(dout_tmp, self.W_tmp.T)
                
        self.dW = self.dW.reshape(in_channels, out_channels, FH, FW)
        
        return dx

class BatchNormalization:
    """
    http://arxiv.org/abs/1502.03167
    """
    def __init__(self, gamma, beta, momentum=0.9, eps=1e-5, running_mean=None, running_var=None):
        self.gamma = gamma
        self.beta = beta
        self.momentum = momentum
        self.eps = eps
        self.input_shape = None                     # Conv層の場合は4次元、全結合層の場合は2次元  

        # テスト時に使用する平均と分散
        self.running_mean = running_mean
        self.running_var = running_var  
        
        # backward時に使用する中間データ
        self.batch_size = None
        self.xc = None
        self.std = None
        self.dgamma = None
        self.dbeta = None

    def forward(self, x, train_flg=True):
        #self.input_dim = x.ndim
        self.input_shape = x.shape
        out = self.__forward(x, train_flg)
        return out.reshape(*self.input_shape)
            
    def __forward(self, x, train_flg):
        if self.running_mean is None:
            if x.ndim == 2:
                N, D = x.shape
                self.running_mean = np.zeros((1,D))
                self.running_var = np.zeros((1,D))
            
            elif x.ndim == 4:
                N, C, H, W = x.shape
                self.running_mean = np.zeros((1,C,1,1))
                self.running_var = np.zeros((1,C,1,1))

        if train_flg:
            if x.ndim == 2:
                mu = x.mean(axis=0)                                   #N,Dのうち、N方向で平均を算出
                xc = x - mu
                var = np.mean(xc**2, axis=0)
                std = np.sqrt(var + self.eps)
                xn = xc / std

            elif x.ndim == 4:
                mu = x.mean(axis=(0,2,3), keepdims=True)              #N,C,H,Wのうち、N.H.W方向で平均を算出
                xc = x - mu
                var = np.mean(xc**2, axis=(0,2,3), keepdims=True)
                std = np.sqrt(var + self.eps)
                xn = xc / std
                
            self.batch_size = self.input_shape[0]
            self.chanel_size = self.input_shape[1]  #kari
            self.xc = xc
            self.xn = xn
            self.std = std
            
            self.running_mean = self.momentum * self.running_mean + (1-self.momentum) * mu
            self.running_var = self.momentum * self.running_var + (1-self.momentum) * var

        else: #テスト時
            xc = x - self.running_mean
            xn = xc / ((np.sqrt(self.running_var + self.eps)))

        if x.ndim == 2:
            out = self.gamma.reshape(1,-1) * xn + self.beta.reshape(1,-1)
        elif x.ndim == 4:
            out = self.gamma.reshape(1,-1,1,1) * xn + self.beta.reshape(1,-1,1,1)
        
        return out

    def backward(self, dout):
        dx = self.__backward(dout)
        dx = dx.reshape(*self.input_shape)
        return dx

    def __backward(self, dout):
        if dout.ndim == 2:
            dbeta = dout.sum(axis=0, keepdims=True)
            dgamma = np.sum(self.xn * dout, axis=0, keepdims=True)
            dxn = self.gamma.reshape(1,-1) * dout
            dxc = dxn / self.std
            dstd = -np.sum((dxn * self.xc) / (self.std * self.std), axis=0)
            dvar = 0.5 * dstd / self.std
            dxc += (2.0 / self.batch_size) * self.xc * dvar
            dmu = np.sum(dxc, axis=0)
            dx = dxc - dmu / self.batch_size

        elif dout.ndim == 4:            # (N,C,H,W)
            dbeta = dout.sum(axis=(0,2,3))
            dgamma = np.sum(self.xn * dout, axis=(0,2,3))
            NHW = dout.shape[0]*dout.shape[2]*dout.shape[3]                           #N*H*Wのサイズ
            dx = self.gamma.reshape(1,-1,1,1) / self.std * ( dout - (dbeta.reshape(1,-1,1,1) + dgamma.reshape(1,-1,1,1) * self.xn) / NHW )

        self.dgamma = dgamma
        self.dbeta = dbeta  
        
        return dx

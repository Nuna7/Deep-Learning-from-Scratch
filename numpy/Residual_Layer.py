from Neural_Network import *

class Residual_Layer:
    def __init__(self,X, Y,kernel_size,no_of_filter,stride,pad,prev_filter,learning_rate=1e-4,transformation="Identity"):
        self.X = X
        self.Y = Y

        fan_in = prev_filter * kernel_size * kernel_size
        self.K1 = np.random.randn(kernel_size,kernel_size,prev_filter,no_of_filter)
        self.b1 = np.zeros((1,1,1,no_of_filter))

        self.K2 = np.random.randn(kernel_size,kernel_size,no_of_filter,no_of_filter)
        self.b2 = np.zeros((1,1,1,no_of_filter))
        
        self.stride = stride
        self.kernel_size = kernel_size
        self.no_of_filter = no_of_filter
        self.pad = pad
        self.learning_rate = learning_rate
        
        self.A_prev = np.array([])
        self.transformation = transformation

    def set_transformation(self, transformation):
        self.transformation = transformation
    
    def initialize(self,shape):
        self.m, self.n_H, self.n_W, self.prev_filter = shape

        self.new_n_H = int((self.n_H + 2*self.pad - self.kernel_size) / self.stride) + 1
        self.new_n_W = int((self.n_W + 2*self.pad - self.kernel_size) / self.stride) + 1

        self.new_n_H = int((self.new_n_H + 2*self.pad - self.kernel_size) / self.stride) + 1
        self.new_n_W = int((self.new_n_W + 2*self.pad - self.kernel_size) / self.stride) + 1

        if (self.new_n_H < self.n_H) and (self.new_n_W < self.n_W):
            self.transformation = "DownSampling"
            self.transform_stride = int((self.n_H + 2*self.pad - self.kernel_size) / (self.new_n_H - 1)) + 1
            
        elif (self.new_n_H > self.n_H) and (self.new_n_W > self.n_W):
            if self.transformation == "Identity":
                self.transformation_padding = (self.new_n_H - self.n_H) // 2
                assert self.transformation_padding % 1 == 0
            elif self.transformation == "UpSampling":
                self.transform_stride = (self.n_H + 2*self.pad - 1) / (self.new_n_H - 1)
        else:
            assert self.new_n_H == self.n_H and self.new_n_W == self.n_W and self.prev_filter == self.no_of_filter

        if self.transformation == "DownSampling" or self.transformation == "UpSampling":
            assert self.transform_stride % 1 == 0
            self.transform_stride = int(self.transform_stride)
            self.W = np.random.randn(1,1,self.prev_filter, self.no_of_filter)

        self.LW = np.random.randn(self.new_n_H * self.new_n_W * self.no_of_filter, 1)
        
    def transform_image(self, A_prev):
        output = np.zeros((self.m,self.new_n_H,self.new_n_W,self.no_of_filter))

        if self.transformation == "Identity":
            pad = self.transformation_padding
            A_prev = np.pad(A_prev, ((0,0),(pad,pad),(pad,pad),(0,0)) ,mode="constant",constant_values=(0,0))
            assert A_prev.shape == output.shape
            output = A_prev
        else:
            for i in range(self.m):
                single_img = A_prev[i, :, :, :]
                for j in range(self.new_n_H):
                    for k in range(self.new_n_W):
                        for l in range(self.no_of_filter):
                            start_h = j * self.transform_stride
                            end_h = start_h + 1
    
                            start_w = k * self.transform_stride
                            end_w = start_w + 1
    
                            output[i, j, k, l] = np.sum(single_img[start_h:end_h,
                                                                      start_w:end_w, :] * self.W[:, :, :, l])
        return output
    
    def convolve(self,A_prev,K,b):
        
        n_H = int((A_prev.shape[1] - self.kernel_size) / self.stride) + 1
        n_W = int((A_prev.shape[2] - self.kernel_size) / self.stride) + 1
        Z = np.zeros((self.m, n_H, n_W, self.no_of_filter))
        
        for img in range(self.m):
            single_img = A_prev[img,:,:,:]
            
            for height in range(n_H):
                start_h = height * self.stride
                end_h = start_h + self.kernel_size
                
                for width in range(n_W):
                    start_w = width * self.stride
                    end_w = start_w + self.kernel_size
                    
                    for channel in range(self.no_of_filter):
                        temp = single_img[start_h:end_h, start_w:end_w, :]
                        kern = K[:,:,:,channel]
                        bias = b[:,:,:,channel]

                        l_trans = np.multiply(temp, kern)
                        Z_temp = np.sum(l_trans) + bias

                        Z[img,height,width,channel] = Z_temp[0, 0, 0] 
        return Z
    
    def relu(self,Z):
        return np.maximum(Z,0)

    def relu_derivative(self,Z):
        return Z > 0 * 1
        
    def transform_image_backward(self,dZ,A_prev=None):
        if self.transformation == "Identity":
            pad = self.transformation_padding
            dZ = dZ[:,pad:-pad,pad:-pad,:]
            return dZ
            
        else:
            output = np.zeros((self.m, self.new_n_H, self.new_n_W, self.no_of_filter))
            self.dW = np.zeros(self.W.shape)
            dA_prev = np.zeros(A_prev.shape)
        
            for i in range(self.m):
                single_img = A_prev[i, :, :, :]
                for j in range(self.new_n_H):
                    for k in range(self.new_n_W):
                        for l in range(self.no_of_filter):
                            start_h = j * self.transform_stride
                            end_h = start_h + 1
                            start_w = k * self.transform_stride
                            end_w = start_w + 1
        
                            self.dW[:, :, :, l] += dZ[i, j, k, l] * single_img[start_h:end_h, start_w:end_w, :]
                            dA_prev[i, start_h:end_h, start_w:end_w, :] += dZ[i, j, k, l] * self.W[:, :, :, l]
                            
        return dA_prev
                            

    def backprop(self, dZ, A_prev, K, b):
        pad = self.pad
        dA_prev = np.zeros(A_prev[:, pad:-pad, pad:-pad, :].shape)
        dK = np.zeros(K.shape)
        db = np.zeros(b.shape)

        dA_prev_pad = np.zeros(A_prev.shape)

        for img in range(self.m):
            da_prev_pad = dA_prev_pad[img]
            a_prev_pad = A_prev[img]

            for height in range(self.n_H):
                start_h = height * self.stride
                end_h = start_h + self.kernel_size
                
                for width in range(self.n_W):
                    start_w = width * self.stride
                    end_w = start_w + self.kernel_size
                    
                    for filter_ in range(self.no_of_filter):
                        
                        a_prev = a_prev_pad[start_h:end_h, start_w:end_w, :]

                        da_prev_pad[start_h:end_h, start_w:end_w, :] += (
                                        dZ[img, height, width, filter_] * K[:, :, :, filter_])

                        dK[:, :, :, filter_] += (dZ[img, height, width, filter_] * a_prev)
                        db[:, :, :, filter_] += (dZ[img, height, width, filter_])

            dA_prev[img, :, :, :] = (da_prev_pad[pad:-pad, pad:-pad, :])

        return dA_prev, dK, db

    def BinaryCrossEntropy(self,y_true, y_pred):
        y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
        term_0 = (1-y_true) * np.log(1-y_pred + 1e-7)
        term_1 = y_true * np.log(y_pred + 1e-7)
        return -np.mean(term_0+term_1, axis=0)

    def forward(self,A_prev):
        if len(self.A_prev) == 0:
            self.initialize(A_prev.shape)

        self.A_prev = np.pad(A_prev,((0,0),(self.pad,self.pad),(self.pad,self.pad),(0,0)),
                                  mode="constant",constant_values=(0,0))

        Z1 = self.convolve(self.A_prev,self.K1,self.b1)
        A1 = self.relu(Z1)

        A1 = np.pad(A1,((0,0),(self.pad,self.pad),(self.pad,self.pad),(0,0)),
                                  mode="constant",constant_values=(0,0))
        
        Z2 = self.convolve(A1,self.K2,self.b2)

        residual = self.transform_image(A_prev)
        A2 = Z2 + residual 
        A3 = self.relu(A2)

        M = A3.reshape((A3.shape[0], -1))
        N = np.dot(M,self.LW)
        pred = expit(N)

        loss = self.BinaryCrossEntropy(self.Y, pred)

        print(loss)

        dZ = (self.Y - pred)
        dW = np.dot(dZ.T,N)
        self.LW -= 1e-4 * dW

        dZ = np.dot(dZ, self.LW.T)
        dZ = dZ.reshape(A3.shape)
        
        dZ = self.relu_derivative(A3) * dZ

        dskip = self.transform_image_backward(dZ) if self.transformation == "Identity" else self.transform_image_backward(dZ, self.A_prev)

        dZ, self.dK2, self.db2 = self.backprop(dZ, A1, self.K2, self.b2)

        dZ = self.relu_derivative(A1[:,self.pad:-self.pad,self.pad:-self.pad,:]) * dZ

        dZ, self.dK1, self.db1 = self.backprop(dZ, self.A_prev, self.K1, self.b1)
        
        self.update()
        
    def update(self):
        self.K1 -= self.learning_rate * self.dK1
        self.b1 -= self.learning_rate * self.db1
        self.K2 -= self.learning_rate * self.dK2
        self.b2 -= self.learning_rate * self.db2
        print("Kernel Updated")

def batching(X,Y,size_of_batch):
    len_ = X.shape[0]
    no_of_batch = int(len_/size_of_batch)
    
    idx = np.array(list(range(len_)))
    np.random.shuffle(idx)
    
    batches_X = []
    batches_Y = []
    
    for batch in range(no_of_batch):
        start = batch * size_of_batch
        end = start + size_of_batch
        batches_X.append(X[idx[start:end]])
        batches_Y.append(Y[idx[start:end]])
        
    remainder = len_ - (no_of_batch*size_of_batch)
        
    if len_ % size_of_batch != 0:
        batches_X.append(X[idx[-remainder:]])
        batches_Y.append(Y[idx[-remainder:]])
        
    return batches_X,batches_Y
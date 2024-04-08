from Neural_Network import *

class Convolutional_Layer:
    def __init__(self,kernel_size,no_of_filter,stride,pad,prev_filter,learning_rate=1e-4,
                 gradient_clip=True,clip_value = 3.0):

        fan_in = prev_filter * kernel_size * kernel_size
        fan_out = no_of_filter * kernel_size * kernel_size
        std_dev = np.sqrt(2 / (fan_in + fan_out))

        self.K = np.random.randn(kernel_size,kernel_size,prev_filter,no_of_filter) * std_dev
        self.b = np.zeros((1,1,1,no_of_filter))
        
        self.stride = stride
        self.kernel_size = kernel_size
        self.no_of_filter = no_of_filter
        self.pad = pad
        self.learning_rate = learning_rate

        self.gradient_clip = gradient_clip
        self.clip_value = clip_value
        
        self.A_prev = None
        
    def convolve(self,A_prev):    
        self.A_prev = np.pad(A_prev,((0,0),(self.pad,self.pad),(self.pad,self.pad),(0,0)),
                                  mode="constant",constant_values=(0,0))
            
        self.m, n_H, n_W, self.prev_filter = A_prev.shape
        
        self.n_H = int((n_H + 2*self.pad - self.kernel_size)/self.stride) + 1
        self.n_W = int((n_W + 2*self.pad - self.kernel_size)/self.stride) + 1
        
        Z = np.zeros((self.m,self.n_H,self.n_W,self.no_of_filter))
        
        for img in range(self.m):
            single_img = self.A_prev[img,:,:,:]
            for height in range(self.n_H): # Vertical
                start_h = height * self.stride
                end_h = start_h + self.kernel_size
                for width in range(self.n_W): # Horizontal
                    start_w = width * self.stride
                    end_w = start_w + self.kernel_size
                    for channel in range(self.no_of_filter):
                        
                        temp = single_img[start_h:end_h,start_w:end_w,:]
                        kern = self.K[:,:,:,channel]
                        bias = self.b[:,:,:,channel]
                        
                        l_trans = np.multiply(temp,kern)
                        Z_temp = np.sum(l_trans) + bias
                        
                        Z[img, height, width, channel] = (Z_temp[0, 0, 0])
        return Z
    
    def relu(self,Z):
        return np.maximum(Z,0)
    
    def update(self):
        self.K = self.K - self.learning_rate * self.dK
        self.b = self.b - self.learning_rate * self.db
    
    def backprop(self,dZ):
        pad = self.pad
        self.dA_prev = np.zeros(self.A_prev[:,pad:-pad,pad:-pad,:].shape)
        self.dK = np.zeros(self.K.shape)
        self.db = np.zeros(self.b.shape)
        
        dA_prev_pad = np.pad(self.dA_prev,((0,0),(pad,pad),(pad,pad),(0,0)),mode="constant",constant_values=(0,0))
        
        for img in range(self.m):
            da_prev_pad = dA_prev_pad[img]
            a_prev_pad = self.A_prev[img]
            
            for height in range(self.n_H):
                start_h = height * self.stride
                end_h = start_h + self.kernel_size
                for width in range(self.n_W):
                    start_w = width * self.stride
                    end_w = start_w + self.kernel_size
                    for filter_ in range(self.no_of_filter):
                        
                        a_prev = a_prev_pad[start_h:end_h,start_w:end_w,:]
                    
                        da_prev_pad[start_h:end_h,start_w:end_w,:] += dZ[img,height,width,filter_]*self.K[:,:,
                                                                                                          :,filter_]
                        
                        self.dK[:,:,:,filter_] += (dZ[img,height,width,filter_]*a_prev)
                        self.db[:,:,:,filter_] += (dZ[img,height,width,filter_])
                        
            self.dA_prev[img,:,:,:] = (da_prev_pad[pad:-pad,pad:-pad,:])

        if self.gradient_clip:
            np.clip(self.dK, -self.clip_value, self.clip_value, out=self.dK)
            np.clip(self.db, -self.clip_value, self.clip_value, out=self.db)
                
        return self.dA_prev

class MaxPooling:
    def __init__(self,pool_size,mode,stride):
        self.pool_size = pool_size
        self.stride = stride
        self.mode = mode
        self.first = True
        
    def Pool(self, A_prev):
        if self.first:
            self.A_prev = A_prev
            self.m, self.n_H, self.n_W, self.n_C = A_prev.shape
            self.first = False
    
        n_H = int((self.n_H - self.pool_size) / self.stride) + 1
        n_W = int((self.n_W - self.pool_size) / self.stride) + 1
    
        A = np.zeros((self.m, n_H, n_W, self.n_C))
    
        for img in range(self.m):
            single_img = self.A_prev[img]
            for height in range(n_H):
                start_h = height * self.stride
                end_h = start_h + self.pool_size
    
                for width in range(n_W):
                    start_w = width * self.stride
                    end_w = start_w + self.pool_size

                    for channel in range(self.n_C):
                        temp = single_img[start_h:end_h, start_w:end_w, channel]
                        if self.mode == "max":
                            A[img, height, width, channel] = np.max(temp)
                        elif self.mode == "average":
                            A[img, height, width, channel] = np.mean(temp)
                        else:
                            raise ValueError("Invalid Mode")
        return A
                
    def mask(self,z):
        return (z == np.max(z))
    
    def distribute_value(self,dz,shape):
        average = np.prod(shape)
        a = (dz/average) * np.ones(shape)
        return a
    
    def backprop(self,dZ):
        self.dZ_prev = np.zeros(self.A_prev.shape)

        n_H = int((self.n_H - self.pool_size) / self.stride) + 1
        n_W = int((self.n_W - self.pool_size) / self.stride) + 1
        
        for img in range(self.m):
            a_prev = self.A_prev[img]
            
            for height in range(n_H):
                start_h = height * self.stride
                end_h = start_h + self.pool_size
                
                for width in range(n_W):
                    start_w = width * self.stride
                    end_w = start_w + self.pool_size
                    
                    for channel in range(self.n_C):
                        
                        if self.mode == "max":
                            a_cur = a_prev[start_h:end_h, start_w:end_w, channel]
                            mask = self.mask(a_cur)
                            self.dZ_prev[img, start_h:end_h, start_w:end_w, channel] += (
                                mask * dZ[img, height, width, channel])

                        
                        elif self.mode == "average":
                            da = dZ[img, height, width, channel]
                            shape = (self.pool_size, self.pool_size)
                            self.dZ_prev[img,start_h:end_h,start_w:end_w,channel] += (
                                self.distribute_value(da,shape)
                            )
                            
                            
        assert(self.A_prev.shape == self.dZ_prev.shape)

        return self.dZ_prev

class Convolutional_NN(Convolutional_Layer,NeuralNetwork,MaxPooling):
    def __init__(self,X,Y,learning_rate,layer_arch,class_):
        self.X = X
        self.Y = Y
        self.learning_rate = learning_rate
        self.layer_arch = layer_arch
        self.num_of_layers = len(layer_arch)
        self.epsilon = 1e-6
        
        if class_ == "Binary":
            self.class_ = "Sigmoid"
            self.loss = "binary_crossentropy"
        elif class_ == "Categorical":
            self.class_ = "Softmax"
            self.loss = "categorical_crossentropy"
            
        self.no_conv = 0
        self.no_ff = 0
        self.no_nn = 0
        
        self.Layers = {}
        self.A = {}
        self.Z = {}
        
        self.m, n_H, n_W, cur_channel = self.X.shape

        n_pool = 0
        self.n_bn = 0
        for layer in self.layer_arch:
            
            if layer["Layer"] == "Convolution":
                kernel_size = layer["kernel_size"]
                no_of_filter = layer["no_of_filter"]
                stride = layer["stride"]
                pad = layer["pad"]
                self.Layers[f"Conv_{self.no_conv + 1}"] = Convolutional_Layer(kernel_size, no_of_filter, stride,pad,cur_channel,
                                                         learning_rate)

                cur_channel = no_of_filter
                n_H = int((n_H + 2*pad - kernel_size)/stride + 1)
                n_W = int((n_W + 2*pad - kernel_size)/stride + 1)
                
                self.no_conv += 1
                
            elif layer["Layer"] == "Pool":
                pool_size = layer["pool_size"]
                mode = layer["mode"]
                stride = layer["stride"]
                self.Layers[f"MaxPool_{n_pool + 1}"] = MaxPooling(pool_size, mode, stride)

                n_H = int((n_H - pool_size)/stride + 1)
                n_W = int((n_W - pool_size)/stride + 1)
                n_pool += 1

            elif layer["Layer"] == "BatchNorm":
                self.Layers[f"BatchNorm_{self.n_bn + 1}"] = BatchNorm("v",cur_channel ,learning_rate)
                self.n_bn += 1
                
            elif layer["Layer"] == "FeedForward":
                layers = [n_H*n_W*cur_channel] + layer["layers"]
                self.Layers[f"NN_{self.no_nn + 1}"] = NeuralNetwork(self.Y,layers,self.learning_rate)
                self.no_nn += 1
                self.no_ff += 1                    

    def forward(self):
        self.Z = {}
        self.A = {}
        self.A[0] = self.X
        
        skip = False
        cur_idx = 0
        
        layer_keys = list(self.Layers.keys())

        for i, current_key in enumerate(layer_keys):
            layer = self.Layers[current_key]
            
            if isinstance(layer,BatchNorm):
                skip = True
                
            if skip:
                skip = False
                continue
        
            if isinstance(layer,Convolutional_Layer):
                self.Z[cur_idx+1] = layer.convolve(self.A[cur_idx])        
                next = self.Layers[layer_keys[i+1]]
                
                if isinstance(next,BatchNorm):
                    self.Z[cur_idx+1] = next.forward(self.Z[cur_idx+1])
                    self.A[cur_idx+1] = self.Layers[layer_keys[-1]].activation("relu",self.Z[cur_idx+1])
                else:
                    self.A[cur_idx+1] = self.Layers[layer_keys[-1]].activation("relu",self.Z[cur_idx+1])
    
            elif isinstance(layer,MaxPooling):
                self.Z[cur_idx+1] = layer.Pool(self.A[cur_idx])
                next = self.Layers[layer_keys[i+1]]
                
                if isinstance(next,NeuralNetwork):
                    self.A[cur_idx+1] = self.Z[cur_idx+1].reshape(self.Z[cur_idx+1].shape[0], -1)   
                else:
                    self.A[cur_idx+1] = self.Z[cur_idx+1]
                    
            elif isinstance(layer,NeuralNetwork):
                self.Z[cur_idx+1] = layer.forward_propagation(self.A[cur_idx])
                prediction = self.Layers[layer_keys[-1]].activation(Model.class_,self.Z[cur_idx+1])
        
            cur_idx += 1

        return prediction

    def backpropagation(self):
        k = self.no_ff
        layer_keys = list(self.Layers.keys())
        idx = len(self.Layers) - 1
        i = len(layer_keys) - self.n_bn - 1

        for current_key in reversed(layer_keys): 
            layer = self.Layers[current_key]
            
            if isinstance(layer, NeuralNetwork):
                for lay in reversed(range(1,layer.num_layer)):
                    if lay == layer.num_layer-1:
                        dZ = layer.back_propagtaion(activation=layer.class_,activation_prev=layer.parameter["A"+str(lay-1)]
                                ,layer=lay,activation_cur=layer.parameter["A"+str(lay)])
                        
                    else:
                        dZ = layer.back_propagtaion(activation="relu",activation_prev=layer.parameter["A"+str(lay-1)],
                              layer=lay,dZ=dZ,W=layer.Weight[lay+1],Z=layer.parameter["Z"+str(lay)])
                        
                d_flat = np.dot(dZ.T,self.Layers["NN_" + str(k)].Weight[1].T)
                dP = np.reshape(d_flat,self.Z[i].shape)
                idx -= 1
                k -= 1

                if k != 0:
                    i -= 1
            
            if isinstance(layer,MaxPooling):
                dP = layer.backprop(dP)
                idx -= 1
                i-=1

            if isinstance(layer,BatchNorm):
                d_activation = self.Layers[layer_keys[-1]].activation_derivative('relu',self.A[i])
                dP = layer.batch_backprop(dP)
                idx -= 1
                
            if isinstance(layer,Convolutional_Layer):
                prev = layer_keys[idx-1]
                if not isinstance(prev,BatchNorm):
                    d_activation = self.Layers[layer_keys[-1]].activation_derivative('relu',self.A[i])
                
                dP = d_activation * dP
                dP = layer.backprop(dP)
                idx -= 1
                i -= 1

    def optimizers(self,optimizer,dW,db,optimizer_param=None):
        if optimizer == "momentum":
            beta = optimizer_param['beta']
            prev_dkv = optimizer_param["dkv"]
            prev_dbv = optimizer_param["dbv"]
        
            dkv = (1-beta)*dW + (beta*prev_dkv)
            dbv = (1-beta)*db + (beta*prev_dbv)
        
            return dkv, dbv

        elif optimizer == "rmsprop":
            beta = optimizer_param['beta']
            prev_dkv = optimizer_param["dkv"]
            prev_dbv = optimizer_param["dbv"]
            
            dkv = (1-beta)*(dW**2) + (beta) * prev_dkv
            dbv = (1-beta)*(db**2) + (beta) * prev_dbv
            
            return dkv, dbv
                    
        elif optimizer == "adam":
            beta1 = optimizer_param['beta1']
            beta2 = optimizer_param['beta2']

            prev_dkv_1 = optimizer_param["dkv_1"]
            prev_dbv_1 = optimizer_param["dbv_1"]

            prev_dkv_2 = optimizer_param["dkv_2"]
            prev_dbv_2 = optimizer_param["dbv_2"]
            
            prev_dkv_1 = (1-beta1)*dW + (beta1)*prev_dkv_1
            prev_dbv_1 = (1-beta1)*db + (beta1)*prev_dbv_1
            
            prev_dkv_2 = (1-beta2)*(dW**2) + (beta2 * prev_dkv_2)
            prev_dbv_2 = (1-beta2)*(db**2) + (beta2 * prev_dbv_2)
            
            return prev_dkv_1, prev_dbv_1, prev_dkv_2, prev_dbv_2
            
    def update(self):
        for layer in self.Layers.values():
            if not isinstance(layer,MaxPooling):
                layer.update()
    
        
    def train(self,EPOCHS,optimizer="gradient_descent",optimizer_param=None):
        prev_conv_gradients = {}
        prev_batch_norm_gradients = {}
        prev_mlp_gradients = {}
        i = 1
        j = 1
        k = 1
        for layer in self.Layers.values():
            if isinstance(layer,Convolutional_Layer):
                prev_conv_gradients[i] = {"dK" : np.zeros_like(layer.K), "db" : np.zeros_like(layer.b)}
                if optimizer == "adam":
                    prev_conv_gradients[i].update({"dK_2" : np.zeros_like(layer.K), "db_2" : np.zeros_like(layer.b)})
                i += 1
                
            if isinstance(layer,BatchNorm):
                prev_batch_norm_gradients[j] = {"dG" : np.zeros_like(layer.G), "db" : np.zeros_like(layer.B)}
                if optimizer == "adam":
                    prev_batch_norm_gradients[j].update({"dG_2" : np.zeros_like(layer.G), "db_2" : np.zeros_like(layer.B)})
                j += 1

            if isinstance(layer,NeuralNetwork):
                for cur_layer in range(int(len(layer.gradients)/2)-1):
                    prev_mlp_gradients[str(k) + "_" + str(cur_layer + 1)] = {"dW" : np.zeros_like(layer.gradients["dW" + str(cur_layer + 1)]), 
                                                                "db" : np.zeros_like(layer.gradients["db" + str(cur_layer + 1)])}
                    
                    if optimizer == "adam":
                        prev_mlp_gradients[str(k) + "_" + str(cur_layer + 1)].update({"dW_2" : np.zeros_like(layer.gradients["dW" + str(cur_layer + 1)]),
                                                                "db_2" : np.zeros_like(layer.gradients["db" + str(cur_layer + 1)])})
                        
                k += 1
        
        for epoch in range(EPOCHS):
            prediction = self.forward()
            print(self.losses(self.loss,prediction))
            self.backpropagation()

            no_conv = self.no_conv
            conv_gradients = {}
            batch_norm_gradients = {}

            i = 1
            j = 1
            k = 1
            for layer in self.Layers.values():
                if isinstance(layer,Convolutional_Layer):
                    if optimizer == "gradient_descent":
                        pass
                        
                    elif optimizer == "momentum":
                        optimizer_param['beta'] = optimizer_param['beta']
                        optimizer_param['dkv'] = prev_conv_gradients[i]['dK']
                        optimizer_param['dbv'] = prev_conv_gradients[i]['db']
                    
                        prev_conv_gradients[i]['dK'],prev_conv_gradients[i]['db'] = self.optimizers(optimizer,
                                                                                        layer.dK,layer.db,
                                                                                        optimizer_param=optimizer_param)
    
                        layer.dK = prev_conv_gradients[i]['dK']
                        layer.db = prev_conv_gradients[i]['db']

                    elif optimizer == "rmsprop":
                        beta = optimizer_param['beta']
                        optimizer_param['beta'] = optimizer_param['beta']
                        optimizer_param['dkv'] = prev_conv_gradients[i]['dK']
                        optimizer_param['dbv'] = prev_conv_gradients[i]['db']
                    
                        prev_conv_gradients[i]['dK'],prev_conv_gradients[i]['db'] = self.optimizers(optimizer,
                                                                                        layer.dK,layer.db,
                                                                                        optimizer_param=optimizer_param)
    
                        layer.dK /= np.sqrt(prev_conv_gradients[i]['dK']/ (1-beta**2) + self.epsilon)
                        layer.db /= np.sqrt(prev_conv_gradients[i]['db']/ (1-beta**2) + self.epsilon)

                    elif optimizer == "adam":
                        beta1 = optimizer_param['beta1']
                        beta2 = optimizer_param['beta2']
                        optimizer_param['beta1'] = optimizer_param['beta1']
                        optimizer_param['beta2'] = optimizer_param['beta2']
                        
                        optimizer_param['dkv_1'] = prev_conv_gradients[i]['dK']
                        optimizer_param['dbv_1'] = prev_conv_gradients[i]['db']

                        optimizer_param['dkv_2'] = prev_conv_gradients[i]['dK_2']
                        optimizer_param['dbv_2'] = prev_conv_gradients[i]['db_2']

                        prev_conv_gradients[i]['dK'], prev_conv_gradients[i]['db'],prev_conv_gradients[i]['dK_2'],prev_conv_gradients[i]['db_2'] = self.optimizers(optimizer,layer.dK,layer.db,optimizer_param=optimizer_param)

                        prev_dkv_1 = prev_conv_gradients[i]['dK'] / (1-beta1**2)
                        prev_dbv_1 = prev_conv_gradients[i]['db'] / (1-beta1**2)
                        
                        prev_dkv_2 = prev_conv_gradients[i]['dK_2'] / (1-beta2**2)
                        prev_dbv_2 = prev_conv_gradients[i]['db_2'] / (1-beta2**2)

                        layer.dK = prev_dkv_1/(np.sqrt(prev_dkv_2 + self.epsilon))
                        
                        layer.db = prev_dbv_1/(np.sqrt(prev_dbv_2 + self.epsilon))
                        
                    i += 1
                    
                if isinstance(layer,BatchNorm):
                    if optimizer == "gradient_descent":
                        pass

                    elif optimizer == "momentum":
                        optimizer_param['beta'] = optimizer_param['beta']
                        optimizer_param['dkv'] = prev_batch_norm_gradients[j]['dG']
                        optimizer_param['dbv'] = prev_batch_norm_gradients[j]['db']

                        prev_batch_norm_gradients[j]['dG'],prev_batch_norm_gradients[j]['db'] = self.optimizers(optimizer,
                                                                                                layer.dG, layer.dB,
                                                                                                optimizer_param=optimizer_param)
                        layer.dG = prev_batch_norm_gradients[j]['dG']
                        layer.dB = prev_batch_norm_gradients[j]['db']

                    elif optimizer == "rmsprop":
                        beta = optimizer_param['beta']
                        optimizer_param['beta'] = optimizer_param['beta']
                        optimizer_param['dkv'] = prev_batch_norm_gradients[j]['dG']
                        optimizer_param['dbv'] = prev_batch_norm_gradients[j]['db']
                    
                        prev_batch_norm_gradients[j]['dG'],prev_batch_norm_gradients[j]['db'] = self.optimizers(optimizer,
                                                                                        layer.dG,layer.dB,
                                                                                        optimizer_param=optimizer_param)
    
                        layer.dG /= np.sqrt(prev_batch_norm_gradients[j]['dG']/(1-beta**2) + self.epsilon)
                        layer.dB /= np.sqrt(prev_batch_norm_gradients[j]['db']/ (1-beta**2) + self.epsilon)[0,0,0]

                    elif optimizer == "adam":
                        optimizer_param['beta1'] = optimizer_param['beta1']
                        optimizer_param['beta2'] = optimizer_param['beta2']

                        beta1 = optimizer_param['beta1']
                        beta2 = optimizer_param['beta2']
                        
                        optimizer_param['dkv_1'] = prev_batch_norm_gradients[j]['dG']
                        optimizer_param['dbv_1'] = prev_batch_norm_gradients[j]['db']

                        optimizer_param['dkv_2'] = prev_batch_norm_gradients[j]['dG_2']
                        optimizer_param['dbv_2'] = prev_batch_norm_gradients[j]['db_2']

                        prev_batch_norm_gradients[j]['dG'], prev_batch_norm_gradients[j]['db'], prev_batch_norm_gradients[j]['dG_2'],prev_batch_norm_gradients[j]['db_2'] = self.optimizers(optimizer,layer.dG,layer.dB,optimizer_param=optimizer_param)

                        prev_dkv_1 = prev_batch_norm_gradients[j]['dG'] / (1-beta1**2)
                        prev_dbv_1 = prev_batch_norm_gradients[j]['db'] / (1-beta1**2)
                        
                        prev_dkv_2 = prev_batch_norm_gradients[j]['dG_2'] / (1-beta2**2)
                        prev_dbv_2 = prev_batch_norm_gradients[j]['db_2'] / (1-beta2**2)

                        layer.dG = prev_dkv_1/(np.sqrt(prev_dkv_2 + self.epsilon))
                        
                        layer.db = prev_dbv_1/(np.sqrt(prev_dbv_2 + self.epsilon))
                    
                    j += 1

                if isinstance(layer,NeuralNetwork):
                    n = int(len(layer.gradients)/2)
                    
                    if optimizer == "gradient_descent":
                        pass
                        
                    elif optimizer == "momentum":
                        optimizer_param['beta'] = optimizer_param['beta']

                        for cur_layer in range(n-1):
                            optimizer_param['dkv'] = prev_mlp_gradients[str(k) + "_" + str(cur_layer + 1)]['dW']
                            optimizer_param['dbv'] = prev_mlp_gradients[str(k) + "_" + str(cur_layer + 1)]['db']

                            dW = layer.gradients["dW" + str(cur_layer+1)]
                            db = layer.gradients["db" + str(cur_layer+1)]
                            
                            prev_mlp_gradients[str(k) + "_" + str(cur_layer + 1)]['dW'], prev_mlp_gradients[str(k) + "_" + str(cur_layer + 1)]['db'] = self.optimizers(optimizer, dW, db, optimizer_param=optimizer_param)                            
                            
                            layer.gradients["dW" + str(cur_layer+1)] = prev_mlp_gradients[str(k) + "_" + str(cur_layer + 1)]['dW']
                            layer.gradients["db" + str(cur_layer+1)] = prev_mlp_gradients[str(k) + "_" + str(cur_layer + 1)]['db'][0]                            
                            
                    elif optimizer == "rmsprop":
                        optimizer_param['beta'] = optimizer_param['beta']

                        for cur_layer in range(n-1):
                            optimizer_param['dkv'] = prev_mlp_gradients[str(k) + "_" + str(cur_layer + 1)]['dW']
                            optimizer_param['dbv'] = prev_mlp_gradients[str(k) + "_" + str(cur_layer + 1)]['db']
                    
                            prev_mlp_gradients[str(k) + "_" + str(cur_layer + 1)]['dW'], prev_mlp_gradients[str(k) + "_" + str(cur_layer + 1)]['db'] = self.optimizers(optimizer,layer.gradients["dW" + str(cur_layer+1)], layer.gradients["db" + str(cur_layer+1)],optimizer_param=optimizer_param)
    
                            layer.gradients["dW" + str(cur_layer+1)] /= np.sqrt(prev_mlp_gradients[str(k) + "_" + str(cur_layer + 1)]['dW'] + self.epsilon)
                            layer.gradients["db" + str(cur_layer+1)] /= np.sqrt(prev_mlp_gradients[str(k) + "_" + str(cur_layer + 1)]['db'] + self.epsilon)
                            
                    elif optimizer == "adam":
                        optimizer_param['beta1'] = optimizer_param['beta1']
                        optimizer_param['beta2'] = optimizer_param['beta2']

                        beta1 = optimizer_param['beta1']
                        beta2 = optimizer_param['beta2']
    
                
                        for cur_layer in range(n-1):
                            optimizer_param['dkv_1'] = prev_mlp_gradients[str(k) + "_" + str(cur_layer + 1)]['dW']
                            optimizer_param['dbv_1'] = prev_mlp_gradients[str(k) + "_" + str(cur_layer + 1)]['db']
    
                            optimizer_param['dkv_2'] = prev_mlp_gradients[str(k) + "_" + str(cur_layer + 1)]['dW_2']
                            optimizer_param['dbv_2'] = prev_mlp_gradients[str(k) + "_" + str(cur_layer + 1)]['db_2']
    
                            prev_mlp_gradients[str(k) + "_" + str(cur_layer + 1)]['dW'], prev_mlp_gradients[str(k) + "_" + str(cur_layer + 1)]['db'], prev_mlp_gradients[str(k) + "_" + str(cur_layer + 1)]['dW_2'], prev_mlp_gradients[str(k) + "_" + str(cur_layer + 1)]['db_2'] = self.optimizers(optimizer,layer.gradients["dW" + str(cur_layer+1)],layer.gradients["db" + str(cur_layer+1)],optimizer_param=optimizer_param)
    
                            prev_dkv_1 = prev_mlp_gradients[str(k) + "_" + str(cur_layer + 1)]['dW'] / (1-beta1**2)
                            prev_dbv_1 = prev_mlp_gradients[str(k) + "_" + str(cur_layer + 1)]['db'] / (1-beta1**2)
                            
                            prev_dkv_2 = prev_mlp_gradients[str(k) + "_" + str(cur_layer + 1)]['dW_2'] / (1-beta2**2)
                            prev_dbv_2 = prev_mlp_gradients[str(k) + "_" + str(cur_layer + 1)]['db_2'] / (1-beta2**2)
                            
                            layer.gradients["dW" + str(cur_layer+1)] = prev_dkv_1/(np.sqrt(prev_dkv_2 + self.epsilon))
                            layer.gradients["db" + str(cur_layer+1)] = prev_dbv_1/(np.sqrt(prev_dbv_2 + self.epsilon))
                    
                    k += 1

            self.update()               

example_layer_arch = [
            {"Layer" : "Convolution","kernel_size":3,"no_of_filter":10,"pad":2,"stride":1},
              {"Layer" : 'BatchNorm'},
              {"Layer" : 'Pool' ,"pool_size":2,"mode":"max","stride":1}, 
              {"Layer" : "Convolution","kernel_size":3,"no_of_filter":5,"pad":2,"stride":1},
              {"Layer" : "BatchNorm"},
              {"Layer" : 'Pool', "pool_size":2,"mode":"max","stride":1}, 
              {"Layer" : 'FeedForward', "layers" :[256,256,10]}
]


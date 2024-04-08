import numpy as np
from scipy.special import expit, softmax
from sklearn.metrics import log_loss


class NeuralNetwork:
    def __init__(self,Y,layer_dim,learning_rate=1e-2,verbose=True,gradient_clip=True,clip_value = 3.0):
        self.Y = Y
        self.num_layer = len(layer_dim)
        self.layer_dim = layer_dim
        self.class_ = "Sigmoid" if layer_dim[-1] == 1 else "Softmax"
        self.learning_rate = learning_rate
        self.m = self.Y.shape[0]
        self.verbose = verbose
        self.gradient_clip = gradient_clip
        self.clip_value = clip_value
            
        self.Weight = {}
        self.Bias = {}

        for j in range(1,self.num_layer): 
            # Xavier Initialization
            self.Weight[j] = np.random.randn(self.layer_dim[j-1],self.layer_dim[j])*np.sqrt(
                                                2/self.layer_dim[j-1])
            
            self.Bias[j] = np.zeros((1,self.layer_dim[j]))

        self.gradients = {"dW" + str(i): np.zeros_like(self.Weight[i]) for i in range(1, self.num_layer)}
        self.gradients.update({"db" + str(i): np.zeros_like(self.Bias[i]) for i in range(1, self.num_layer)})
    
    def linear(self,X,W,b):
        return np.dot(X,W) + b
    
    def losses(self,losses,Y_pred):

        if losses == "binary_crossentropy":
            # Y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
            # term_0 = (1-self.Y) * np.log(1-Y_pred + 1e-7)
            # term_1 = self.Y * np.log(Y_pred + 1e-7)
            # return -np.mean(term_0+term_1, axis=0)
            # return -1/self.m * np.sum((self.Y*np.log(Y_pred)) + ((1-self.Y)*np.log(1-Y_pred)),axis=0,keepdims=True)
            return log_loss(self.Y, Y_pred)
        
        if losses == "categorical_crossentropy":
            #return -np.mean(np.sum(self.Y * np.log(Y_pred), axis=0))
            return log_loss(self.Y, Y_pred)
    
    def activation(self,activation,Z):
        if activation == "relu":
            return np.maximum(0,Z)
        
        if activation == "Sigmoid":
            #return 1/(1+np.exp(-Z))
            return expit(Z)
        
        if activation == "Softmax":
            # Z = Z - np.expand_dims(np.max(Z, axis = 1), 1)
            # Z = np.exp(Z)
            # ax_sum = np.expand_dims(np.sum(Z, axis = 1), 1)
            # p = Z / ax_sum
            # return p
            return softmax(Z)
    
    def activation_derivative(self,activation,Z):
        if (activation == "Sigmoid" or activation == "Softmax"):
            return (self.Y - Z).T
        
        if activation == "relu":
            return (Z>0)*1
    
    def back_propagtaion(self,activation,activation_prev,layer,Z=None,W=None,dZ=None,activation_cur=None):
        
        if (activation == "Sigmoid") | (activation == "Softmax"):
            self.dZ = self.activation_derivative(activation,activation_cur)
            self.gradients["dW"+str(layer)] = 1/self.m * np.dot(self.dZ,activation_prev).T
            self.gradients["db"+str(layer)] = 1/self.m * np.sum(self.dZ,axis=1,keepdims=True).T

            if self.gradient_clip:
                for grad in ["dW" + str(layer), "db" + str(layer)]:
                    np.clip(self.gradients[grad], -self.clip_value, self.clip_value, out=self.gradients[grad])
            
        elif (activation == "relu"): 
            self.dZ = np.dot(W,dZ) * self.activation_derivative(activation,Z).T
            self.gradients["dW"+str(layer)] = 1/self.m * np.dot(self.dZ,activation_prev).T
            self.gradients["db"+str(layer)] = 1/self.m * np.sum(self.dZ,axis=1,keepdims=True).T

            if self.gradient_clip:
                for grad in ["dW" + str(layer), "db" + str(layer)]:
                    np.clip(self.gradients[grad], -self.clip_value, self.clip_value, out=self.gradients[grad])
        
        return self.dZ
    
    def forward_propagation(self,X):
        
        self.gradients = {"dW" + str(i): np.zeros_like(self.Weight[i]) for i in range(1, self.num_layer )}
        self.gradients.update({"db" + str(i): np.zeros_like(self.Bias[i]) for i in range(1, self.num_layer )})

        self.parameter = {}

        self.parameter['A0'] = X
        
        for layer in range(1,self.num_layer):
            self.parameter["Z"+str(layer)] = self.linear(self.parameter['A'+str(layer-1)],self.Weight[layer],
                                                         self.Bias[layer])

            if layer != self.num_layer-1:
                self.parameter["A"+str(layer)] = self.activation("relu",self.parameter["Z"+str(layer)])
            else:
                self.parameter["A"+str(layer)] = self.parameter["Z"+str(layer)]
         
        return self.parameter["A"+str(self.num_layer-1)]

    
    def update(self):
        for layer in range(1,self.num_layer):
            self.Weight[layer] = self.Weight[layer] - self.learning_rate * self.gradients["dW"+str(layer)]
            self.Bias[layer] = self.Bias[layer] - self.learning_rate * self.gradients["db"+str(layer)]
            
    def train(self,X,EPOCHS):
         for epoch in range(EPOCHS):
            # FORWARD PROPAGATION
            Z = self.forward_propagation(X)

            prediction = self.activation(self.class_,Z)

            if self.class_ == "Sigmoid":
                loss_type = "binary_crossentropy"
            elif self.class_ == "Softmax":
                loss_type = "categorical_crossentropy"

            loss = self.losses(loss_type,prediction)

            if self.verbose:
                print(loss)
                    
            # BACKPROPAGATION 
            for layer in reversed(range(1,self.num_layer)):
                
                if layer == self.num_layer-1:
                    dZ = self.back_propagtaion(activation=self.class_,activation_prev=self.parameter["A"+str(layer-1)]
                            ,layer=layer,activation_cur=self.parameter["A"+str(layer)])
                    
                else:
                    dZ = self.back_propagtaion(activation="relu",activation_prev=self.parameter["A"+str(layer-1)],
                          layer=layer,dZ=dZ,W=self.Weight[layer+1],Z=self.parameter["Z"+str(layer)])
                    
            self.update()
    
    def predict(self,X):
        return self.forward_propagation(X)

class BatchNorm:
    def __init__(self, task, out_dim, learning_rate=1e-4, mode="training"):
        # task = v for vision(4 dimensional dataset), l for language (3 dimensional dataset) 
        if task == "v":
            self.G = np.random.randn(1,1,1,out_dim)#Scaling factor
            self.B = np.random.randn(1,1,1,out_dim)#Shifting factor
        elif task == "l":
            self.G = np.random.randn(1,1,out_dim)#Scaling factor
            self.B = np.random.randn(1,1,out_dim)#Shifting factor
        else:
           self.G = np.random.randn(1,out_dim)#Scaling factor
           self.B = np.random.randn(1,out_dim)#Shifting factor 

        self.task = task
        self.learning_rate = learning_rate
        self.mode = mode

        self.dG = None
        self.dB = None

    def setMode(self,mode):
        self.mode = mode
        
    def forward(self, Z):
        self.mean_ = 0
        self.std_ = 0
        self.M = 0
        self.N = 0
        epsilon = 1e-5

        self.dG = np.zeros_like(self.G)
        self.dB = np.zeros_like(self.B)
        
        self.temp = Z
        
        if self.mode == "training":
            if self.task == "v":
                self.mean_ = np.mean(Z,axis=(0,1,2))
                self.std_ = np.std(Z,axis=(0,1,2))
            elif self.task == "l":
                self.mean_ = np.mean(Z,axis=(0,1))
                self.std_ = np.std(Z,axis=(0,1))
            else:
                self.mean_ = np.mean(Z,axis=0)
                self.std_ = np.std(Z,axis=0)
                
            decay = 0.9
            
            if not hasattr(self,"running_mean"):
                self.running_mean = self.mean_
                self.running_std_ = self.std_
            else:
                self.running_mean = decay * self.running_mean + (1-decay)*self.mean_
                self.running_std_ = decay * self.running_std_ + (1-decay)*self.std_
        else:
            self.mean_ = self.running_mean
            self.std_ = self.running_std_
                
        # Standardization
        self.M = (Z - self.mean_) / (np.sqrt(self.std_**2 + epsilon))
        self.N = self.M * self.G + (self.B)
        
        return self.N

    def batch_backprop(self,dZ):
        dZ_ = dZ*self.G
        self.dG = dZ*self.M
        self.dB = np.sum(dZ,axis=(0,1,2))
        m = dZ.shape[0]

        temp_1 = dZ*(self.temp - self.mean_)*(-1 /(self.std_*(self.std_**2)))

        if self.task == "v":
            dsigma = np.sum(temp_1, axis=(0, 1, 2), keepdims=True)
        elif self.task == "l":
            dsigma = np.sum(temp_1, axis=(0, 1), keepdims=True)
        else:
            dsigma = np.sum(temp_1, axis=0, keepdims=True)

        if self.task == "v":
            dmu = np.sum(-dZ*(1/self.std_),axis=(0,1,2))+dsigma*np.sum(-2*(self.temp-self.mean_),axis=(0,1,2))/m
        elif self.task == "l":
            dmu = np.sum(-dZ*(1/self.std_),axis=(0,1))+dsigma*np.sum(-2*(self.temp-self.mean_),axis=(0,1))/m
        else:
            dmu = np.sum(-dZ*(1/self.std_),axis=0)+dsigma*np.sum(-2*(self.temp-self.mean_),axis=0)/m 
    
        dback = dZ * (1 / self.std_) + dsigma * 2 * (self.temp - self.mean_) / m + (dmu / m)
        
        return dback

    def update(self):
        self.G = self.G - self.learning_rate * self.dG
        self.B = self.B - self.learning_rate * self.dB


import numpy as np

def xavier_factor(input_dim,output_dim):
    return np.sqrt(2.0/(input_dim + output_dim))

class RNN:
    def __init__(self,X,Y,num_of_class,hidden_size,lr=1e-4,dropout=False,dropout_rate=0.2):
        self.X = X
        self.Y = Y
        self.embed_dim = X.shape[2]
        self.sequence_length = X.shape[1]
        self.training_size = X.shape[0]
        self.hidden_size = hidden_size
        self.lr = lr
        self.dropout = dropout
        self.dropout_rate = dropout_rate
        
        self.h_prev = np.random.randn(self.training_size,self.hidden_size)
        
        self.Wa = np.random.randn(self.hidden_size,self.hidden_size) * xavier_factor(self.hidden_size,
                                                                                     self.hidden_size)
        self.dWa = np.zeros((self.hidden_size,self.hidden_size))
        
        self.Wx = np.random.randn(self.embed_dim,self.hidden_size) * xavier_factor(self.embed_dim,self.hidden_size)
        self.dWx = np.zeros((self.embed_dim,self.hidden_size))
        
        self.Wy = np.random.randn(self.hidden_size,num_of_class) * xavier_factor(self.hidden_size,num_of_class)
        self.dWy = np.zeros((self.hidden_size,num_of_class))
    
        self.b = np.random.randn(self.hidden_size)
        self.db = np.zeros((self.hidden_size))
        
        self.by = np.random.randn(num_of_class)
        self.dby = np.zeros((num_of_class))
        
        self.cache = {"A0":self.h_prev}
        self.K = np.zeros((self.training_size,self.sequence_length,num_of_class))
        
    def cross_entropy(self,Y,pred):
        epsilon = 1e-8
        pred = np.clip(pred,epsilon,1.0 - epsilon)
        return -np.sum(Y * np.log(pred))
    
    def calculate_cross_entropy(self,pred):
        total_loss = 0
        for i in range(pred.shape[0]):
            for j in range(pred.shape[1]):
                total_loss += self.cross_entropy(self.Y[i,j,:],pred[i,j,:])
        return total_loss
        
    def linear_forward(self,h_prev,X):
        Z = np.dot(X,self.Wx) + np.dot(h_prev,self.Wa)
        A = np.tanh(Z)
        drop = 0
        
        if self.dropout:
            prob = np.random.random(A.shape)
            drop = (prob > self.dropout_rate).astype(int)
            A = (A * drop)/(1 - self.dropout_rate)
        
        K = np.dot(A,self.Wy)
        Y = tf.nn.softmax(K)
        
        return Z,A,K,drop
    
    def forward_propagation(self):
        self.drop = {}
        for i in range(1,self.sequence_length):
            self.cache["Z"+str(i)],self.cache["A"+str(i)],self.K[:,i,:],self.drop["D"+str(i)] = self.linear_forward(
                                                                        self.cache["A"+str(i-1)],self.X[:,i,:])
        print(self.calculate_cross_entropy(self.K))
        return self.cache,self.K
    
    def change_lr(self,new_lr):
        self.lr = new_lr
    
    def back_propagation(self):
        dh_next = np.zeros_like(self.cache["A1"])

        for i in reversed(range(1, self.sequence_length)):
            dy = self.K[:, i, :] - self.Y[:, i, :]
            self.dWy += np.dot(self.cache["A" + str(i)].T, dy)
            self.dby += np.sum(dy, axis=0)
            dh = np.dot(dy, self.Wy.T) + np.dot(dh_next, self.Wa.T) * (1 - self.cache["A" + str(i)] ** 2)
            
            if self.dropout:
                dh  = (dh * self.drop["D"+str(i)])/(1-self.dropout_rate)

            self.dWa += np.dot(self.cache["A" + str(i - 1)].T, dh)
            self.dWx += np.dot(self.X[:, i, :].T, dh)
            self.db += np.sum(dh, axis=0)
            dh_next = dh

        dy = self.K[:, 0, :] - self.Y[:, 0, :]
        self.dWy += np.dot(self.cache["A0"].T, dy)
        self.dby += np.sum(dy, axis=0)
        dh = np.dot(dy, self.Wy.T) + np.dot(dh_next, self.Wa.T) * (1 - self.cache["A0"] ** 2)
        if self.dropout:
            dh  = (dh * self.drop["D1"])/(1-self.dropout_rate)
        
        self.dWx += np.dot(self.X[:, 0, :].T, dh)
        self.db += np.sum(dh, axis=0)

        max_grad_norm = 1.0  # Set the maximum gradient norm
        for gradient in [self.dWx, self.dWa, self.dWy, self.db, self.dby]:
            np.clip(gradient, -max_grad_norm, max_grad_norm, out=gradient)

        return self.dWy, self.dby, self.dWa, self.dWx, self.db
        

    def update(self):
        self.Wx = self.Wx - self.lr * self.dWx
        self.Wa = self.Wa - self.lr * self.dWa        
        self.Wy = self.Wy - self.lr * self.dWy        
        self.by = self.by - self.lr * self.dby    
        self.b = self.b - self.lr * self.db        

class GRU:
    def __init__(self,X,Y,num_of_class,hidden_size,lr=1e-5):
        self.X = X
        self.Y = Y
        self.num_of_class = num_of_class
        self.hidden_size = hidden_size
        self.lr = lr
        
        self.training_size = X.shape[0]
        self.sequence_length = X.shape[1]
        self.embed_size = X.shape[2]
        
        # Reset Gate Weight
        self.Wr = np.random.randn(self.embed_size,self.hidden_size)
        self.Ur = np.random.randn(self.hidden_size,self.hidden_size)
        self.br = np.zeros((self.hidden_size,1))
        
        # Update Gate Weight
        self.Wu = np.random.randn(self.embed_size,self.hidden_size)
        self.Uu = np.random.randn(self.hidden_size,self.hidden_size)
        self.bu = np.zeros((self.hidden_size,1))
        
        # Weight controlling the link between Reset Gate and Candidate
        self.Wh = np.random.randn(self.hidden_size,self.hidden_size)
        self.bh = np.zeros((self.hidden_size,1))
        
        # Weight for prediction
        self.Wy = np.random.randn(self.num_of_class,self.hidden_size)
        self.by = np.zeros((self.num_of_class,1))
        
        self.H = []
        self.A = []
        self.candidates = []
        self.Reset_Gates = []
        self.Updated_Gates = []
        self.pred = np.zeros((self.training_size,self.sequence_length,self.num_of_class))
        
    def sigmoid(self,Z):
        Z = np.clip(Z, -100, 100)
        return 1/(1+np.exp(-Z))
    
    def softmax(self,Z):
        return tf.nn.softmax(Z)
    
    def cross_entropy(self,Y,pred):
        epsilon = 1e-8
        pred = np.clip(pred,epsilon,1.0 - epsilon)
        return -np.sum(Y * np.log(pred))
    
    def calculate_cross_entropy(self,pred):
        total_loss = 0
        for i in range(pred.shape[0]):
            for j in range(pred.shape[1]):
                total_loss += self.cross_entropy(self.Y[i,j,:],pred[i,j,:])
        return total_loss
        
    def forward_propagation(self,X,h_prev):
        Z1 = np.dot(X,self.Wr) + np.dot(h_prev,self.Ur) + self.br.T
        
        self.A.append(Z1)
        Reset_Gates = expit(Z1)
        
        Z2 = np.dot(X,self.Wu) + np.dot(h_prev,self.Uu) + self.bu.T
       
        self.A.append(Z2)
        Updated_Gates = expit(Z2)
        
        Candidates = np.tanh(np.dot(self.Wh,(Reset_Gates*h_prev).T)) + self.bh
        self.candidates.append(Candidates)
        H = Updated_Gates * Candidates.T + (1 - Updated_Gates) * h_prev
        
        K = np.dot(H,self.Wy.T) + self.by.T
        Y = self.softmax(K)
        
        return Reset_Gates,Updated_Gates,H,Y
    
    def backpropagation(self):
        
        dUr = np.zeros_like(self.Ur)
        dWr = np.zeros_like(self.Wr)
        dbr = np.zeros_like(self.br)
        
        dUu = np.zeros_like(self.Uu)
        dWu = np.zeros_like(self.Wu)
        dbu = np.zeros_like(self.bu)
        
        dWh = np.zeros_like(self.Wh)
        dbh = np.zeros_like(self.bh)
        
        dWy = np.zeros_like(self.Wy)
        dby = np.zeros_like(self.by)
        
        for i in reversed(range(1,self.sequence_length)):
            dy = self.pred[:,i,:] - self.Y[:,i,:]
            da = np.dot(dy, self.Wy)
            
            # Update for dWy and dby
            dWy += np.dot(dy.T,self.H[i])
            dby += np.sum(dy.T,axis=1,keepdims=True)
            
            # Update for relevant gates
            dr1 = da * self.Updated_Gates[i] * (self.Reset_Gates[i] * (1 - self.Reset_Gates[i])) *self.H[i-1]
            dr2 = np.dot(dr1,self.Wh) * (1-np.square(self.candidates[i])).T
            dUr += np.dot(dr2.T, self.H[i])
            dWr += np.dot(dr2.T, self.X[:,i,:]).T
            dbr += np.sum(dr2.T, axis=1, keepdims=True)
            
            # Update for updated gates
            du1 = (da * (self.Updated_Gates[i]*(1 - self.Updated_Gates[i])))
            du2 = du1*((-self.H[i-1])+(np.dot(self.Reset_Gates[i]*self.H[i-1],self.Wh.T)))
            dWu += np.dot(du2.T,self.X[:,i,:]).T
            dUu += np.dot(du2.T,self.H[i])
            dbu += np.sum(du2.T,axis=1,keepdims=True)
            
            
            dWh += np.dot((da*(1-np.square(self.candidates[i])).T*self.Updated_Gates[i]).T,
                          self.H[i-1]*self.Reset_Gates[i])
            dbh += np.sum(da*(1-np.square(self.candidates[i])).T*self.Updated_Gates[i]*self.Reset_Gates[i], axis=0,
                          keepdims=True).T
            
        max_grad = 5.0
        for i in [dWh,dWr,dWu,dWy,dUr,dUu,dbh,dbr,dbu,dby]:
            np.clip(i,-max_grad,max_grad,out=i)
        
        self.update(dWh,dWr,dWu,dWy,dUr,dUu,dbh,dbr,dbu,dby)
            
    def update(self,dWh,dWr,dWu,dWy,dUr,dUu,dbh,dbr,dbu,dby): 
        self.Wh = self.Wh -  self.lr * dWh
        self.Wr = self.Wr -  self.lr * dWr
        self.Wu = self.Wu -  self.lr * dWu
        self.Wy = self.Wy -  self.lr * dWy
        
        self.Ur = self.Ur - self.lr * dUr
        self.Uu = self.Uu -  self.lr * dUu
        
        self.bh = self.bh - self.lr * dbh
        self.br = self.br - self.lr * dbr
        self.bu = self.bu - self.lr * dbu
        self.by = self.by - self.lr * dby
    
    def train(self,EPOCHS):
        h_prev = np.random.randn(self.training_size,self.hidden_size)
        for j in range(EPOCHS):
            for i in range(self.sequence_length):
                RG,UG,h_prev,self.pred[:,i,:] = self.forward_propagation(self.X[:,i,:], h_prev)
                self.Reset_Gates.append(RG)
                self.Updated_Gates.append(UG)
                self.H.append(h_prev)
            print(self.calculate_cross_entropy(self.pred))
            self.backpropagation()
        return self.Y,self.H
    
class LSTM:
    def __init__(self,X,Y,num_of_class,hidden_size,lr=1e-4):
        self.X = X
        self.Y = Y
        self.num_of_class = num_of_class
        self.hidden_size = hidden_size
        self.lr = lr
        
        self.train_size,self.seq_len,self.embed_size = X.shape
        
        #Weight for Input Gates
        self.Wi = np.random.randn(self.embed_size,self.hidden_size)
        self.Ui = np.random.randn(self.hidden_size,self.hidden_size)
        self.bi = np.random.randn(self.hidden_size,1)
        
        #Weight for Forget Gates
        self.Wf = np.random.randn(self.embed_size,self.hidden_size)
        self.Uf = np.random.randn(self.hidden_size,self.hidden_size)
        self.bf = np.random.randn(self.hidden_size,1)
        
        #Weight for Output Gates
        self.Wo = np.random.randn(self.embed_size,self.hidden_size)
        self.Uo = np.random.randn(self.hidden_size,self.hidden_size)
        self.bo = np.random.randn(self.hidden_size,1)
        
        #Weight for cell state
        self.Wc = np.random.randn(self.embed_size,self.hidden_size)
        self.Uc = np.random.randn(self.hidden_size,self.hidden_size)
        self.bc = np.random.randn(self.hidden_size,1)
        
        #Weight for prediction
        self.Wy = np.random.randn(self.hidden_size,self.num_of_class)
        self.by = np.random.randn(self.num_of_class,1)
        
        self.Y_pred = np.zeros((self.train_size,self.seq_len,self.num_of_class))
        
    def softmax(self,Z):
        ex = np.exp(Z - np.max(Z))
        return ex/(np.sum(ex,axis=1,keepdims=True))
    
    def cross_entropy(self,Y,pred):
        epsilon = 1e-8
        pred = np.clip(pred,epsilon,1.0 - epsilon)
        return -np.sum(Y * np.log(pred))
    
    def calculate_cross_entropy(self):
        total_loss = 0
        for i in range(self.Y_pred.shape[0]):
            for j in range(self.Y_pred.shape[1]):
                total_loss += self.cross_entropy(self.Y[i,j,:],self.Y_pred[i,j,:])
        return total_loss/self.Y_pred.shape[0]
        
    def forward_propagation(self):
        self.C_state = [np.random.randn(self.train_size,self.hidden_size)]
        self.H_state = [np.random.randn(self.train_size,self.hidden_size)]
        self.Z3_ = []
        self.Z5_ = []
        self.Input_Gates = []
        self.Output_Gates = []
        self.Forget_Gates = []
        self.Candidates = []
        
        for i in range(self.seq_len):
            Z1 = np.dot(self.X[:,i,:],self.Wf) + np.dot(self.H_state[i],self.Uf) + self.bf.T
            Forget_Gate = expit(Z1)
            self.Forget_Gates.append(Forget_Gate)
            
            Z2 = np.dot(self.X[:,i,:],self.Wi) + np.dot(self.H_state[i],self.Ui) + self.bi.T
            Input_Gate = expit(Z2)
            self.Input_Gates.append(Input_Gate)
            
            Z3 = np.dot(self.X[:,i,:],self.Wc) + np.dot(self.H_state[i],self.Uc) + self.bc.T
            candidates = np.tanh(Z3)
            self.Candidates.append(candidates)
            self.Z3_.append(Z3)
            
            self.C_state.append(Forget_Gate * self.C_state[i]  + Input_Gate * candidates)
            
            Z4 = np.dot(self.X[:,i,:],self.Wo) + np.dot(self.H_state[i],self.Uo) + self.bo.T
            Output_Gate = expit(Z4)
            self.Output_Gates.append(Output_Gate)
            
            Z5 = np.tanh(self.C_state[i+1])
            h_state = Output_Gate * Z5
            self.Z5_.append(Z5)
            self.H_state.append(h_state)

            K = np.dot(h_state,self.Wy) + self.by.T
            self.Y_pred[:,i,:] = self.softmax(K)
            
            
        return self.Y_pred
    
    def change_lr(self,new_lr):
        self.lr = new_lr
    
    def backpropagation(self):
        self.dWf = np.zeros_like(self.Wf)
        self.dUf = np.zeros_like(self.Uf)
        self.dbf = np.zeros_like(self.bf)

        self.dWi = np.zeros_like(self.Wi)
        self.dUi = np.zeros_like(self.Ui)
        self.dbi = np.zeros_like(self.bi)

        self.dWc = np.zeros_like(self.Wc)
        self.dUc = np.zeros_like(self.Uc)
        self.dbc = np.zeros_like(self.bc)

        self.dWo = np.zeros_like(self.Wo)
        self.dUo = np.zeros_like(self.Uo)
        self.dbo = np.zeros_like(self.bo)

        self.dWy = np.zeros_like(self.Wy)
        self.dby = np.zeros_like(self.by)

        
        for i in range(seq_len):
            dy = np.dot(self.Y_pred[:,i,:] - self.Y[:,i,:],self.Wy.T) * self.Output_Gates[i] * (1-np.square(np.tanh(
                                                                        self.C_state[i])))  

            df = dy * self.C_state[i-1] * self.Forget_Gates[i]*(1-self.Forget_Gates[i])
            self.dWf += np.dot(df.T,self.X[:,i,:]).T
            self.dUf += np.dot(df.T,self.H_state[i-1])
            self.dbf += np.sum(df.T,axis=1,keepdims=True)

            di = dy * self.Candidates[i] * (self.Input_Gates[i]*(1-self.Input_Gates[i]))
            self.dWi += np.dot(di.T,self.X[:,i,:]).T
            self.dUi += np.dot(di.T,self.H_state[i-1])
            self.dbi += np.sum(di.T,axis=1,keepdims=True)

            do = np.dot(self.Y_pred[:,i,:] - self.Y[:,i,:],self.Wy.T) * self.Z5_[i] * self.Output_Gates[i]*(
                                                                                1-self.Output_Gates[i])
            self.dWo += np.dot(self.X[:,i,:].T,do)
            self.dUo += np.dot(self.H_state[i].T,do)
            self.dbo += np.sum(do.T,axis=1,keepdims=True)

            dc = dy * (self.Input_Gates[i]*(1-np.square(np.tanh(self.Z3_[i]))))
            self.dWc += np.dot(dc.T,self.X[:,i,:]).T
            self.dUc += np.dot(dc.T,self.H_state[i-1])
            self.dbc += np.sum(dc.T,axis=1,keepdims=True)

            self.dWy += np.dot(self.H_state[i].T,self.Y_pred[:,i,:] - self.Y[:,i,:])
            self.dby += np.sum((self.Y_pred[:,i,:] - self.Y[:,i,:]).T,axis=1,keepdims=True)

    def update(self):
        self.Wf = self.Wf - (self.lr*self.dWf)
        self.Uf = self.Uf - (self.lr*self.dUf)
        self.bf = self.bf - (self.lr*self.dbf)

        self.Wi = self.Wi - (self.lr*self.dWi)
        self.Ui = self.Ui - (self.lr*self.dUi)
        self.bi = self.bi - (self.lr*self.dbi)

        self.Wc = self.Wc - (self.lr*self.dWc)
        self.Uc = self.Uc - (self.lr*self.dUc)
        self.bc = self.bc - (self.lr*self.dbc)

        self.Wo = self.Wo - (self.lr*self.dWo)
        self.Uo = self.Uo - (self.lr*self.dUo)
        self.bo = self.bo - (self.lr*self.dbo)

        self.Wy = self.Wy - (self.lr*self.dWy)
        self.by = self.by - (self.lr*self.dby)
        
    def train(self,EPOCHS):
        for epoch in range(EPOCHS):
            self.forward_propagation()
            print(self.calculate_cross_entropy())
            self.backpropagation()
            self.update()
    
    def predict(self,test):
        # test shape - Batch size, seq_len, embed_size
        Prediction = np.zeros_like(test)
        
        
        H_state = self.H_state[0]
        C_state = self.C_state[0]
        for i in range(test.shape[2]):
            Z1 = np.dot(test[:,i,:],self.Wf) + np.dot(H_state,self.Uf) + self.bf.T
            Forget_Gate = expit(Z1)
            
            Z2 = np.dot(test[:,i,:],self.Wi) + np.dot(H_state,self.Ui) + self.bi.T
            Input_Gate = expit(Z2)
            
            Z3 = np.dot(test[:,i,:],self.Wc) + np.dot(H_state,self.Uc) + self.bc.T
            candidates = np.tanh(Z3)
            
            C_state = Forget_Gate * C_state  + Input_Gate * candidates
            
            Z4 = np.dot(test[:,i,:],self.Wo) + np.dot(H_state,self.Uo) + self.bo.T
            Output_Gate = expit(Z4)
            
            Z5 = np.tanh(C_state)
            H_state = Output_Gate * Z5

            K = np.dot(H_state,self.Wy) + self.by.T
            Y[:,i,:] = softmax(K)
        return Y


import numpy as np
import matplotlib as plt
from maths_functions import *
from tqdm import *
from sklearn.preprocessing import OneHotEncoder
from skimage.feature import hog

class NeuralNetwork:
    def __init__(self, sizes):
        self.sizes      = sizes
        self.model      = {}
        self.b          = np.array([np.zeros((x, 1))  for x in sizes[1:]])
        self.d          = self.sizes[0]  
        self.s          = self.sizes[-1]      
        self.w          = np.array([(np.random.randn(x, y) / np.sqrt(x)) for x,y in zip(self.sizes[1:],self.sizes[:-1])])
        for w in self.w:
            w = np.random.uniform(-1/np.sqrt(w.shape[1]), 1/np.sqrt(w.shape[1]),(w.shape[0],w.shape[1]))

    def fprop(self,x):
        model = {}
        for j in range(1,len(self.sizes)):
            weight_j = 'W{}'.format(j)  
            bias_j = 'b{}'.format(j)
            model[weight_j] = self.w[j-1]
            model[bias_j] = self.b[j-1]

        model['a0'] = x
        for i in range(1,len(self.sizes)):
            z = model['W'+str(i)] @ model['a'+str(i-1)] + model['b'+str(i)]
            model['z'+str(i)] = z
            a = g_x(self.h_act,z) if i < (len(self.sizes)-1) else g_x(self.out_act,z) 
            model['a'+str(i)] = a
        
        self.model = model
       

    def bprop(self,y):
        delta3 = self.model['a'+str(len(self.sizes)-1)] - y
        self.model['delta'+str(len(self.sizes))] = delta3
                
        for j in range(len(self.sizes)-1,0,-1):
            a = self.model['a'+str(j-1)]
            z = self.model['z'+str(1)]
            delta = self.model['delta'+str(j+1)]
            self.model['grad_w'+str(j)] = delta @ a.T 
            #_lambda *    
            self.model['grad_b'+str(j)] = np.sum(delta, axis=1, keepdims=True)
            if (j>1):
                self.model['delta'+str(j)] = (delta.T@self.model['W'+str(j)]).T * (gPrime_x(self.h_act,z))
    	
    def SGD(self, data, epochs, minibatch_size, L1_lambda, L2_lambda, lrate, h_act, out_act, test, images):

        self.minibatch_size    = minibatch_size
        self.L1_lambda         = L1_lambda
        self.L2_lambda         = L2_lambda
        self.lrate             = lrate
        self.h_act             = h_act
        self.out_act           = out_act 

        train_inputs = data['train_inputs']
        train_labels = data['train_labels']
        test_data = data['test_inputs']

        self.onehot = self.onehot_encode(train_labels)

        if images :
            train_inputs = np.array(train_inputs)
            sqrt = int(np.sqrt(train_inputs.shape[1]))
            length = train_inputs.shape[0]
            train_inputs = train_inputs.reshape((length,sqrt,sqrt))
            train_inputs = self.images_preprocessing(train_inputs)
            test_inputs = self.images_preprocessing(test_inputs)

            if self.sizes[0] != train_inputs.shape[1]:
                self.sizes[0] = train_inputs.shape[1]
                self.d          = self.sizes[0]  
                self.b          = np.array([np.zeros((x, 1))  for x in self.sizes[1:]])
                self.w          = np.array([(np.random.randn(x, y) / np.sqrt(x)) for x,y in zip(self.sizes[1:],self.sizes[:-1])])
                for w in self.w:
                    w = np.random.uniform(-1/np.sqrt(w.shape[1]), 1/np.sqrt(w.shape[1]),(w.shape[0],w.shape[1]))

        for j in tqdm(range(epochs)):
    
            x = train_inputs
            y = self.onehot

            if not images :
                self.verif_gradient(x,y)
            data_left = len(train_inputs)%minibatch_size
            total_loss_per_epoch = 0

            for i in range(0, train_inputs.shape[0] -data_left, minibatch_size):
                x_minibatch = train_inputs[i:i+ minibatch_size]
                y_minibatch = self.onehot[i:i+ minibatch_size]
                total_loss_per_epoch += self.train_mini_Batch(x_minibatch,y_minibatch)


            if test :
                tqdm.write("Epoch {0}: {1} / {2}".format(
                    j, self.evaluate(test_data), onehot))
            else:
                tqdm.write("Epoch {0} complete : \nLoss {1} ".format(j,total_loss_per_epoch))


    def train_mini_Batch(self,x,y):
        L = 0
        for i in range(len(x)):
            x_i = x[i].reshape(self.d,1)
            y_i = y[i].reshape((self.s,1))
            self.fprop(x_i)
            self.bprop(y_i)
            self.update_param()
            L += self.loss_function(y_i)
        return L 
            
    def onehot_encode(self,y):
        y = y.reshape(len(y), 1)
        onehot_encoder = OneHotEncoder(categories='auto',sparse=False)
        return onehot_encoder.fit_transform(y)
    
    def verif_gradient(self,x,y,epsilon= 10**-5):
        for i in range(len(x)):
            x_i = x[i].reshape((self.d,1))
            y_i = y[i].reshape((self.s,1))
            self.fprop(x_i)
            L = self.loss_function(y_i)
            self.bprop(y_i)
            gradients_w = [self.model['grad_w'+str(i+1)] for i in range(len(self.w))]
            gradients_b = [self.model['grad_b'+str(i+1)] for i in range(len(self.w))]
            ratio_w = []
            ratio_b = []
            for k,matrix_weight in enumerate (self.w):
                shape = matrix_weight.shape
                for i in range(shape[0]):
                    for j in range(shape[1]):
                        matrix_weight[i][j] -= epsilon                    
                        self.fprop(x_i)
                        L2 = self.loss_function(y_i)
                        estimated_gradient = (L-L2)/epsilon
                        matrix_weight[i][j] += epsilon
                        ratio_w.append(((estimated_gradient+epsilon)/(gradients_w[k][i][j]+epsilon)))    
       
        print(ratio_w)

    def update_param(self):
        for i in range(len(self.w)-1,-1,-1):
             self.w[i] -= self.lrate * self.model['grad_w'+str(i+1)]
             self.b[i] -= self.lrate * self.model['grad_b'+str(i+1)]

    def loss_function(self,y):
        p = self.model['a'+str(len(self.sizes)-1)]
        x = np.sum(p*y, axis = 0)
        return(-np.log(x))

    def evaluate(self, test_data, test_labels):
        test_results = [(np.argmax(np.array(self.feedforward(x) for x in test_data)))]
        return sum(int(x == y) for x in test_results for y in test_labels)

    def images_preprocessing(self,images):
        list_hog_fd = []
        for feature in tqdm(images):
                fd = hog(feature, orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualize=False)
                list_hog_fd.append(fd)
        return np.array(list_hog_fd, 'float64')

    


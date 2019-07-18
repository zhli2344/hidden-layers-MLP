#!/usr/bin/env python
# coding: utf-8

# ## COMP5329 Assignment 1
# Members:
# 1. Zhuoyang Li (480164337)
# 2. Melissa Tan (200249191)
# 3. Jimmy Yue   (440159151)

# ### Achieved Modules
# More than one hidden layer [5]
# ReLU activation [5]
# Weight decay [5]
# Momentum in SGD [5]
# Dropout [5]
# Softmax and cross-entropy loss [5]
# Mini-batch training [5]
# Batch Normalization [5]

# ## Instruction to run
# 
# 1. Change the path to the location where the 'train_128.h5', 'train_label.h5', 'test_128.h5' resides. If running on colab, place data at  "/content/drive"
# 2. Click on run all to execute the entire code (Takes approx 30 mins including hyperparatemer tuning & 3 fold cross validation)

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import h5py
import random
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('ggplot')

seed= 42
fold = 3 # Number of cross validation fold

#path = '/home/ubuntoo/Documents/python/COMP5329/' # melissa's drive
#Inpath = '/content/drive/My Drive/'  # colab drive
#Outpath = Inpath
Inpath = '/Users/ningwang/Desktop/LZY/lzy1/DL-ASS1/Input/'  # soleil's drive
Outpath ='/Users/ningwang/Desktop/LZY/lzy1/DL-ASS1/Output/' # 


# Helper method to track time elapse
from time import time                                             
def timeit(method):
    def timed(*args, **kw):
        ts = time()
        result = method(*args, **kw)
        te = time()
        print('%r took %2.2f sec' % (method.__name__,  te-ts))
        return result
    return timed

# ### Load Data
#from google.colab import drive
#drive.mount('/content/drive')
def fileIO(Inpath):
    with h5py.File(Inpath + 'train_128.h5','r') as H: 
        trainSetX = np.copy(H['data'])
    print("The shape of train instance is:",trainSetX.shape)
    
    with h5py.File(Inpath + 'train_label.h5','r') as H:
        trainSetY = np.copy(H['label'])
    print("\nThe shape of label is:",trainSetY.shape)
    
    with h5py.File(Inpath + 'test_128.h5','r') as H:
        testSetX = np.copy(H['data'])
    print("\nThe shape of test dataset is:",testSetX.shape)
    
    return trainSetX,trainSetY,testSetX

print('==================Load Dataset========================\n')
trainSetX,trainSetY,testSetX = fileIO(Inpath)
print('\nLoad Successful')

# ### Check Class Balance
# Each class is evenly distributed.
print('\n==================Check Class Balance=================')
plt.figure(figsize=(8,6))
sns.set(style='whitegrid')
ax_label = sns.countplot(x = trainSetY, palette="cool")
ax_label.set(ylabel = 'Count', xlabel = 'Label')
plt.show()
print('Each class is evenly distributed.')

# ### Basic Function 
# Normlize dataset and check our createKfold function with scikit-learn k-fold cross-validation.
# Just for ensure our DIY split data function is correct and fair enough.

# Normalize dataset
@timeit
def standardize(x):
    return (x - x.mean(axis=0)) / x.std(axis=0)  
  
# Create crossvalidation fold
def createKFolds(X, folds=fold, seed=seed, shuffle=True):
    n = len(X)
    foldSize = n // folds
    remainder = n % folds
    idx = np.arange(n)
    if shuffle:
        np.random.seed(seed)
        np.random.shuffle(idx)    
    startIdx =  0
    kfolds = []
    for i in range(folds):
        if remainder > 0 :
            orphan = 1 
            remainder -= 1
        else:
            orphan = 0        
        endIdx = startIdx + foldSize + orphan
        y = idx[startIdx:endIdx]
        mask = np.ones(n, dtype=bool) 
        mask[y] = False
        x = idx[mask]
        startIdx = endIdx 
        kfolds.append((x,y))
    return kfolds


# Scaled dataset
print('\n==================Scaled Dataset======================')
trainSetX_scaled = standardize(trainSetX)
# Only use when predict the test dataset result, only method==0,the X_test will be used and predict label.
X_test = standardize(testSetX)
print('\nScaled complete')

# Scikit-Learn is ONLY used for testing the implementation of the createKFolds function.
# Scikit-Learn is NOT used in any other part of this notebook.
print('\n===========check our createKfold function with scikit-learn k-fold cross-validation===============')
# scikit-learn k-fold cross-validation
from sklearn.model_selection import KFold
# prepare cross validation
shuffle=True
np.random.seed(seed)
kfold = KFold(fold, shuffle, seed)

# enumerate splits
sk_train = []
sk_test = []
print('sk_learn kfold function')
for trainIdx, valIdx in kfold.split(trainSetX_scaled):
    #print(train, test)
    sk_train.append(trainIdx.tolist())
    sk_test.append(valIdx.tolist())
    print(len(trainIdx), len(valIdx), trainIdx, valIdx)
    
# compare with DIY kfold
diy_train = []
diy_test = []
print('\ndiy createKFolds function')
for trainIdx, testIdx in createKFolds(trainSetX_scaled, seed=seed, folds=fold, shuffle=shuffle):
    #print(trainIdx, testIdx)
    diy_train.append(trainIdx.tolist())
    diy_test.append(testIdx.tolist())
    print(len(trainIdx), len(valIdx), trainIdx, valIdx)

print('\nadd the sort to make sure that even after shuffling, the index assigned are the same\n')
print('sk_train == diy_train : ' , sk_train.sort() == diy_train.sort())
print('sk_test == diy_test   : ', sk_test.sort() == diy_test.sort())

print('\nThe above test verified that the function "createKFolds" generates the same indexes as per scikit learn kfold function. As such, the remaining codes will use the createKFolds function to split the training dataset to 3 folds.Because we are not allowed to use other libraries.')

# ### Activation function
def relu_forward(x):
    out = None
    out = np.copy(x)
    out[out<0] = 0
    cache = x
    return out, cache

def relu_backward(dout, cache):
    dx, x = None, cache
    dx = np.copy(dout)
    dx[x<0] = 0
    return dx

def layer_forward(X, w, b):
    out = None
    N = X.shape[0]
    x_temp = X.reshape(N,-1)
    out = x_temp.dot(w) + b
    cache = (X, w, b)
    return out, cache

def layer_backward(dout, cache):
    X, w, b = cache
    dx, dw, db = None, None, None
    db = np.sum(dout, axis = 0)
    x_temp = X.reshape(X.shape[0],-1)
    dw = x_temp.T.dot(dout)
    dx = dout.dot(w.T).reshape(X.shape)
    return dx, dw, db

def layer_Relu_forward(x,w,b):
    a,fc_cache=layer_forward(x,w,b)
    out,relu_cache=relu_forward(a)
    cache=(fc_cache,relu_cache)
    return out,cache

def layer_Relu_backward(dout,cache):
    fc_cache,relu_cache=cache
    da=relu_backward(dout,relu_cache)
    dx,dw,db=layer_backward(da,fc_cache)
    return dx,dw,db

def layer_BN_Relu_forward(x , w , b, gamma, beta, bn_param):
    a, fc_cache = layer_forward(x, w, b)
    bn, bn_cache = BN_forward(a, gamma, beta, bn_param)
    out, relu_cache = relu_forward(bn)
    cache = (fc_cache, bn_cache, relu_cache)
    return out, cache

def layer_BN_Relu_backward(dout, cache):
    fc_cache, bn_cache, relu_cache = cache
    dbn = relu_backward(dout, relu_cache)
    da, dgamma, dbeta =  BN_backward(dbn, bn_cache)
    dx, dw, db = layer_backward(da, fc_cache)
    return dx, dw, db, dgamma, dbeta


# ### Dropout function
def dropout_forward(x, dropout_param):
    p, mode = dropout_param['p'], dropout_param['mode']
    if 'seed' in dropout_param:  
        np.random.seed(dropout_param['seed'])
    mask = None
    out = None
    if mode == 'training':    
        mask = (np.random.rand(*x.shape) >= p) / (1-p)    
        out = x * mask
    elif mode == 'testing':
        out = x
    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)
    return out, cache


def dropout_backward(dout, cache):
    dropout_param, mask = cache
    mode = dropout_param['mode']
    dx = None
    if mode == 'training':    
        dx = dout * mask
    elif mode == 'testing':    
        dx = dout
    return dx


# ### Batch Normlization (BN)
def BN_forward(x, gamma, beta, bn_param):
    mode = bn_param['mode']
    eps = bn_param.get('eps', 1e-5)
    momentum = bn_param.get('momentum', 0.9)
    N, D = x.shape
    running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))

    out, cache = None, None
    if mode == 'training':    
        sample_mean = np.mean(x, axis=0, keepdims=True)       # [1,D]    
        sample_var = np.var(x, axis=0, keepdims=True)         # [1,D] 
        x_normalized = (x - sample_mean) / np.sqrt(sample_var + eps)    # [N,D]    
        out = gamma * x_normalized + beta    
        cache = (x_normalized, gamma, beta, sample_mean, sample_var, x, eps)    
        running_mean = momentum * running_mean + (1 - momentum) * sample_mean    
        running_var = momentum * running_var + (1 - momentum) * sample_var
    elif mode == 'testing':    
        x_normalized = (x - running_mean) / np.sqrt(running_var + eps)    
        out = gamma * x_normalized + beta
        
    # Store the updated running means back into bn_param
    bn_param['running_mean'] = running_mean
    bn_param['running_var'] = running_var

    return out, cache

def BN_backward(dout, cache):
    dx, dgamma, dbeta = None, None, None
    x_normalized, gamma, beta, sample_mean, sample_var, x, eps = cache
    N, D = x.shape
    dx_normalized = dout * gamma       # [N,D]
    x_mu = x - sample_mean             # [N,D]
    sample_std_inv = 1.0 / np.sqrt(sample_var + eps)    # [1,D]
    dsample_var = -0.5 * np.sum(dx_normalized * x_mu, axis=0, keepdims=True) * sample_std_inv**3
    dsample_mean = -1.0 * np.sum(dx_normalized * sample_std_inv, axis=0, keepdims=True) - 2.0 * dsample_var * np.mean(x_mu, axis=0, keepdims=True)
    dx1 = dx_normalized * sample_std_inv
    dx2 = 2.0/N * dsample_var * x_mu
    dx = dx1 + dx2 + 1.0/N * dsample_mean
    dgamma = np.sum(dout * x_normalized, axis=0, keepdims=True)
    dbeta = np.sum(dout, axis=0, keepdims=True)

    return dx, dgamma, dbeta


# ### Softmax and Cross Entropy
def softmax_loss(x, y):
    probs = np.exp(x - np.max(x, axis=1, keepdims=True))
    probs /= np.sum(probs, axis=1, keepdims=True)
    N = x.shape[0]
    loss = -np.sum(np.log(probs[np.arange(N), y])) / N
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N
    return loss, dx
   
# ### SGD/SGDMomentum/Adam
def sgd(w, dw, search=None):
    if search is None: search = {}
    search.setdefault('learning_rate', 1e-2)
    w -= search['learning_rate'] * dw
    return w, search

def sgd_momentum(w, dw, search=None):
    if search is None: search = {}
    search.setdefault('learning_rate', 1e-2)
    search.setdefault('momentum', 0.9)
    v = search.get('velocity', np.zeros_like(w))
    next_w = None 
    next_w = w
    v = search['momentum']* v - search['learning_rate']*dw
    next_w +=v
    search['velocity'] = v
    return next_w, search

def adam(x, dx, search=None):
    if search is None: search = {}
    search.setdefault('learning_rate', 1e-2)
    search.setdefault('beta1', 0.9)
    search.setdefault('beta2', 0.999)
    search.setdefault('epsilon', 1e-8)
    search.setdefault('m', np.zeros_like(x))
    search.setdefault('v', np.zeros_like(x))
    search.setdefault('t', 0)
    next_x = None
    search['t']+=1 
    search['m'] = search['beta1']*search['m'] + (1- search['beta1'])*dx
    search['v'] = search['beta2']*search['v'] + (1- search['beta2'])*(dx**2)   
    mb = search['m']/(1-search['beta1']**search['t'])
    vb = search['v']/(1-search['beta2']**search['t'])
    next_x = x -search['learning_rate']* mb / (np.sqrt(vb) + search['epsilon'])
    return next_x, search


# ### Modularized MLP
print('\n========================Load Modularized MLP======================')
print('\nFully modularized MLP with support to dropout and batch normalization.')
print('Here we actually try two initialization method.One is random initialization, other is He initialization.')
print('The final predicted label with these two method are same totally.Accuracy are similar too.')
print('BUT the special difference is that: when we use random init,the loss is large and converges badly. When we use He init, the loss is smooth and almost does not diverge!')

# #### MLP structure
class MLP(object):
    def __init__(self, hidden_dims, input_dim=128, reg=0.0, 
                 output_dim=10,use_batchnorm=False,dropout=0,      
                 dtype=np.float32, seed=None):

        self.use_batchnorm = use_batchnorm
        self.use_dropout = dropout > 0
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        inputs = input_dim
        for i,hl in enumerate(hidden_dims):
            #random init
            #self.params['W%d'%(i+1)]=weight_scale*np.random.randn(inputs,hl)
            self.params['W%d'%(i+1)]=np.random.randn(inputs,hl)*np.sqrt(2./inputs)#He init
            self.params['b%d'%(i+1)]=np.zeros((1,hl))
            if self.use_batchnorm:
                self.params['gamma%d' %(i+1)] = np.ones((1,hl))        
                self.params['beta%d' %(i+1)] = np.zeros((1,hl))
            inputs = hl
            self.params['W%d'%(self.num_layers)]=np.random.randn(inputs,output_dim)*np.sqrt(2./inputs)
            self.params['b%d'%(self.num_layers)]=np.zeros((1,output_dim))
        
        self.dropout_param = {}
        if self.use_dropout:    
            self.dropout_param = {'mode': 'training', 'p': dropout}    
            if seed is not None:        
                self.dropout_param['seed'] = seed       
                
        self.bn_params = []
        if self.use_batchnorm:
            self.bn_params = [{'mode': 'training'} for i in range(self.num_layers - 1)]
        
        self.dropout_param = {}
        if self.use_dropout:    
            self.dropout_param = {'mode': 'training', 'p': dropout}    
            if seed is not None:        
                self.dropout_param['seed'] = seed           
        for k, v in self.params.items():    
            self.params[k] = v.astype(dtype)

    def nn(self, X, y=None):    
        X = X.astype(self.dtype)    
        mode = 'testing' if y is None else 'training' 
        if self.dropout_param is not None: 
            self.dropout_param['mode'] = mode    
        if self.use_batchnorm:
            for bn_param in self.bn_params:            
                bn_param['mode'] = mode    
        scores = None    
        cache1,cache2 = {}, {}    
        n_in = X
        
        for L in range(self.num_layers-1): 
            if self.use_batchnorm:
                n_in,cache1[L]=layer_BN_Relu_forward(n_in,self.params['W%d'%(L+1)],self.params['b%d'%(L+1)],
                                    self.params['gamma%d' % (L + 1)],self.params['beta%d' % (L + 1)],
                                    self.bn_params[L])
                #print(L)    
                
            else:
                n_in, cache1[L] =layer_Relu_forward(n_in, self.params['W%d' % (L + 1)],
                                                            self.params['b%d' % (L + 1)])
            if self.use_dropout:
                n_in, cache2[L] =dropout_forward(n_in, self.dropout_param)
                
        n_out, cache1[self.num_layers] = layer_forward(n_in,self.params['W%d' % (self.num_layers)],
                                                           self.params['b%d' % (self.num_layers)])
        #probs = softmax(n_out)
        
        if mode == 'testing':   
            return n_out
        
        #weight decay
        loss = 0.0
        grads = {}
        loss,dscores = softmax_loss(n_out,y)
        dhout = dscores
        loss =loss+0.5*self.reg*np.sum(self.params['W%d' % (self.num_layers)] * self.params['W%d' %(self.num_layers)])
    
        dx, dw, db = layer_backward(dhout, cache1[self.num_layers])
        
        grads['W%d' % (self.num_layers)] = dw + self.reg *self.params['W%d' % (self.num_layers)]
        grads['b%d' % (self.num_layers)] = db
        
        dhout = dx 
        # Backward pass: compute gradients
        t = self.num_layers-1
        for w in range(t):
            L = t - w - 1
            loss = loss + 0.5 * self.reg * np.sum(self.params['W%d' %(L + 1)] * self.params['W%d' % (L + 1)])
            if self.use_dropout:
                dhout = dropout_backward(dhout, cache2[L])
            if self.use_batchnorm:
                dx, dw, db, dgamma, dbeta =layer_BN_Relu_backward(dhout, cache1[L])
                grads['gamma%d' % (L + 1)] = dgamma
                grads['beta%d' % (L + 1)] = dbeta
            else:
                dx, dw, db = layer_Relu_backward(dhout, cache1[L])               
            
            grads['W%d' % (L + 1)] = dw + self.reg * self.params['W%d'% (L + 1)]
            grads['b%d' % (L + 1)] = db
            dhout = dx
        return loss, grads
    
# #### Package function
class Run(object):
    def __init__(self,model,X_train,y_train,X_val,y_val,X_test,**kwargs):
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.X_test = X_test
        self.optim_param = kwargs.pop('optim_param', {})
        self.lr_decay = kwargs.pop('lr_decay', 1.0)
        self.batch_size = kwargs.pop('batch_size', 100)
        self.num_epochs = kwargs.pop('num_epochs', 10)
        self.print_every = kwargs.pop('print_every', 100)
        self.verbose = kwargs.pop('verbose', True)      
        self._reset()       
        
    def _reset(self):
        self.epoch = 0
        self.best_val_acc = 0
        self.best_params = {}
        self.loss_history = []
        self.train_acc_history = []
        self.val_acc_history = []
        self.optim_params = {}
        for p in self.model.params:
            d = {k: v for k, v in self.optim_param.items()}
            self.optim_params[p] = d
    
    def _fit1(self):
        # Make a minibatch of training data        
        num_train = self.X_train.shape[0]
        batch_mask = np.random.choice(num_train, self.batch_size)
        X_batch = self.X_train[batch_mask]
        y_batch = self.y_train[batch_mask]
        # Compute loss and gradient
        loss, grads = self.model.nn(X_batch, y_batch)
        self.loss_history.append(loss)
        # Perform a parameter update
        for p, w in self.model.params.items():
            dw = grads[p]
            param = self.optim_params[p]
            next_w, next_param = sgd(w,dw,param)
            self.model.params[p] = next_w
            self.optim_params[p] = next_param
            
    def _fit2(self):
        # Make a minibatch of training data        
        num_train = self.X_train.shape[0]
        batch_mask = np.random.choice(num_train, self.batch_size)
        X_batch = self.X_train[batch_mask]
        y_batch = self.y_train[batch_mask]
        # Compute loss and gradient
        loss, grads = self.model.nn(X_batch, y_batch)
        self.loss_history.append(loss)
        #parameter update
        for p, w in self.model.params.items():
            dw = grads[p]
            param = self.optim_params[p]
            next_w, next_param = sgd_momentum(w,dw,param)
            self.model.params[p] = next_w
            self.optim_params[p] = next_param
        
    def _fit3(self):
        # Make a minibatch of training data        
        num_train = self.X_train.shape[0]
        batch_mask = np.random.choice(num_train, self.batch_size)
        X_batch = self.X_train[batch_mask]
        y_batch = self.y_train[batch_mask]     
        # Compute loss and gradient
        loss, grads = self.model.nn(X_batch, y_batch)
        self.loss_history.append(loss)
        # parameter update
        for p, w in self.model.params.items():
            dw = grads[p]
            param = self.optim_params[p]
            next_w, next_param = adam(w,dw,param)
            self.model.params[p] = next_w
            self.optim_params[p] = next_param

    def check_accuracy(self, X, y,num_samples=None,batch_size=100):
        N = X.shape[0]
        if num_samples is not None and N > num_samples:
            mask = np.random.choice(N, num_samples)
            N = num_samples
            X = X[mask]
            y = y[mask]      
        num_batches = N / batch_size
        if N % batch_size != 0:
            num_batches += 1
        y_pred = []
        num_batches=int(num_batches)
        for i in range(num_batches):
            start = i * batch_size
            end = (i + 1) * batch_size
            scores = self.model.nn(X[start:end])
            y_pred.append(np.argmax(scores, axis=1))
        y_pred = np.hstack(y_pred)
        acc = np.mean(y_pred == y)         
        return acc
    
    def output_predictLabel(self,X):
        scores = self.model.nn(X)
        pl = np.array(scores)
        label = (pl==pl.max(axis=1,keepdims=1))
        label = [np.where(l==1)[0][0]for l in label]
        return label
        
    @timeit
    def processing(self,method):
        if method ==0:
            print('Predict the test data and save to file')
            predict_label = self.output_predictLabel(self.X_test)
            output_data = pd.DataFrame(predict_label)
            output_data.to_csv(Outpath+"Predicted_labels.h5",sep=',',index=False,header=False)
            return output_data
        num_train = self.X_train.shape[0] 
        iterations_per_epoch = max(num_train / self.batch_size, 1) 
        num_iterations = self.num_epochs * iterations_per_epoch
        num_iterations = int(num_iterations)
        for t in range(num_iterations): 
            if method ==1: self._fit1() 
            elif method ==2:self._fit2()
            elif method ==3:self._fit3()  
            if self.verbose and t % self.print_every == 0:
                print ('(Iteration %d / %d) loss: %f' % (t + 1, num_iterations, self.loss_history[-1]))

            epoch_end = (t + 1) % iterations_per_epoch == 0 
            if epoch_end: 
                self.epoch += 1 
                for k in self.optim_params:  
                    self.optim_params[k]['learning_rate']*= self.lr_decay

            first_it = (t == 0) 
            last_it = (t == num_iterations + 1) 
            if first_it or last_it or epoch_end: 
                train_acc = self.check_accuracy(self.X_train, self.y_train,
                                        num_samples=1000)
                val_acc = self.check_accuracy(self.X_val, self.y_val)
                self.train_acc_history.append(train_acc)
                self.val_acc_history.append(val_acc)

                if self.verbose:
                    print ('(Epoch %d / %d) train acc: %f; val_acc: %f' % (self.epoch, self.num_epochs, train_acc, val_acc))
                if val_acc > self.best_val_acc:
                    self.best_val_acc = val_acc
                    self.best_params = {}
                    for k, v in self.model.params.items():
                        self.best_params[k] = v.copy()

        # At the end of training swap the best params into the model
        self.model.params = self.best_params
        #print(self.best_params)
        if method ==1:print('SGD')
        if method ==2:print('SGD momentum')
        if method ==3:print('Adam')
            
            
# ### Regularization method comparison

# #### Dropout rate
# The premise behind Dropout is to introduce some noise into each hidden layer. Drop out is used only during training. 
# 0.1，0.3，0.5
print('\n================Model 1 : Only Dropout[0.1,0.3,0.5]=============================')

@timeit
def dropout_Benchmark(optimizationMethod=2):#default 2 is SGD momentum
    dropouts = [0.1,0.3,0.5]
    train_acc=[]
    val_acc=[]
    loss_history = []

    for i, dropout in enumerate(dropouts):
        model=MLP([100,60],dropout=dropout)
        print('Dropout :',dropout)

        cv_train_acc=[]
        cv_val_acc=[]

        for trainIdx, valIdx in createKFolds(trainSetX_scaled, folds=fold):
            X_train = trainSetX_scaled[trainIdx]
            y_train = trainSetY[trainIdx]
            X_val = trainSetX_scaled[valIdx]
            y_val = trainSetY[valIdx]
            run=Run(model,X_train,y_train,X_val,y_val,X_test,num_epochs=60,batch_size=64,
                      optim_param={'learning_rate':5e-4},verbose=False,print_every=1000)        
            run.processing(optimizationMethod) 

            # store accuracy
            cv_train_acc.append(run.train_acc_history)
            cv_val_acc.append(run.val_acc_history)

        train_acc.append(np.mean(cv_train_acc, axis=0))
        val_acc.append(np.mean(cv_val_acc, axis=0))
        loss_history.append(run.loss_history)
        t = np.mean(train_acc,axis=0)
        print('average train acc with only Dropout:',np.mean(t))
        v = np.mean(val_acc, axis=0)
        print('average val acc with only Dropout:',np.mean(v))
        l = np.mean(loss_history,axis=0)
        print('average training loss with only Dropout:',np.mean(l))
            
    plt.subplot(311)
    for i, dropout in enumerate(dropouts):
        plt.plot(train_acc[i],'-',label='%.2f dropout'% dropout)
    plt.title('Train accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1.0)
    plt.legend(ncol=4,loc='lower left')

    plt.subplot(312)
    for i, dropout in enumerate(dropouts):
        plt.plot(val_acc[i],'-',label='%.2f dropout'% dropout)
    plt.title('Validation accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(ncol=4, loc='lower left')
    plt.ylim(0, 1.0)
    plt.gcf().set_size_inches(15, 15)
    plt.show();
    
# SGD
dropout_Benchmark(1)
# SGD Momentum
dropout_Benchmark(2)
# Adam
dropout_Benchmark(3)



# #### BN Performance
print('\n================Model 2 : Only BatchNorm=============================')
@timeit
def bn_Benchmark(optimizationMethod=2):
    print('Apply Batch Normalization without Dropout')
    train_acc=[]
    val_acc=[]
    loss_history = []

    model=MLP([100,60],use_batchnorm=True)
    for trainIdx, valIdx in createKFolds(trainSetX_scaled, folds=fold):
        X_train = trainSetX_scaled[trainIdx]     
        y_train = trainSetY[trainIdx]
        X_val = trainSetX_scaled[valIdx]
        y_val = trainSetY[valIdx]      

        run=Run(model,X_train,y_train,X_val,y_val,X_test,num_epochs=60,batch_size=64,
                      optim_param={'learning_rate':5e-4},verbose=False,print_every=1500)        
        run.processing(optimizationMethod) 

        # store accuracy
        train_acc.append(run.train_acc_history)
        val_acc.append(run.val_acc_history)
        loss_history.append(run.loss_history)

    t = np.mean(train_acc,axis=0)
    print('average train acc with only BN:',np.mean(t))
    v = np.mean(val_acc, axis=0)
    print('average val acc with only BN:',np.mean(v))
    l = np.mean(loss_history,axis=0)
    print('average training loss with only BN:',np.mean(l))

    plt.subplot(311)
    plt.plot(np.mean(loss_history,axis=0),'o')
    plt.title('Training loss history')
    plt.xlabel('Iteration')
    plt.ylabel('Training loss')
    plt.subplot(312)

    x_axis = np.arange(1,len(run.train_acc_history)+1)
    plt.plot(x_axis, np.mean(train_acc,axis=0), 'r', label='Training Accuracy')
    plt.legend()
    plt.plot(x_axis, np.mean(val_acc, axis=0), 'g', label='Validation Accuracy')
    plt.legend()
    plt.xlabel('Training epoch')
    plt.ylabel('Accuracy')
    plt.gcf().set_size_inches(15,15)
    plt.show()

# SGD
bn_Benchmark(1)
# SGD Momentum
bn_Benchmark(2)
# Adam
bn_Benchmark(3)


# #### BN & Dropout together
print('\n================Model 3 : BN & Dropout[0.5]=============================')
@timeit
def bndo_Benchmark(optimizationMethod=2):
    print('Apply both Dropout and Batch Normalisation')
    train_acc=[]
    val_acc=[]
    loss_history = []

    model = MLP([100,60],dropout=0.5,use_batchnorm=True)

    for trainIdx, valIdx in createKFolds(trainSetX_scaled, folds=fold):
        X_train = trainSetX_scaled[trainIdx]     
        y_train = trainSetY[trainIdx]
        X_val = trainSetX_scaled[valIdx]
        y_val = trainSetY[valIdx]      

        run=Run(model,X_train,y_train,X_val,y_val,X_test,num_epochs=60,batch_size=64,
                  optim_param={'learning_rate':5e-4},verbose=False,print_every=1000)        
        run.processing(optimizationMethod) 

        # store accuracy
        train_acc.append(run.train_acc_history)
        val_acc.append(run.val_acc_history)
        loss_history.append(run.loss_history)

    t = np.mean(train_acc,axis=0)
    print('average train acc with BN&Dropout:',np.mean(t))
    v = np.mean(val_acc, axis=0)
    print('average train acc with BN&Dropout:',np.mean(v))
    l = np.mean(loss_history,axis=0)
    print('average training loss with BN&Dropout:',np.mean(l))
    
    plt.subplot(311)
    plt.plot(np.mean(loss_history,axis=0),'*')
    plt.title('Training loss history')
    plt.xlabel('Iteration')
    plt.ylabel('Training loss')
    plt.subplot(312)
    x_axis = np.arange(1,len(run.train_acc_history)+1)
    plt.plot(x_axis, np.mean(train_acc,axis=0), 'r', label='Training Accuracy')
    plt.legend()
    plt.plot(x_axis, np.mean(val_acc, axis=0), 'g', label='Validation Accuracy')
    plt.legend()
    plt.xlabel('Training epoch')
    plt.ylabel('Accuracy')
    plt.gcf().set_size_inches(15,15)
    plt.show();

# SGD
bndo_Benchmark(1)
# SGD Momentum
bndo_Benchmark(2)
# Adam
bndo_Benchmark(3)


print('\nSo by compare with "Only Dropout","Only BN" and "BN & Dropout", we found that "Only BN" performed better. Then we decided to use "Only BN" to train our model and make Hyperparametric tuning for it.')


print('\n==================Hyperparametric tuning=======================')

print('In this section, we opt to use Adam as the training time is significantly faster than SGD Momentum. We also experimented with various combination of parameters.')
print('Comparing the loss chart, we discovered a trend :\n')
print('BN can be regarded as a constraint on the input sample. The biggest role is to accelerate convergence, reduce the model dependence on dropout as well as careful weight initialnization,the advantage of BN is it allow model to adopt higher learning rate, which can results in the convergence speed up.\n')
print('But for the simplicity of the code and the flexibility of the final run time, we have not shown the process of all the adjustments.Here we only show the range of parameters we finally delineated, that is:\n')
print('Learning Rate:')
print('-Learning Rate is too small (such as 1e-5), the cost drops very slowly')
print('-The Learning Rate is too large (such as 1e-2), and the cost growth explosion (cur cost > 3* original cost')
print('-Suitable from 1e-3 to 1e-4\n')

print("\nStart Params tuning from those parameter lists:")
np.random.seed(seed)
learning_rates = [5.6678923e-4,4.93056310e-03]#10**np.random.uniform(-7,-2,4)[1:]
weight_decay = [1e-4,1e-6]
learning_rates_decay = [0.91,0.95]
print('learning_rates',learning_rates)
print('weight_decay',weight_decay)
print('learning_rates_decay',learning_rates_decay)
#but actually I have run 72 times(24 kinds of params) to find the good range of params, to save time, just list 8 kinds of them.
print('\nWill totally run 24 (8 kinds of params combination * 3 folds cv) times to find best one\n')

train_acc=[]
val_acc=[]
loss_history = []

results = {}
best_val = -1
best_model = None

start = time()
for lr in learning_rates:
    for wd in weight_decay:
         for ld in learning_rates_decay:
            final_model = MLP([100,60],reg=wd,use_batchnorm = True)
            cv_train_acc=[]
            cv_val_acc=[]
            cv_loss_history=[]    
            for trainIdx, valIdx in createKFolds(trainSetX_scaled, folds=fold):
                X_train = trainSetX_scaled[trainIdx]
                y_train = trainSetY[trainIdx]
                X_val = trainSetX_scaled[valIdx]
                y_val = trainSetY[valIdx]                 
                final_run = Run(final_model,X_train, y_train, X_val,y_val,X_test,
                            print_every=1000, num_epochs=60, batch_size=64,
                            optim_param={'learning_rate': lr,},lr_decay = ld,verbose = False)          
                final_run.processing(3)#Adam
                # store accuracy
                cv_train_acc.append(final_run.train_acc_history)
                cv_val_acc.append(final_run.val_acc_history)
                cv_loss_history.append(final_run.loss_history)
                
            train_acc.append(np.mean(cv_train_acc, axis=0))
            val_acc.append(np.mean(cv_val_acc, axis=0))
            loss_history.append(np.mean(cv_loss_history, axis=0))
                
            plt.figure(figsize=(20,8))
            for i in range(1,len(loss_history)+1):                    
                plt.plot(loss_history[i-1],'*',label='%ith kind combination'%i)
                label = f'Training loss history                         with lr:{lr:.6f}, wd:{wd:.7f},ld:{ld:.3f},                        Train | Valid Acc: {np.mean(train_acc[i-1], axis=0):.3f} | {np.mean(val_acc[i-1],0):.3f}'         
                plt.title(label)
                plt.xlabel('Iteration')
                plt.ylabel('Training loss')
                plt.legend(ncol=6, loc='upper right')
            plt.show();
            results[(lr,wd,ld)] = train_acc,val_acc

end = time()
print ("\nParams tuning time:%2f minutes "%((end - start)/60))

# Print out results for easy comparison
i = 0
for lr,wd,ld in results:
    train_acc,val_acc = results[(lr,wd,ld)]
    print (f'lr:{lr:.6f},wd:{wd:.7f},ld:{ld:.3f},Train| Valid Acc: {np.mean(train_acc[i]*100, axis=0):.3f}% | {np.mean(val_acc[i]*100,0):.3f}%')
    i +=1

print('\nThe above results proved our conclusion again, after apply BN,we can choose relatively higher learning rate. And weight decay params effect the training time.')


print('\nAlso refer to above result we can see that:')
print('The model with:')
print('learning_rate 4.93056310e-03')
print('weight_decay 1e-6')
print('learning_rate_decay 0.95')
print('can perform best: Train| Valid Acc: 97.917% | 95.296%')
print("\nAt the end of training, the Run fuction swap the best params(w,b,gamma,beta) into the model, and the best Hyperparameters we found happened to be the last set, so we didn't have to pick the params then run the model again, because now the model is saving the last set of Hyperparameters! Now we just need to use this model to predict the X_test label directly!\n")

print('A quick check of the Hyperparameters in the our final model, just for ensure')
print(final_run.optim_param)
print('weight decay:',final_model.reg)
print('learning rate decay:',final_run.lr_decay)

import pandas as pd 
predictL = final_run.processing(0)#0 refer to predict label for test data set without training.

print('\n=========================Predicted label of test set=================================')
print(predictL)

print('\nCheck the predicted label distribution')
import pandas as pd  
SnapYourfingers = pd.read_csv(Outpath+'Predicted_labels.h5',header=None)
s = np.array(SnapYourfingers)
plt.figure(figsize=(8,6))
sns.set(style='whitegrid')
ax_label = sns.countplot(x = s[:,0], palette="cool")
ax_label.set(ylabel = 'Count', xlabel = 'Label')
plt.show()

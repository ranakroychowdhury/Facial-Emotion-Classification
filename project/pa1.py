#!/usr/bin/env python
# coding: utf-8

# In[295]:


import numpy as np
import matplotlib.pyplot as plt
from dataloader import *
from logistic_reg_batch import logistic_reg_batch
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[224]:


def get_folds(data, k, ohe):
    """
    Takes a dataset, k and returns k mutually exclusive batches of the data
    """
    assert k > 2, 'k should be greater than 2'
    arr = accumulate(data, ohe)
    np.random.shuffle(arr)
    folds = [[] for _ in range(k)]
    for idx,data in enumerate(arr):
        folds[idx%k].append(data)
    return folds


# In[203]:


def pca(images,k):
    #Center the images
    flat_images = images - images.mean(0)
        
    #Create A and A_transpose Matrices
    flat_images = np.array(flat_images)
    A = flat_images.T
    A_transpose = flat_images
    
    #Scale the transformation
    ATA = np.matmul(A_transpose,A) * 1/len(A_transpose)
    
    #Calculate the EigenVectors
    eigVals,eigVecs = np.linalg.eig(ATA)
    
    #Sort by largest Principal Component
    idx = eigVals.argsort()[::-1]   
    eigVals = eigVals[idx]
    
    PCs = np.matmul(A,eigVecs)

    # Get k pc's
    pc = PCs[:, 0:k]
    return pc,images.mean(0),eigVals


def projectImages(images, mean, pc, eigVals):
    k = len(pc[0, :])

    # Center the images
    flat_images = images = images - mean

    # Flatten the images
    A_transpose = flat_images

    norm = np.linalg.norm(pc, axis=0)
    pc = pc / norm

    projection = np.matmul(A_transpose, pc)

    # Normalize
    projection = projection / eigVals[0:k] ** .5

    return projection


# In[543]:


def kfold(data, model, kfold, ohe, stochastic, k, p, epochs, lr, name):
    """
    Performs the K-Fold cross validation algorithm noted in writeup
    data -> list of images
    model -> Function
    k -> Number of folds
    p -> Number of principal components
    epochs -> Number of epochs
    n_classes -> Number of unique classes in data
    lr -> Learning rate
    """
    t_losses = [] # k x epochs
    v_losses = [] # k x epochs
    t_accs = [] # k x epochs
    v_accs = [] # k x epochs
    best_te_loss = [] # k x 1
    best_te_acc = [] # k x 1
    n = 1 # Internal implementation for OHE
    if ohe:
        n = len(data.keys())
    folds = get_folds(data, k, ohe)
    
    loops = k if kfold else 1
    
    for i in range(0, loops):
        val, test = folds[i], folds[(i+1)%k]
        train = []
        for j in range(2, k):
            train += folds[(i+j)%k]
        train = np.array(train)
        val = np.array(val)
        test = np.array(test)
        X_train = train[:,:-1]
        y_train = train[:,-n:]
        X_val = val[:,:-1]
        y_val = val[:,-n:]
        X_test = test[:,:-1]
        y_test = test[:,-n:]

        # PCA
        training_pca_axes, training_mean, training_eigenvalues  = pca(X_train, p)
        X_train = projectImages(X_train, training_mean, training_pca_axes, training_eigenvalues)
        X_val = projectImages(X_val, training_mean,training_pca_axes,training_eigenvalues)
        X_test = projectImages(X_test, training_mean, training_pca_axes, training_eigenvalues)

        # Train Model
        fold_t_losses, fold_v_losses,        fold_t_accs, fold_v_accs,        fold_best_loss, fold_best_acc = model(X_train, X_val, X_test, y_train,                                              y_val, y_test, stochastic, epochs, lr)
        t_losses.append(fold_t_losses)
        v_losses.append(fold_v_losses)
        t_accs.append(fold_t_accs)
        v_accs.append(fold_v_accs)
        best_te_loss.append(fold_best_loss)
        best_te_acc.append(fold_best_acc)

    print('Best Test Loss: %s' % np.mean(best_te_loss))
    print('Best Test Accuracy: %s' % np.mean(best_te_acc))
    plot(t_losses, v_losses, t_accs, v_accs, k, p, name) # Plotting Loss and Performance Curves


# In[552]:


def get_accuracy(y, t, confusion=False):
    preds = np.argmax(y, 1)
    true = np.argmax(t, 1)
    if confusion:
        print(confusion_matrix(true, preds))
    return (preds == true).mean()

def accumulate(imgs, ohe):
    """
    Takes a defaultdict and returns an array of all images in it
    """
    X = []
    y = []
    for cat in imgs:
        X += imgs[cat]
        y += [target_dict[cat]]*len(imgs[cat])
    X = np.array(X)
    y = np.array(y)
    X = X.reshape(-1, 224*192)
    y = y.reshape(-1,1)
    if ohe:
        y = encode(y)
    return np.hstack([X,y])

def encode(targets):
    ohe = np.zeros((len(targets), len(np.unique(targets))))
    ohe[np.arange(len(targets)), targets.reshape(-1)] = 1
    return ohe


# In[429]:


def plot(t_loss, v_loss, t_acc, v_acc, k, p, name):
    fig, ax = plt.subplots(2, figsize=(18, 12))
    t_loss = np.array(t_loss)
    v_loss = np.array(v_loss)
    t_acc = np.array(t_acc)
    v_acc = np.array(v_acc)
    assert (t_loss.shape == v_loss.shape) and (t_acc.shape == v_acc.shape),    'Both Training and Validation sets should have equal number of values'
    
    x = np.arange(t_loss.shape[1])
    # Loss Plot
    ax[0].set_title('{} Loss Plot (k={}, p={})'.format(name,k,p), fontsize=20, fontweight=200)
    ax[0].set_xlabel('Epoch Number', fontsize=15)
    ax[0].set_ylabel('Normalized Loss', fontsize=15)
    y_t_err = t_loss.std(0)
    y_v_err = v_loss.std(0)
    y_t_err[(x % 10 != 0)] = 0
    y_v_err[(x % 10 != 0)] = 0
    ax[0].errorbar(x, t_loss.mean(0), yerr=y_t_err, label='Average Train Loss')
    ax[0].errorbar(x, v_loss.mean(0), yerr=y_v_err, label='Average Val Loss')
    ax[0].legend()
    
    # Performance Plot
    y_loss = t_loss + v_loss
    ax[1].set_title('{} Performance Plot (k={}, p={})'.format(name,k,p), fontsize=20, fontweight=200)
    ax[1].set_xlabel('Epoch Number', fontsize=15)
    ax[1].set_ylabel('Performance (Accuracy)', fontsize=15)
    y_t_err = t_acc.std(0)
    y_v_err = v_acc.std(0)
    y_t_err[(x % 10 != 0)] = 0
    y_v_err[(x % 10 != 0)] = 0
    ax[1].errorbar(x, t_acc.mean(0), yerr=y_t_err, label='Average Train Accuracy')
    ax[1].errorbar(x, v_acc.mean(0), yerr=y_v_err, label='Average Val Accuracy')
    ax[1].legend()
    fig.savefig('%s.png' % name, bbox_inches='tight')
    plt.show()


# In[553]:


def softmax(X, W):
    A = np.exp(X @ W)  # activations (n x c)
    Y = A/A.sum(1).reshape(-1,1) # sum across all targets (n x c)
    return Y

def update(W, X_train, t_train, lr):
    y_train = softmax(X_train, W)
    dE = X_train.T @ (y_train - t_train) # (d x c)
    W -= lr * dE # Gradient update
    return W, y_train

def softmax_reg(X_train, X_val, X_test, t_train, t_val, t_test, stochastic, epochs, lr):
    """
    t_train, t_val, t_test must be one-hot encoded
    """
    assert (t_train.shape[1] == t_val.shape[1]) and (t_val.shape[1] == t_test.shape[1]),    'One-hot encoded targets must have matching dimensions'
    n = X_train.shape[0] # number of training examples
    d = X_train.shape[1] # dimensions of input
    c = t_train.shape[1] # number of classes
    W = np.zeros((d,c))
    
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    best_val = float('inf')
    best_test_loss = float('inf')
    best_test_acc = 0
    
    for _ in range(epochs):
        y_train = None
        if stochastic:
            for X_train, t_train in batch(X_train, t_train):
                W, y_train = update(W, X_train, t_train, lr)
        else:
            W, y_train = update(W, X_train, t_train, lr)
        
        # Computing Losses
        y_val = softmax(X_val, W)
        y_test = softmax(X_test, W)
        train_loss = cross_entropy(y_train, t_train)
        val_loss = cross_entropy(y_val, t_val)
        test_loss = cross_entropy(y_test, t_test)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        # Computing Accuracies
        train_acc = get_accuracy(y_train, t_train)
        val_acc = get_accuracy(y_val, t_val)
        test_acc = get_accuracy(y_test, t_test)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        # Early stopping
        if val_loss < best_val:
            best_val = val_loss
            best_test_loss = test_loss
            best_test_acc = test_acc
            # Printing confusion matrix
            get_accuracy(y_test, t_test, True)
            
    return train_losses, val_losses, train_accs, val_accs, best_test_loss, best_test_acc


# In[551]:


def cross_entropy(y, t):
    """
    y -> Probability values (n x c)
    t -> One-hot encoded targets (n x c)
    where,
    n -> Number of examples
    c -> Number of classes
    """
    assert y.shape == t.shape, 'Output and Target shapes must match'
    return -(t * np.log(y)).mean()


# In[554]:


# images, counts = load_data()
# classes = counts.keys() #['happiness', 'anger'] #counts.keys()
# images = balanced_sampler(images, counts, classes) 
# target_dict = dict(zip(classes, np.arange(len(classes))))
kfold(images, softmax_reg, True, True, False, 10, 30, 50, 0.001, 'Softmax Six Emotions')


# In[440]:


def batch(X, t):
    idxs = np.arange(len(X))
    np.random.shuffle(idxs)
    for idx in idxs:
        yield X[idx].reshape(1,-1), t[idx].reshape(1,-1)


# In[533]:


from sklearn.metrics import confusion_matrix


# In[ ]:


confusion_matrix


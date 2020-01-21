#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm_notebook as tqn
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from skimage.transform import resize
import warnings as wn

tqdm.pandas()
wn.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df = pd.read_csv('train.csv')
df.head()


# Reading in images

# In[3]:


df['image'] = df.fname.progress_apply(lambda x : plt.imread('image/train/{}.jpg'.format(x))/255)


# Reshaping images for Resnet

# In[4]:


df.image = df.image.progress_apply(lambda x: resize(x, (224,320)))


# In[5]:


def process_img(x):
    img_new = x.reshape(x.shape[2],x.shape[0],x.shape[1])
    return img_new[:3,:,:]  # removing 4 channel images

df.image = df.image.apply(process_img)


# Train test split

# In[6]:


X = np.stack(df.image.values).astype(np.double)
y = df.breedID.values
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)


# In[7]:


X_train = torch.from_numpy(X_train)
y_train = torch.from_numpy(y_train).long()
X_val = torch.from_numpy(X_val)
y_val = torch.from_numpy(y_val).long()

train_set = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_set, batch_size=64)
val_set = TensorDataset(X_val, y_val)
val_loader = DataLoader(val_set, batch_size=64)


# ### Training model

# In[8]:


epochs = 30
lr = 0.01
wd = 0.0001
mt = 0.9
SAVE_PATH = 'dog_cat.ct'


# In[11]:


model = torch.hub.load('pytorch/vision:v0.4.2', 'resnet18', pretrained=False, num_classes=37)


# In[ ]:


best_loss = float('inf')
epoch_train_losses = []
epoch_val_losses = []
criterion = torch.nn.CrossEntropyLoss()
opt = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=wd, momentum=mt)

for _ in tqn(range(epochs)):
    train_losses = []
    val_losses = []
    
    for data, target in tqn(train_loader):
        opt.zero_grad()
        t_loss = criterion(model(data.float()), target-1)
        train_losses.append(t_loss.item())
        t_loss.backward()
        opt.step()
    epoch_train_losses.append(np.mean(train_losses))
    
    for data, target in val_loader:
        opt.zero_grad()
        v_loss = criterion(model(data.float()), target-1)
        val_losses.append(v_loss.item())
    mean_val_loss = np.mean(val_losses)
    epoch_val_losses.append(mean_val_loss)
    if mean_val_loss < best_loss:
        best_loss = mean_val_loss
        torch.save(model.state_dict(), SAVE_PATH)

        
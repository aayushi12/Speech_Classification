#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd

path = np.load("path.npy")
feat = np.load("feat.npy", allow_pickle = True)

train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

y_train = train_df['word']


# In[2]:


#Identifying redundant files and ignoring them
shape_1 = []
shape_2 = []
count = 0
path_outliers = []

for i in range(len(feat)):
    shape_1.append(feat[i].shape[0])
    shape_2.append(feat[i].shape[1])
    
    if feat[i].shape[0] == 1999:
        count = count + 1
        path_outliers.append(path[i])
    
shape_1_unique = set(shape_1)
shape_2_unique = set(shape_2)


# In[3]:


dic = {}

for i in range(path.shape[0]):
    
    if path[i] in path_outliers:
        continue
    else:
        dic[path[i]] = feat[i]


# In[4]:


shape1 = []

for path in dic:
    shape1.append(dic[path].shape[0])


# In[5]:


# Zero padding
#Source of the following code: https://stackoverflow.com/questions/35751306/python-how-to-pad-numpy-array-with-zeros
for path in dic:
    result = np.zeros((max(shape1),max(shape_2_unique)))
    dim1 = dic[path].shape[0]
    dim2 = dic[path].shape[1]
    result[:dim1,:dim2] = dic[path]
    dic[path] = result


# In[7]:


X_train = np.zeros((len(y_train),max(shape1),max(shape_2_unique)))
X_test = np.zeros((len(test_df),max(shape1),max(shape_2_unique)))


# In[8]:


for i in range(len(y_train)):
    X_train[i,:,:] = dic[train_df['path'][i]]


# In[9]:


for i in range(len(test_df)):
    X_test[i,:,:] = dic[test_df['path'][i]]


# In[10]:


train_df['word'] = train_df['word'].astype("category")
train_labels = train_df['word'].cat.codes 

y = np.array(train_labels)


#Source of the following code: https://stackoverflow.com/questions/51102205/how-to-know-the-labels-assigned-by-astypecategory-cat-codes
d = dict(enumerate(train_df['word'].cat.categories)) #Keeping codes to word mapping in a dictionary 


# In[11]:


import numpy as np


# In[12]:


# Source: https://stackoverflow.com/questions/50125844/how-to-standard-scale-a-3d-matrix
from sklearn.preprocessing import StandardScaler
scalers = {}
X_scale = np.zeros(X_train.shape)

for i in range(X_train.shape[2]):
    scalers[i] = StandardScaler()
    X_scale[:, :, i] = scalers[i].fit_transform(X_train[:, :, i]) 


# In[13]:


from sklearn.model_selection import train_test_split

x_train, x_val, y_train, y_val = train_test_split(X_scale,y, test_size=0.20, random_state=42, stratify = y)


# In[26]:


get_ipython().run_line_magic('pylab', 'inline --no-import-all')
import matplotlib.pyplot as plt
n, bins, patches = plt.hist(y_train, 100 , facecolor='blue', alpha=0.5)
plt.ylabel('Class distribution')
plt.show() 


# In[27]:


n, bins, patches = plt.hist(y_val, 100 , facecolor='blue', alpha=0.5)
plt.ylabel('Class distribution')
plt.show() 


# In[14]:


x_train = x_train.reshape(x_train.shape[0], 99, 13,1)
x_val =  x_val.reshape(x_val.shape[0], 99, 13,1)


# In[15]:


from keras.utils import to_categorical

y_train_onehot = to_categorical(y_train, num_classes= 35)
y_val_onehot = to_categorical(y_val, num_classes= 35)


# In[16]:


from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D
from keras import regularizers, optimizers
from keras.layers.advanced_activations import LeakyReLU


#Source of the following code: https://medium.com/gradientcrescent/urban-sound-classification-using-convolutional-neural-networks-with-keras-theory-and-486e92785df4
# We have made some modifications
model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=(99,13,1)))
model.add(LeakyReLU(alpha=.001))
model.add(Conv2D(64, (3, 3),strides = 2))
model.add(LeakyReLU(alpha=.001))

model.add(Dropout(0.25))
model.add(Conv2D(64, (3, 3), padding='same'))
model.add(LeakyReLU(alpha=.001))
model.add(Conv2D(64, (3, 3),strides = 3))
model.add(LeakyReLU(alpha=.001))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(128))
model.add(LeakyReLU(alpha=.001))

model.add(Dense(35, activation='softmax'))


# In[17]:


# Source: https://datascience.stackexchange.com/questions/13490/how-to-set-class-weights-for-imbalanced-classes-in-keras
from sklearn.utils import class_weight
class_weights = class_weight.compute_class_weight("balanced", np.unique(y_train), y_train)


# In[22]:


import keras.optimizers

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(lr=0.0004),
              metrics=['accuracy'])
model.fit(x_train, y_train_onehot, validation_data=(x_val, y_val_onehot), epochs= 100, class_weight = class_weights)


# In[19]:


#You will have to scale X_test as well

scalers = {}
X_scale_test = np.zeros(X_test.shape)

for i in range(X_test.shape[2]):
    scalers[i] = StandardScaler()
    X_scale_test[:, :, i] = scalers[i].fit_transform(X_test[:, :, i]) 

x_test = X_scale_test.reshape(X_scale_test.shape[0], 99, 13,1)


# In[20]:


y_pred = model.predict_on_batch(x_test)
#These predictions will have to be converted into the corresponding category and then to csv


# In[21]:


# Source of the code: https://stackoverflow.com/questions/45738308/scikit-convert-one-hot-encoding-to-encoding-with-integers
y_pred_labels = np.zeros((len(y_pred,)))


# In[22]:


for i in range(len(y_pred)):
    y_pred_labels[i] = np.argmax(y_pred[i])


# In[23]:


test_words = []
for i in range(len(y_pred_labels)):
    test_words.append(d[y_pred_labels[i]])


# In[24]:


test_df["word"] = test_words


# In[53]:


test_df.to_csv("result.csv",index = None)


#!/usr/bin/env python
# coding: utf-8

# In[1]:


from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator


# In[2]:


import tensorflow as tf


# In[24]:


image_datagen =  ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True,
                                   vertical_flip=True)


# In[25]:


train_set = image_datagen.flow_from_directory(r'C:\Users\vasu0\Desktop\lenovo data(descktop)\MLLL\Projects\dogcat_new\Monkey\training\training',
                                             target_size = (224, 224),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')


# In[26]:


test_datagen = ImageDataGenerator(rescale = 1./255)
test_set = test_datagen.flow_from_directory(r'C:\Users\vasu0\Desktop\lenovo data(descktop)\MLLL\Projects\dogcat_new\Monkey\validation\validation',
                                            target_size = (224, 224),
                                            batch_size = 32,
                                            class_mode = 'categorical')


# In[27]:


cnn= tf.keras.models.Sequential()
cnn.add(tf.keras.layers.Conv2D(filters= 32,padding='same',kernel_size=3,activation='relu',input_shape=[224,224,3]))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2))


cnn.add(tf.keras.layers.Conv2D(filters= 32,padding='same',kernel_size=3,activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2))

cnn.add(tf.keras.layers.Flatten())

cnn.add(tf.keras.layers.Dense(units=128,activation='relu'))
cnn.add(tf.keras.layers.Dense(units=10,activation='softmax'))


# In[28]:


cnn.summary()


# In[30]:


cnn.compile(optimizer='adam',loss='categorical_crossentropy',
            metrics=['accuracy'])

history=cnn.fit(x=train_set,validation_data=test_set,epochs=25)


# In[32]:


history.model.save("C:\\Users\\vasu0\\Desktop\\lenovo data(descktop)\\MLLL\\Projects\\dogcat_new\\Monkey\\model-25epoch.h5")
print("Saved model to disk")


# In[33]:


from tensorflow import keras
model = keras.models.load_model('C:\\Users\\vasu0\\Desktop\\lenovo data(descktop)\\MLLL\\Projects\\dogcat_new\\Monkey\\model-25epoch.h5')


# In[ ]:





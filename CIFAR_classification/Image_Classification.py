import pandas as pd
import matplotlib.pyplot as plt
from keras.datasets import cifar10
from keras.utils import to_categorical

(X_train,y_train),(X_test,y_test)=cifar10.load_data()

n=6
plt.figure(figsize=(20,10))
for i in range(n):
    plt.subplot(330+1+i)
    plt.imshow(X_train[i])
plt.show()

X_train=X_train/255.0
X_test=X_test/255.0

y_train=to_categorical(y_train)
y_test=to_categorical(y_test)

from keras.layers import Flatten, Dense, MaxPooling2D, Conv2D, Dropout
from keras.models import Sequential
from keras.optimizers import SGD,Adadelta
from keras.losses import categorical_crossentropy

model=Sequential()
model.add(Conv2D(16,kernel_size=(3,3),activation='relu',input_shape=(32,32,3)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(32,kernel_size=(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(64,kernel_size=(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(258,activation='relu'))
model.add(Dense(10,activation='softmax'))

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

model.fit(X_train,y_train,batch_size=20,epochs=20,validation_data=(X_test,y_test))

_,acc=model.evaluate(X_test,y_test)
print(acc*100)

model.save("Classification.h5")
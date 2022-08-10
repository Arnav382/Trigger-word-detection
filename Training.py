import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense,Activation,Dropout
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.utils import to_categorical

df=pd.read_pickle("final_audio_data_csv/audio_data.csv")
X=df['features'].values
X=np.concatenate(X,axis=0).reshape(len(X),40)
y=np.array(df["class"].tolist())
y=to_categorical(y)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)

model=Sequential([
    Dense(256,input_shape=X_train[0].shape),
    Activation('relu'),
    Dropout(0.2),
    Dense(128),
    Activation('relu'),
    Dropout(0.2),
    Dense(32),
    Activation('relu'),
    Dropout(0.4),
    Dense(2,activation='softmax')
])

print(model.summary())
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy'])

print("Model Score :\n")
model.fit(X_train,y_train,batch_size=32,epochs=150)
model.save("models/v_1.0.h5") 
score=model.evaluate(X_test,y_test)
print(score)

print("model classification report")
y_pred=np.argmax(model.predict(X_test),axis=1)
cm=confusion_matrix(np.argmax(y_test,axis=1),y_pred)
print(classification_report(np.argmax(y_test,axis=1),y_pred,))
print(cm)

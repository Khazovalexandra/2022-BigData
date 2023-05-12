import os
import pandas as pd
import numpy as np
import pathlib
import tensorflow as tf
import re
from sklearn.model_selection import train_test_split
from tensorflow import keras
from keras.utils import plot_model
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Sequential
from keras.datasets import mnist
import matplotlib.pylab as plt

df = pd.read_csv(r'..\2022-BigData\_lab-6\movies_dataset.csv')
                 
#print(df.describe())

data_dir = pathlib.Path("./_lab-6/movies_posters/")
img_height = 281 // 5
img_width = 190 // 5
batch_size = 32
epochs = 1
#genres = ['Action', 'Crime', 'Drama', 'Thriller', 'Comedy', 'Romance', 'Family', 'Adventure', 'History', 'Music', 'Documentary', 'Horror', 'Science Fiction', 'Fantasy', 'Foreign', 'Mystery']
#genres_num = len(genres)
genres_set = str(set(str([i.rstrip('\n')[1:-1] for i in df['genres_list'].drop(index=[19730, 35587, 29503])]).replace('"','').replace(' ', '').replace(']','').replace('[','').split(','))).replace(' ', '').replace('"', '').replace("'", "").replace("{", "").replace("}", "").split(',')
valg = len(genres_set)
print(valg, genres_set)

def load_imgs(data_dir=data_dir):
    image_count = len(list(data_dir.glob('*/*.jpg')))
    list_ds = tf.data.Dataset.list_files(str(data_dir/'*'), shuffle=True)
    #list_ds = list_ds.shuffle(image_count, reshuffle_each_iteration=False)
    return list_ds, image_count

def get_id(file_path):
    parts = tf.strings.split(file_path, sep=os.path.sep)
    img_id = tf.strings.split(parts[-1], ".")[-2]
    one_hot = img_id == df['id']
    #print(one_hot)
    return img_id
    #return list(df[df['id'] == img_id]['genres_list'])

def decode_img(img, img_height=img_height, img_width=img_width):
    img = tf.io.decode_jpeg(img, channels=3)
    return tf.image.resize(img, [img_height, img_width])

def get_label(file_path):
    label = tf.strings.split(file_path, sep=os.path.sep)
    one_hot = label[-2] == genres_set
    return tf.argmax(one_hot)

def get_year(id):
    year = df.loc[df['id'] == id, 'release_date'].iloc[0]
    year = tf.strings.split(year, "-")[-3]
    return year

def processing_path(file_path):
    id = get_id(file_path)
    label = get_label(id)
    img = tf.io.read_file(file_path)
    img = decode_img(img)
    return img, label

list_ds, image_count = load_imgs()
val_size = image_count // 5
train_ds = list_ds.skip(val_size)
val_ds = list_ds.take(val_size)

train_ds = train_ds.map(processing_path, num_parallel_calls=tf.data.AUTOTUNE)
val_ds = val_ds.map(processing_path, num_parallel_calls=tf.data.AUTOTUNE)
train_ds=train_ds.batch(batch_size)
val_ds=val_ds.batch(batch_size)

"""""
val_g = valg // 5
listg = list(range(0,valg))
train_genres = list(listg)[:val_g]
val_genres = list(listg)[val_g:]
genres_train = keras.utils.to_categorical(train_genres, valg)
genres_val = keras.utils.to_categorical(val_genres, valg)
"""""

"""""
for img, id in train_ds.take(1):
    print(f"Name film: {df.loc[df['id'] == id, 'title'].iloc[0]}")
    print(f"image shape: {img.numpy().shape}") #в img хранятся массивы пикселей
    print(f"Labels: {get_label(id)}")
    print(f"Year: {get_year(id)}")
"""""
model = tf.keras.models.Sequential([
    #tf.keras.layers.Flatten(input_shape=(batch_size, img_height, img_width, 3)),
    #tf.keras.Input(shape=(img_height, img_width, 3)),
    tf.keras.layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
    tf.keras.layers.Conv1D(32, kernel_size = 3, padding='same', activation = 'sigmoid'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv1D(128, kernel_size = 3, padding='same', activation = 'sigmoid'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, kernel_size = 3, padding='same', activation = 'sigmoid'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    #tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(valg, activation='relu')
])

model.compile(optimizer='adam', 
              loss='BinaryCrossentropy',
              metrics=['accuracy'])

model.summary()

plot_model(model, to_file=r'..\2022-BigData\Хазова Александра Сергеевна\интеллектуальный анализ\lab6\my_first_model.png', show_shapes=True)

history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)

print(history.history)
print('Test loss:', history.history['loss'])
print('Test accuracy:', history.history['accuracy'])
plt.plot(range(1, 2), history['accuracy'])
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.show()
score = model.evaluate(train_ds, batch_size=batch_size)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
#plt.savefig(r'..\2022-BigData\Хазова Александра Сергеевна\интеллектуальный анализ\lab6\my_network.png')
plt.show()

"""""

model = tf.keras.models.Sequential([
    #tf.keras.layers.Flatten(input_shape=(batch_size, img_height, img_width, 3)),
    #tf.keras.Input(shape=(img_height, img_width, 3)),
    #tf.keras.layers.Conv1D(1000, kernel_size = 3, padding='same', activation = 'sigmoid'),
    #tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
    tf.keras.layers.Conv2D(32, kernel_size = 3, padding='same', activation = 'relu'),
    tf.keras.layers.MaxPooling2D(),
    #tf.keras.layers.Conv2D(128, kernel_size = 3, padding='same', activation = 'relu'),
    #tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(16)
])

model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

plot_model(model, to_file=r'..\2022-BigData\Хазова Александра Сергеевна\интеллектуальный анализ\lab6\my_first_model.png', show_shapes=True)

early_stopping=keras.callbacks.EarlyStopping(monitor='val_loss')  #ранняя остановка(остановка обучения если ошибка перестала уменьшаться)

history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs, 
  validation_split=0.1
)
print(history.history)

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
#plt.savefig(r'..\2022-BigData\Хазова Александра Сергеевна\интеллектуальный анализ\lab6\my_network.png')
plt.show()



(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
input_shape = (img_height, img_width, 3)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)
print(y_train, y_test)

model = Sequential()
model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1),
                 activation='relu',
                 input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(64, (5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(1000, activation='relu'))
#model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])

class AccuracyHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.acc = []

    def on_epoch_end(self, batch, logs={}):
        self.acc.append(logs.get('acc'))

history = AccuracyHistory()

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test),
          callbacks=[history])
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
plt.plot(range(1, 11), history.acc)
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.show()
"""""
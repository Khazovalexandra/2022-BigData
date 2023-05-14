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
epochs = 10
#genres = ['Action', 'Crime', 'Drama', 'Thriller', 'Comedy', 'Romance', 'Family', 'Adventure', 'History', 'Music', 'Documentary', 'Horror', 'Science Fiction', 'Fantasy', 'Foreign', 'Mystery']
#genres_num = len(genres)
#genres_set = str(set(str([i.rstrip('\n')[1:-1] for i in df['genres_list'].drop(index=[19730, 35587, 29503])]).replace('"','').replace(' ', '').replace(']','').replace('[','').split(','))).replace(' ', '').replace('"', '').replace("'", "").replace("{", "").replace("}", "").split(',')
genres_set = np.array([item.name for item in data_dir.glob('*') if item.name != "LICENSE.txt"])
valg = len(genres_set)
print(valg, genres_set)

def load_imgs(data_dir=data_dir):
    image_count = len(list(data_dir.glob('*/*.jpg')))
    list_ds = tf.data.Dataset.list_files(str(data_dir/'*/*'), shuffle=True)
    #list_ds = list_ds.shuffle(image_count, reshuffle_each_iteration=False)
    return list_ds, image_count
"""""
def get_id(file_path):
    parts = tf.strings.split(file_path, sep=os.path.sep)
    img_id = tf.strings.split(parts[-1], ".")[-2]
    one_hot = img_id == df['id']
    #print(one_hot)
    return img_id
    #return list(df[df['id'] == img_id]['genres_list'])
"""""
def decode_img(img, img_height=img_height, img_width=img_width):
    img = tf.io.decode_jpeg(img, channels=3)
    return tf.image.resize(img, [img_height, img_width])

def get_label(file_path):
    label = tf.strings.split(file_path, sep=os.path.sep)
    one_hot = label[-2] == genres_set
    return tf.argmax(one_hot)
"""""
def get_year(id):
    year = df.loc[df['id'] == id, 'release_date'].iloc[0]
    year = tf.strings.split(year, "-")[-3]
    return year
"""""
def processing_path(file_path):
    label = get_label(file_path)
    img = tf.io.read_file(file_path)
    img = decode_img(img)
    return img, label

list_ds, image_count = load_imgs()
val_size = image_count // 5
train_ds = list_ds.skip(val_size)
val_ds = list_ds.take(val_size)

train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)
val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)
"""""
for img, label in train_ds.take(3):
    print(f"image shape: {img.numpy().shape}") #в img хранятся массивы пикселей
    print(f"Labels: {label}")
"""""
model = tf.keras.models.Sequential([
    tf.keras.layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
    tf.keras.layers.Conv2D(128, kernel_size=8, padding='same', activation='sigmoid'),
    tf.keras.layers.MaxPool2D(),
    tf.keras.layers.Conv2D(64, kernel_size=8, padding='same', activation='sigmoid'),
    tf.keras.layers.MaxPool2D(),
    tf.keras.layers.Conv2D(32, kernel_size=8, padding='same', activation='sigmoid'),
    tf.keras.layers.MaxPool2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, kernel_regularizer=tf.keras.regularizers.l2(0.0001), activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(64, kernel_regularizer=tf.keras.regularizers.l2(0.0001), activation='softmax'),
    tf.keras.layers.Dense(len(genres_set))
])

model.compile(optimizer='adam', 
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.summary()

plot_model(model, to_file=r'..\2022-BigData\Хазова Александра Сергеевна\интеллектуальный анализ\lab6\my_first_model.png', show_shapes=True)

checkpoint_path = "./Хазова Александра Сергеевна/интеллектуальный анализ/lab6/training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

model.save_weights(checkpoint_path.format(epoch=0))

history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs,
  callbacks=[cp_callback]
)
"""""
print(history.history)
print('Test loss:', history.history['loss'])
print('Test accuracy:', history.history['accuracy'])
plt.plot(range(1, 4), history.history['accuracy'])
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.show()
"""""
#score = model.evaluate(train_ds, batch_size=batch_size)

os.listdir(checkpoint_dir)

loss, acc = model.evaluate(val_ds, verbose=2)
print("Untrained model, accuracy: {:5.2f}%".format(100 * acc))

latest = tf.train.latest_checkpoint(checkpoint_dir)
model.load_weights(latest)

loss, acc = model.evaluate(val_ds, verbose=2)
print("Restored model, accuracy: {:5.2f}%".format(100 * acc))

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
plt.savefig(r'..\2022-BigData\Хазова Александра Сергеевна\интеллектуальный анализ\lab6\my_network.png')
plt.show()

#предсказания модели
test_images = np.random(val_ds)
predictions = tf.keras.probability_model.predict(test_images)
predictions[0]
print(np.argmax(predictions[0]))#показывает где наибольшая вероятность(какой класс наиболее вероятен)

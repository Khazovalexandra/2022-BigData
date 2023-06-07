import os
import pandas as pd
import numpy as np
import pathlib
import tensorflow as tf
import re
from sklearn.model_selection import train_test_split
from tensorflow import keras
from keras.utils import plot_model
from keras.datasets import mnist
import matplotlib.pylab as plt

df = pd.read_csv(r'..\2022-BigData\_lab-6\movies_dataset.csv')
                 
#print(df.describe())

data_dir = pathlib.Path("./_lab-6/movies_posters/")
test_dir = pathlib.Path("./Хазова Александра Сергеевна/интеллектуальный анализ/lab6/test")
img_height = 281 // 5
img_width = 190 // 5
batch_size = 32
epochs = 1
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
    batch_size=batch_size, 
    label_mode='categorical')

val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size, 
    label_mode='categorical')

#расширение данных(data augmentation)
img_train_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255, 
                                rotation_range = 40,
                                width_shift_range = 0.2, 
                                height_shift_range = 0.2,
                                shear_range = 0.2,
                                zoom_range = 0.2,
                                horizontal_flip = True, 
                                fill_mode='nearest')

data_augmentation = tf.keras.models.Sequential([
    tf.keras.layers.RandomFlip("horizontal", input_shape=(img_height, img_width, 3)),
    tf.keras.layers.RandomRotation(0.2),
    tf.keras.layers.RandomZoom(0.2, 0.2)
])

train_ds = train_ds.map(
    lambda img, label: (data_augmentation(img), label),
    num_parallel_calls=tf.data.AUTOTUNE,
)

# Prefetching samples in GPU memory helps maximize GPU utilization.
train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

"""""
plt.figure(figsize=(10, 10))
for img, label in train_aug_ds.take(2):
    lbl = label.numpy()
    for i in range(9):
        ax = plt.subplot(3, 3, i+1)
        plt.imshow(img[i].numpy().astype("uint8"))
        plt.title(genres_set[lbl[i]])
        plt.axis("off")
plt.show()"""""

"""""
for img, label in train_ds.take(3):
    print(f"image shape: {img.numpy().shape}") #в img хранятся массивы пикселей
    print(f"Labels: {label}")
"""""

model = tf.keras.models.Sequential([
    tf.keras.layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
    tf.keras.layers.Conv2D(128, kernel_size=(3,3), padding='same', activation='relu', strides=(2,2)),
    tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=None,
                              padding='valid',
                              data_format='channels_last'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(64, kernel_size=(3,3), padding='same', activation='relu', strides=(2,2)),
    tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=None,
                              padding='valid',
                              data_format='channels_last'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(32, kernel_size=(3,3), padding='same', activation='relu', strides=(1,1)),
    tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=None,
                              padding='valid',
                              data_format='channels_last'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='softmax'),
    tf.keras.layers.Dense(len(genres_set))
])

model.compile(optimizer='adam', 
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'], jit_compile=True)

model.summary()

plot_model(model, to_file=r'..\2022-BigData\Хазова Александра Сергеевна\интеллектуальный анализ\lab6\my_first_model.png', show_shapes=True)

checkpoint_path = "./Хазова Александра Сергеевна/интеллектуальный анализ/lab6/training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_best_only=True,
                                                 save_weights_only=True,
                                                 verbose=1)

def scheduler(epoch, lr):
  if epoch < 10:
    return lr
  else:
    return lr * tf.math.exp(-0.1)

callback = tf.keras.callbacks.LearningRateScheduler(scheduler) #изменяет скорость обучения(берет из цикла выше)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', factor=0.2, #если не наблюдается изменения обучения,
                                                 patience=3, min_lr=0.001)       #снижает скорость обучения

#latest = tf.train.latest_checkpoint(checkpoint_dir) #берет последние веса из файла
#model.load_weights(latest) #достает веса из файла ckpt

#loss, acc = model.evaluate(val_ds, verbose=2)
#print("Untrained model, accuracy: {:5.2f}%".format(100 * acc))

history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs,
  shuffle=True,
  callbacks=[callback, cp_callback, reduce_lr]
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

#model.save_weights(checkpoint_path) #сохраняет веса

loss, acc = model.evaluate(val_ds, verbose=2)
print("Untrained model, accuracy: {:5.2f}%".format(100 * acc))

model.save('./Хазова Александра Сергеевна/интеллектуальный анализ/lab6/my_model.h5') #сохраняет модель целиком

#os.listdir(checkpoint_dir)

print(model.get_weights())#показывает все веса в виде массива numpy

#new_model = tf.keras.models.load_model('./Хазова Александра Сергеевна/интеллектуальный#анализ/lab6/my_model.h5')

#loss, acc = new_model.evaluate(val_ds, verbose=2)
#print("Restored model, accuracy: {:5.2f}%".format(100 * acc))

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Точность на обучающих данных')
plt.plot(epochs_range, val_acc, label='Точность на валидационных данных')
plt.legend(loc='lower right')
plt.title('Точность')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Потери на обучении')
plt.plot(epochs_range, val_loss, label='Потери на валидации')
plt.legend(loc='upper right')
plt.title('Потери')
plt.savefig(r'..\2022-BigData\Хазова Александра Сергеевна\интеллектуальный анализ\lab6\my_network.png')
plt.show()

#предсказания модели
for image, lab in val_ds.take(1):
      pred = model.predict(image)
      print(pred)

for item_path in os.listdir(str(test_dir)):
    item_path = os.path.join(str(test_dir / item_path))
    img = tf.io.read_file(item_path)
    img = tf.image.decode_jpeg(img)
    reshaped_img = tf.image.resize(img, (img_height, img_width))
    reshaped_img = tf.reshape(reshaped_img, (img_height, img_width, 3))
    img_arr = keras.utils.img_to_array(reshaped_img)
    img_arr = tf.expand_dims(img_arr, 0)

    predictions = model.predict(img_arr)
    score = tf.nn.softmax(predictions[0])
    item_name = item_path.split(os.path.sep)[-1]
    print("Score for {}: {}, {}%".format(
        item_name,
        genres_set[np.argmax(score)],
        np.max(score) * 100)
    )
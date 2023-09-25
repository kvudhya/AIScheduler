
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import sys

import tensorflow as tf

import pandas as pd

from main import *
from tensorflow import keras

from tensorflow.keras import layers

from tensorflow.keras.models import Sequential

from tensorflow.keras.datasets import cifar10
import tensorflow as tf

import pandas as pd
import numpy as np

from tensorflow import keras

from tensorflow.keras import layers

from tensorflow.keras.models import Sequential


gpus = tf.config.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only use the first GPU
  try:
    tf.config.set_visible_devices(gpus[0], 'GPU')
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
  except RuntimeError as e:
    # Visible devices must be set before GPUs have been initialized
    print(e)

def normalization_label(label):
  if label==1:
    return 3
  if label == 15:
    return 0
  if label ==30:
    return 1
  if label ==45:
    return 2
  if label ==2:
    return 4


label_map = {0: 15, 1:30, 2:45,3:1, 4:2}


def filter_train(time):
  return (True
          if time<4
          else False)

dataset = pd.read_csv("present_csv.csv")

dataset['Time Duration']= dataset['Time Duration'].map(normalization_label)

#dataset = dataset.drop(columns=["Unnamed"])
print(dataset.head())
raw_train_ds = tf.data.Dataset.from_tensor_slices((dataset['Events'], dataset['Time Duration']))


print(raw_train_ds)
raw_test_ds= raw_train_ds.take(4)

raw_train_ds = raw_train_ds.skip(4)

max_features = 10000
vectorize_layer = layers.TextVectorization(
  max_tokens= 10000,
  output_mode = 'int',
  output_sequence_length  =250
)

text = raw_train_ds.map(lambda text, label: text)

vectorize_layer.adapt(text)





def converter(text,label):
  return text, tf.cast(label, dtype=tf.int64)

def vectorized_text(text, label):
  text= tf.expand_dims(text,-1)

  return vectorize_layer(text), label

AUTOTUNE =tf.data.AUTOTUNE

train_ds = raw_train_ds.padded_batch(2)

test_ds= raw_test_ds.padded_batch(2)
train_ds= train_ds.map(vectorized_text).map(converter).cache().prefetch(AUTOTUNE).shuffle(1000)

test_ds = test_ds.map(vectorized_text).cache().prefetch(AUTOTUNE)


for text_batch, label_batch in train_ds.take(1):
  for i in range(2):
    print(text_batch.shape)
    print(label_batch.shape)

print(train_ds)


def textclass_model(vocab_size,num_classes):
  model = keras.Sequential([
    #vectorize_layer,
    layers.Embedding(vocab_size, 16, mask_zero = True),
    layers.Conv1D(16, 5, padding = 'valid', activation = 'relu', strides = 2),
    #layers.Bidirectional(layers.LSTM(64)),
    layers.BatchNormalization(),
    layers.Conv1D(32, 5, padding='valid', activation='relu', strides=2),
    #   layers.Bidirectional(layers.LSTM(64)),
    layers.BatchNormalization(),
    layers.GlobalMaxPooling1D(),
    layers.Dropout(0.5),
    layers.Dense(64, activation = 'relu'),
    layers.Dense(num_classes),

  ])
  return model




int_model = textclass_model(vocab_size=max_features + 1, num_classes=5)

checkpoints_callback = keras.callbacks.ModelCheckpoint('checkpoints/', 'accuracy', save_best_only=True,)
#saving_callback= keras.callbacks.EarlyStopping('accuracy', min_delta=0.30, restore_best_weights=True)
int_model.compile(
  loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True),
  optimizer =keras.optimizers.Adam(lr = 3e-4),
  metrics = ['accuracy'],
)

int_model.fit(train_ds,epochs = 10,verbose=2, callbacks=[checkpoints_callback])

#int_model.load_weights('checkpoints')

int_model.evaluate(test_ds, verbose=2)




#savemodel
saved_model = int_model.save('scemodelclassification', save_format='h5')


exported_model = tf.keras.models.load_model('scemodelclassification')



def prediction(examples,trained_Model):

  exported_model = keras.Sequential([
    vectorize_layer,
    trained_Model,
    keras.layers.Activation('softmax')
  ])
  if examples is None:
    return
  else:
    if len(examples)>1:
      predictions = exported_model.predict(examples)
      for example in predictions:
        score = tf.nn.softmax(example)
        return label_map[np.argmax(score)]
    else:
      predictions = exported_model.predict(examples)
      score = tf.nn.softmax(predictions[0])
      return label_map[np.argmax(score)]
#print(exported_model.predict(examples))

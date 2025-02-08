# import libraries
# This was originally a Colab notebook
#try:
  # %tensorflow_version only exists in Colab.
#  !pip install tf-nightly
#except Exception:
#  pass
import tensorflow as tf
import pandas as pd
from tensorflow import keras
#!pip install tensorflow-datasets
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re

print(tf.__version__)

# get data files
#!wget https://cdn.freecodecamp.org/project-data/sms/train-data.tsv
#!wget https://cdn.freecodecamp.org/project-data/sms/valid-data.tsv

train_file_path = "train-data.tsv"
test_file_path = "valid-data.tsv"

train_dataset = pd.read_csv('train-data.tsv',
                            sep='\t',
                            header=None,
                            names=['label', 'text'])
test_dataset = pd.read_csv('valid-data.tsv',
                            sep='\t',
                            header=None,
                            names=['label', 'text'])

max_len = 100
embedding_dim = 16
vocab_size = 500

tokenizer = Tokenizer(num_words = vocab_size)

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\W+', ' ', text)
    return text

train_dataset['text'] = train_dataset['text'].apply(preprocess_text)
test_dataset['text'] = test_dataset['text'].apply(preprocess_text)

tokenizer = Tokenizer(num_words=vocab_size)
tokenizer.fit_on_texts(train_dataset['text'])
train_sequences = tokenizer.texts_to_sequences(train_dataset['text'])
test_sequences = tokenizer.texts_to_sequences(test_dataset['text'])

train_padded = pad_sequences(train_sequences, maxlen=max_len, padding='post')
test_padded = pad_sequences(test_sequences, maxlen=max_len, padding='post')

train_labels = np.where(train_dataset['label'] == 'spam', 1, 0)
test_labels = np.where(test_dataset['label'] == 'spam', 1, 0)

model = tf.keras.Sequential([
  tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_len),
  tf.keras.layers.GlobalAveragePooling1D(),
  tf.keras.layers.Dense(24, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(1, activation='sigmoid')
  ])
model.build((vocab_size, 24))
model.summary()

model.compile(loss="binary_crossentropy",
              optimizer="adam",
              metrics=['accuracy'])

epochs = 30
history = model.fit(
    train_padded,
    train_labels,
    epochs=epochs,
    validation_data=(test_padded, test_labels)
)

# function to predict messages based on model
# (should return list containing prediction and label, ex. [0.008318834938108921, 'ham'])
def predict_message(pred_text):
    processed_message = preprocess_text(pred_text)
    sequence = tokenizer.texts_to_sequences([processed_message])
    padded_sequence = pad_sequences(sequence, maxlen=max_len, padding='post')
    prediction_prob = model.predict(padded_sequence)[0][0]
    prediction_label = 'spam' if prediction_prob > 0.5 else 'ham'

    return [prediction_prob, prediction_label]

pred_text = "how are you doing today?"

prediction = predict_message(pred_text)
print(prediction)

# Run this cell to test your function and model. Do not modify contents.
def test_predictions():
  test_messages = ["how are you doing today",
                   "sale today! to stop texts call 98912460324",
                   "i dont want to go. can we try it a different day? available sat",
                   "our new mobile video service is live. just install on your phone to start watching.",
                   "you have won £1000 cash! call to claim your prize.",
                   "i'll bring it tomorrow. don't forget the milk.",
                   "wow, is your arm alright. that happened to me one time too"
                  ]

  test_answers = ["ham", "spam", "ham", "spam", "spam", "ham", "ham"]
  passed = True

  for msg, ans in zip(test_messages, test_answers):
    prediction = predict_message(msg)
    print(prediction)
    if prediction[1] != ans:
      passed = False

  if passed:
    print("You passed the challenge. Great job!")
  else:
    print("You haven't passed yet. Keep trying.")

test_predictions()
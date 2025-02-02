import keras.src.legacy.preprocessing.text
from keras.datasets import imdb
from keras.preprocessing import sequence
import tensorflow as tf
import os
import numpy as np
from keras.src.saving.saving_lib import save_weights_only


class SentimentAnalyzer:
    def __init__(self):
        self.VOCAB_SIZE = 88584
        self.MAXLEN = 250
        self.BATCH_SIZE = 64

        (train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words = self.VOCAB_SIZE)

        self.train_data = sequence.pad_sequences(train_data, self.MAXLEN)
        self.test_data = sequence.pad_sequences(test_data, self.MAXLEN)

        self.model = tf.keras.Sequential([
            tf.keras.layers.Embedding(self.VOCAB_SIZE, 32),
            tf.keras.layers.LSTM(32),
            tf.keras.layers.Dense(1, activation="sigmoid")
        ])

        self.model.build((self.VOCAB_SIZE, 32))
        #self.model.summary()
        self.model.compile(loss="binary_crossentropy", optimizer="rmsprop", metrics=['acc'])
        self.history = self.model.fit(train_data, train_labels, epochs=10, validation_split=0.2)
        self.results = self.model.evaluate(test_data, test_labels)
        self.word_index = imdb.get_word_index()
        self.reverse_word_index = {value: key for (key, value) in self.word_index.items()}

    def encode_text(self, text):
        tokens = keras.src.legacy.preprocessing.text.text_to_word_sequence(text)
        tokens = [self.word_index[word] if word in self.word_index else 0 for word in tokens]

        return sequence.pad_sequences([tokens], self.MAXLEN)[0]

    def decode_integers(self, integers):
        PAD = 0
        text = ""
        for num in integers:
            if num != PAD:
                text += self.reverse_word_index[num]+ " "

        return text[:1]

    def predict(self, text):
        encoded_text = self.encode_text(text)
        pred = np.zeros((1,250))
        pred[0] = encoded_text
        result = self.model.predict(pred)
        print(result[0])

class PlayGenerator:
    def __init__(self):
        self.path_to_file = tf.keras.utils.get_file('shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')
        self.text = open(self.path_to_file, 'rb').read().decode(encoding='utf-8')
        self.vocab = sorted(set(self.text))
        self.char2idx = {u: i for i, u in enumerate(self.vocab)}
        self.idx2char = np.array(self.vocab)
        self.text_as_int = self.text_to_int(self.text)
        self.seq_length = 100
        self.examples_per_epoch = len(self.text)//(self.seq_length+1)
        self.char_dataset = tf.data.Dataset.from_tensor_slices(self.text_as_int)
        self.sequences = self.char_dataset.batch(self.seq_length+1, drop_remainder=True)
        self.dataset = self.sequences.map(self.split_input_target)

        self.BATCH_SIZE = 64
        self.VOCAB_SIZE = len(self.vocab)
        self.EMBEDDING_DIM = 256
        self.RNN_UNITS = 1024
        self.BUFFER_SIZE = 10000

        self.data = self.dataset.shuffle(self.BUFFER_SIZE).batch(self.BATCH_SIZE, drop_remainder=True)
        self.model = None

    def text_to_int(self, text):
        return np.array([self.char2idx[c] for c in text])

    def int_to_text(self, ints):
        try:
            ints = ints.numpy()
        except:
            pass
        return ''.join(self.idx2char[ints])

    def split_input_target(self, chunk):
        input_text = chunk[:-1]
        target_text = chunk[1:]

        return input_text, target_text

    def build_model(self, vocab_size, embedding_dim, rnn_units, batch_size):
        model = tf.keras.Sequential([
            tf.keras.layers.Embedding(vocab_size, embedding_dim),
            tf.keras.layers.LSTM(rnn_units,
                                 return_sequences=True,
                                 stateful=True,
                                 recurrent_initializer='glorot_uniform'),
            tf.keras.layers.Dense(vocab_size)
        ])
        model.build(input_shape=(batch_size, None))
        self.model = model
        return model

    def loss(self, labels, logits):
        return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

    def generate_text(self, model, start_string):
        num_generate = 800

        input_eval = [self.char2idx[s] for s in start_string]
        input_eval = tf.expand_dims(input_eval, 0)

        text_generated = []

        # Lower temperatures results in more predictable text.
        # Higher temperatures results in more surprising text.
        temperature = 1.0

        model.reset_states()
        for i in range(num_generate):
            predictions = model(input_eval)
            predictions = tf.squeeze(predictions, 0)

            predictions = predictions / temperature
            predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()

            input_eval = tf.expand_dims([predicted_id], 0)

            text_generated.append(self.idx2char[predicted_id])

        return start_string + ''.join(text_generated)

if __name__ == "__main__":
    #model = SentimentAnalyzer()
    #positive_review = "That movie was so awesome! I really loved it and would watch it again because it was amazingly great"
    #model.predict(positive_review)

    #negative_review = "That movie sucked. I hated it and wouldn't watch it again. Was one of the worst things I've ever watched."
    #model.predict(negative_review)

    generator = PlayGenerator()
    generator.build_model(generator.VOCAB_SIZE, generator.EMBEDDING_DIM, generator.RNN_UNITS, generator.BATCH_SIZE)
    #generator.model.summary()
    generator.model.compile(optimizer='adam', loss=generator.loss)
    checkpoint_dir = 'training_checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_prefix = os.path.join(checkpoint_dir, "chkpt_{epoch}.weights.h5")

    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_prefix,
        save_weights_only=True
    )

    history = generator.model.fit(generator.data, epochs=40, callbacks=[checkpoint_callback])

    model = generator.build_model(generator.VOCAB_SIZE, generator.EMBEDDING_DIM, generator.RNN_UNITS, batch_size=1)
    model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
    model.build(tf.TensorShape([1, None]))
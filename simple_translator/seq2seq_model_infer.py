import tensorflow as tf
import numpy as np
import sys
from tensorflow.contrib.seq2seq import GreedyEmbeddingHelper, dynamic_decode
from tensorflow.contrib.rnn import LSTMCell
from tensorflow.python.layers import core as layers_core
from simple_translator.load_language_data import get_data, get_embeddings
import os.path

'''
When building a machine learning model in TensorFlow, it's often best to build three separate graphs [...]
(train, evaluate, infer)
The inference graph is usually very different from the other two, so it makes sense to build it separately.

Source: https://github.com/tensorflow/nmt
'''


class seq2seq_model_infer:
    def __init__(self, word_index_1, word_index_2, weights, encoder_hidden_units=500, decoder_hidden_units=500,
                 embedding_size=300, batch_size=1):
        self.batch_size = batch_size

        # Get vocab length
        vocab_len_1 = len(word_index_1)
        vocab_len_2 = len(word_index_2)

        # Make word indices available in model
        self.word_index_1 = word_index_1
        self.word_index_2 = word_index_2

        # Make reverse dicts
        self.rev_word_index_1 = {v: k for k, v in word_index_1.items()}
        self.rev_word_index_2 = {v: k for k, v in word_index_2.items()}

        # Session
        tf.reset_default_graph()
        self.sess = tf.InteractiveSession()

        # Placeholders
        self.encoder_inputs = tf.placeholder(shape=(None, None), dtype=tf.int32, name='encoder_inputs')
        self.encoder_inputs_length = tf.placeholder(shape=(None,), dtype=tf.int32, name='encoder_inputs_length')
        self.go_symbol = tf.placeholder(shape=(None,), dtype=tf.int32, name='go_symbol')

        # Variable embedding, error when placing on gpu (i don't know why)
        with tf.device('/cpu:0'):
            embeddings_english = tf.Variable(tf.random_uniform([vocab_len_1, embedding_size], -1.0, 1.0),
                                             dtype=tf.float32, name="embeddings1")
            embeddings_german = tf.Variable(tf.random_uniform([vocab_len_2, embedding_size], -1.0, 1.0),
                                            dtype=tf.float32, name="embeddings2")

        # Embedding lookups for encoder and decoder
        encoder_inputs_embedded = tf.nn.embedding_lookup(embeddings_english, self.encoder_inputs)

        # Encoder: Single LSTM Cell
        with tf.variable_scope("LSTM_Encoder"):
            encoder_cell = LSTMCell(encoder_hidden_units)
            # encoder_cell2 = LSTMCell(encoder_hidden_units)
            (outputs, self.encoder_final_state) = tf.nn.dynamic_rnn(encoder_cell, inputs=encoder_inputs_embedded,
                                                                    sequence_length=self.encoder_inputs_length,
                                                                    dtype=tf.float32,
                                                                    time_major=True)

        helper = GreedyEmbeddingHelper(embeddings_german, self.go_symbol, word_index_2['<EOS>'])

        # Projection Layer brings output from hidden layer count to count of unique classes (eg. 50 -> 10)
        projection_layer = layers_core.Dense(vocab_len_2, use_bias=False)

        # Define Basic Decoder
        decoder = tf.contrib.seq2seq.BasicDecoder(
            LSTMCell(decoder_hidden_units), helper, self.encoder_final_state, output_layer=projection_layer)

        # Dynamic decoding to handle Basic Decoder
        (self.final_outputs, final_state, final_sequence_lengths) = dynamic_decode(decoder, output_time_major=True
                                                                                   , maximum_iterations=300)

        # Initialize all variables
        self.sess.run(tf.global_variables_initializer())

        # For Saving weights
        self.saver = tf.train.Saver()

        # Load weights if exist
        try:
            self.saver.restore(self.sess, self.weights)
            print("Loading weights.")
        except:
            print("No weights found.")

    def predict(self):
        while True:
            text = input("Input (English): ")
            text = text.split()
            text_len = [len(text)]

            text_to_number = [self.word_index_1[word] if word in self.word_index_1 else self.word_index_1['<UNK>'] for word
                              in text]

            text_to_number = tf.contrib.keras.preprocessing.sequence.pad_sequences([text_to_number], maxlen=30,
                                                                                   padding='post')

            fd = {
                self.encoder_inputs: np.array(text_to_number).T,
                self.encoder_inputs_length: np.array(text_len),
                self.go_symbol: np.array([self.word_index_2['<GO>']])
            }

            # Get prediction
            output = self.sess.run(self.final_outputs.sample_id, fd)
            print(output.T)
            output = " ".join([self.rev_word_index_2[word] for word in output.T[0]])

            print(output)

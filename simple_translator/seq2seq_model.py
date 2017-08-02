import tensorflow as tf
import numpy as np
import sys
from tensorflow.contrib.seq2seq import TrainingHelper, dynamic_decode
from tensorflow.contrib.rnn import LSTMCell
from tensorflow.python.layers import core as layers_core
from simple_translator.load_language_data import get_data, get_embeddings
import os.path


class seq2seq_model:
    def __init__(self, word_index_1, word_index_2, weights, encoder_hidden_units=500, decoder_hidden_units=500,
                 embedding_size=300, batch_size=100):

        self.batch_size = batch_size
        self.weights = weights

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

        # Placeholders for Encoder
        self.encoder_inputs = tf.placeholder(shape=(None, None), dtype=tf.int32, name='encoder_inputs')
        self.encoder_inputs_length = tf.placeholder(shape=(None,), dtype=tf.int32, name='encoder_inputs_length')

        # Placeholders for Decoder
        self.decoder_target_input = tf.placeholder(shape=(None, None), dtype=tf.int32, name='decoder_target_input')
        self.decoder_target_output = tf.placeholder(shape=(None, None), dtype=tf.int32, name='decoder_target_output')
        self.decoder_target_length = tf.placeholder(shape=(None,), dtype=tf.int32, name='decoder_target_length')

        # Variable embedding, error when placing on gpu (i don't know why)
        with tf.device('/cpu:0'):
            embeddings_english = tf.Variable(tf.random_uniform([vocab_len_1, embedding_size], -1.0, 1.0),
                                             dtype=tf.float32, name="embeddings1")
            embeddings_german = tf.Variable(tf.random_uniform([vocab_len_2, embedding_size], -1.0, 1.0),
                                            dtype=tf.float32, name="embeddings2")

        # Embedding lookups for encoder and decoder
        encoder_inputs_embedded = tf.nn.embedding_lookup(embeddings_english, self.encoder_inputs)
        decoder_inputs_embedded = tf.nn.embedding_lookup(embeddings_german, self.decoder_target_input)

        # Encoder: Single LSTM Cell
        with tf.variable_scope("LSTM_Encoder"):
            encoder_cell = LSTMCell(encoder_hidden_units)
            # encoder_cell2 = LSTMCell(encoder_hidden_units)
            (outputs, self.encoder_final_state) = tf.nn.dynamic_rnn(encoder_cell, inputs=encoder_inputs_embedded,
                                                                    sequence_length=self.encoder_inputs_length,
                                                                    dtype=tf.float32,
                                                                    time_major=True)

        # Training Helper for Decoder Training
        helper = TrainingHelper(decoder_inputs_embedded, self.decoder_target_length, time_major=True)

        # Projection Layer brings output from hidden layer count to count of unique classes (eg. 50 -> 10)
        projection_layer = layers_core.Dense(vocab_len_2, use_bias=False)

        # Define Basic Decoder
        decoder = tf.contrib.seq2seq.BasicDecoder(
            LSTMCell(decoder_hidden_units), helper, self.encoder_final_state, output_layer=projection_layer)

        # Dynamic decoding to handle Basic Decoder
        self.final_outputs, final_state = dynamic_decode(decoder, output_time_major=True)

        target_weights = tf.cast(tf.greater(self.decoder_target_output, 0), tf.float32)

        # Cross entropy Loss
        self.cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
            labels=tf.one_hot(self.decoder_target_output, depth=vocab_len_2, dtype=tf.float32),
            logits=self.final_outputs.rnn_output,
        )

        # Loss masked padding with target_weights
        self.loss = tf.reduce_sum(self.cross_entropy * target_weights) / batch_size

        # OR: simple mean loss for all batches
        # self.loss = tf.reduce_mean(self.cross_entropy)

        # Train with adam optimizer
        self.train_op = tf.train.AdamOptimizer().minimize(self.loss)

        # Initialize all variables
        self.sess.run(tf.global_variables_initializer())

        # For Saving weights
        self.saver = tf.train.Saver()

        if os.path.isfile(weights + ".index"):
            # Load weights if exist
            self.saver.restore(self.sess, self.weights)
            print("Loading weights.")
        else:
            print("No weights found.")

    def train(self, iterations, src_data, src_len, tgt_input, tgt_output, tgt_len, train_ratio=0.8):

        # Compute dataset sizes for train and eval
        train_size = int(tgt_input.shape[0] * train_ratio)

        ids = np.arange(0, tgt_input.shape[0])
        np.random.shuffle(ids)
        train_ids = ids[:train_size]
        eval_ids = ids[train_size:]

        # Split train data
        train_src_data = src_data[train_ids]
        train_src_len = src_len[train_ids]
        train_tgt_input = tgt_input[train_ids]
        train_tgt_output = tgt_output[train_ids]
        train_tgt_len = tgt_len[train_ids]

        # Split eval data
        eval_src_data = src_data[eval_ids]
        eval_src_len = src_len[eval_ids]
        eval_tgt_input = tgt_input[eval_ids]
        eval_tgt_output = tgt_output[eval_ids]
        eval_tgt_len = tgt_len[eval_ids]

        for batch in range(iterations):

            # Get train batch
            ids = np.random.choice(train_src_data.shape[0], self.batch_size, replace=False)

            # batch data
            train_batch_src_data = train_src_data[ids].T
            train_batch_tgt_input = train_tgt_input[ids].T
            train_batch_tgt_output = train_tgt_output[ids].T

            # batch seq len
            train_batch_src_len = train_src_len[ids]
            train_batch_tgt_len = train_tgt_len[ids]

            # Feed dict
            fd = {
                self.encoder_inputs: train_batch_src_data,
                self.encoder_inputs_length: train_batch_src_len,
                self.decoder_target_input: train_batch_tgt_input,
                self.decoder_target_output: train_batch_tgt_output,
                self.decoder_target_length: train_batch_tgt_len
            }

            # train net
            _ = self.sess.run(self.train_op, fd)

            if batch % 100 == 0:
                # Get loss and examples for train data
                train_loss, train_output = self.sess.run([self.loss, self.final_outputs.sample_id], fd)

                # Get eval batch
                ids = np.random.choice(eval_src_data.shape[0], self.batch_size, replace=False)

                # batch data
                eval_batch_src_data = eval_src_data[ids].T
                eval_batch_tgt_input = eval_tgt_input[ids].T
                eval_batch_tgt_output = eval_tgt_output[ids].T

                # batch seq len
                eval_batch_src_len = eval_src_len[ids]
                eval_batch_tgt_len = eval_tgt_len[ids]

                # Feed dict with eval data
                eval_fd = {
                    self.encoder_inputs: eval_batch_src_data,
                    self.encoder_inputs_length: eval_batch_src_len,
                    self.decoder_target_input: eval_batch_tgt_input,
                    self.decoder_target_output: eval_batch_tgt_output,
                    self.decoder_target_length: eval_batch_tgt_len
                }

                # Get loss and examples for train data
                eval_loss, eval_output = self.sess.run([self.loss, self.final_outputs.sample_id], eval_fd)

                print("Epoch", batch)
                print("Train loss:", train_loss)
                print("Eval loss:", eval_loss)
                print("Examples (train):")
                for i in range(3):
                    mapped_seq1 = " ".join([self.rev_word_index_1[key] for key in train_batch_src_data.T[i]])
                    mapped_seq2 = " ".join([self.rev_word_index_2[key] for key in train_output.T[i]])
                    print("Input    ", i, " :", mapped_seq1)
                    print("Output   ", i, " :", mapped_seq2)
                    print()
                print("Examples (eval):")
                for i in range(3):
                    mapped_seq1 = " ".join([self.rev_word_index_1[key] for key in eval_batch_src_data.T[i]])
                    mapped_seq2 = " ".join([self.rev_word_index_2[key] for key in eval_output.T[i]])
                    print("Input    ", i, " :", mapped_seq1)
                    print("Output   ", i, " :", mapped_seq2)
                    print()

                # Save the variables to disk.

                self.saver.save(self.sess, self.weights)

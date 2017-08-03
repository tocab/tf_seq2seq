import os.path

import numpy as np
import tensorflow as tf
import time
import sys
from tensorflow.contrib.rnn import LSTMCell
from tensorflow.contrib.seq2seq import TrainingHelper, dynamic_decode
from tensorflow.python.layers import core as layers_core


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

    def train(self, iterations, src_data, src_len, tgt_input, tgt_output, tgt_len, bucketing=1):

        bucket_list = [[src_data, src_len, tgt_input, tgt_output, tgt_len]]

        # If bucketing, assign bucket to seq
        if bucketing > 1:
            # lower batch size to have the same batch size on every epoch and every bucket size
            self.batch_size = int(round(self.batch_size / bucketing))

            # Create buckets
            bucket_list = self.create_bucketing(src_data, src_len, tgt_input, tgt_output, tgt_len, bucketing)

        start_time = time.clock()

        for batch in range(iterations):

            if batch % 100 == 0:
                print("Epoch", batch)

            for bucket_number, bucket in enumerate(bucket_list):

                # Get train batch
                ids = np.random.choice(bucket[0].shape[0], self.batch_size, replace=False)

                src_data_batch = bucket[0][ids].T
                src_len_batch = bucket[1][ids].T
                tgt_input_batch = bucket[2][ids].T
                tgt_output_batch = bucket[3][ids].T
                tgt_len_batch = bucket[4][ids].T

                # Feed dict
                fd = {
                    self.encoder_inputs: src_data_batch,
                    self.encoder_inputs_length: src_len_batch,
                    self.decoder_target_input: tgt_input_batch,
                    self.decoder_target_output: tgt_output_batch,
                    self.decoder_target_length: tgt_len_batch
                }

                # train net
                _ = self.sess.run(self.train_op, fd)

                if batch % 100 == 0:
                    print("Bucket", bucket_number)
                    # Get loss and examples for train data
                    bucket_loss, bucket_output = self.sess.run([self.loss, self.final_outputs.sample_id], fd)
                    print("Loss:", bucket_loss)
                    print("Example:")
                    for i in range(1):
                        mapped_seq1 = " ".join([self.rev_word_index_1[key] for key in src_data_batch.T[i]])
                        mapped_seq2 = " ".join([self.rev_word_index_2[key] for key in bucket_output.T[i]])
                        print("Input    ", i, " :", mapped_seq1)
                        print("Output   ", i, " :", mapped_seq2)
                        print()

                        # Save the variables to disk.

                if batch % 100 == 0:
                    self.saver.save(self.sess, self.weights)

        end_time = time.clock() - start_time
        print(iterations, "epochs after", end_time, "seconds")
        print("That are", end_time/iterations, "seconds per iteration.")

    def create_bucketing(self, src_data, src_len, tgt_input, tgt_output, tgt_len, bucketing):

        # Get size for each bucket
        bucket_size = int(max(src_len) / bucketing) + 1

        # Create a list where each dataset is assigned to a bucket
        bucket_assign = []
        for i in range(len(src_len)):
            input_len = src_len[i]
            output_len = tgt_len[i]

            # Decide in which dataset a bucket is sorted
            for bucket in range(1, bucketing + 1):

                bucket_len = bucket_size * bucket

                if input_len <= bucket_len and output_len <= bucket_len:
                    bucket_assign.append(bucket - 1)
                    break

        # Initialize bucket list
        bucket_list = []
        for i in range(bucketing):
            bucket_list.append([[], [], [], [], []])

        # Assign data to bucket list
        for i, bucket in enumerate(bucket_assign):
            src_data_buc = src_data[i, :bucket_size * (bucket + 1)]
            tgt_input_buc = tgt_input[i, :bucket_size * (bucket + 1)]
            tgt_output_buc = tgt_output[i, :bucket_size * (bucket + 1)]

            bucket_list[bucket][0].append(src_data_buc)
            bucket_list[bucket][1].append(src_len[i])
            bucket_list[bucket][2].append(tgt_input_buc)
            bucket_list[bucket][3].append(tgt_output_buc)
            bucket_list[bucket][4].append(tgt_len[i])

        # Convert to np array
        for i, bucket in enumerate(bucket_list):
            for j, data in enumerate(bucket):
                bucket_list[i][j] = np.array(bucket_list[i][j])

        return bucket_list

import tensorflow as tf
import numpy as np
import sys
from tensorflow.contrib.rnn import LSTMCell
from tensorflow.python.layers import core as layers_core


def main():
    # Settings
    tf.reset_default_graph()
    sess = tf.InteractiveSession()

    # For batches
    batch_size = 100
    input_seq_len = 10
    target_seq_len = 15

    # Size of vocabulary and embedding
    vocab_size = 10
    input_embedding_size = 50

    # Hidden units of encoder and decoder
    encoder_hidden_units = 50
    decoder_hidden_units = 50

    # Placeholders
    encoder_inputs = tf.placeholder(shape=(None, None), dtype=tf.int32, name='encoder_inputs')
    encoder_inputs_length = tf.placeholder(shape=(None,), dtype=tf.int32, name='encoder_inputs_length')
    decoder_targets = tf.placeholder(shape=(None, None), dtype=tf.int32, name='decoder_targets')

    # Variable embedding, error when placing on gpu (i don't know why)
    with tf.device('/cpu:0'):
        embeddings = tf.Variable(tf.random_uniform([vocab_size, input_embedding_size], -1.0, 1.0), dtype=tf.float32)

    # Embedding lookups for encoder and decoder
    encoder_inputs_embedded = tf.nn.embedding_lookup(embeddings, encoder_inputs)
    decoder_inputs_embedded = tf.nn.embedding_lookup(embeddings, decoder_targets)

    # Encoder: Single LSTM Cell
    with tf.variable_scope("LSTM_Encoder") as vs:
        encoder_cell = LSTMCell(encoder_hidden_units)
        # encoder_cell2 = LSTMCell(encoder_hidden_units)
        (outputs, encoder_final_state) = tf.nn.dynamic_rnn(encoder_cell, inputs=encoder_inputs_embedded,
                                                           sequence_length=encoder_inputs_length, dtype=tf.float32,
                                                           time_major=True)

    # Decoder length should be a litte longer because it has the EOS at the end
    decoder_lengths = encoder_inputs_length + (target_seq_len - input_seq_len + 1)

    # Training Helper for Decoder Training
    helper = tf.contrib.seq2seq.TrainingHelper(decoder_inputs_embedded, decoder_lengths, time_major=True)

    # Projection Layer brings output from hidden layer count to count of unique classes (eg. 50 -> 10)
    projection_layer = layers_core.Dense(vocab_size, use_bias=False)

    # Define Basic Decoder
    decoder = tf.contrib.seq2seq.BasicDecoder(
        LSTMCell(decoder_hidden_units), helper, encoder_final_state, output_layer=projection_layer)

    # Dynamic decoding to handle Basic Decoder
    (final_outputs, final_state, final_sequence_lengths) = tf.contrib.seq2seq.dynamic_decode(decoder,
                                                                                             output_time_major=True)

    # Cross entropy Loss
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
        labels=tf.one_hot(decoder_targets, depth=vocab_size, dtype=tf.float32),
        logits=final_outputs.rnn_output,
    )

    # Mean loss for all batches
    loss = tf.reduce_mean(cross_entropy)

    # Train with adam optimizer
    train_op = tf.train.AdamOptimizer().minimize(loss)

    # Initialize all variables
    sess.run(tf.global_variables_initializer())

    max_batches = 3001

    try:
        for batch in range(max_batches):
            # Get batch
            encoder_input, encoder_inputs_lengths, decoder_target = createBatch(batch_size, input_seq_len, vocab_size,
                                                                                target_seq_len)

            # Feed dict
            fd = {
                encoder_inputs: encoder_input,
                encoder_inputs_length: encoder_inputs_lengths,
                decoder_targets: decoder_target,
            }

            # train net
            _ = sess.run(train_op, fd)

            if batch % 1000 == 0:
                # Get loss and examples
                train_loss, output = sess.run([loss, final_outputs.sample_id], fd)

                print("Epoch", batch)
                print("Loss:", train_loss)
                print("Examples:")
                for i in range(3):
                    print("Input    ", i, " :", encoder_input.T[i])
                    print("Output   ", i, " :", output.T[i])
                    print()

    except KeyboardInterrupt:
        print('training interrupted')


def createBatch(batch_size, seq_len, vocab_size, target_len, time_major=True):
    # Symbol for PAD (padding of a sequence) and end of sentence (EOS)
    PAD = 0
    EOS = 1

    # First define length of every sequence
    lengths = np.random.randint(1, seq_len, batch_size)

    # Create encoder input, begin at 2 because 0 and 1 are reserved for PAD and EOS
    encoder_inputs = []
    decoder_targets = []
    for i in range(batch_size):
        rand_seq = np.random.randint(2, vocab_size, lengths[i]).tolist()
        encoder_inputs.append(rand_seq + [PAD] * (seq_len - len(rand_seq)))
        decoder_targets.append(rand_seq + [EOS] + [0] * (target_len - len(rand_seq) - 1))

    # to np array
    encoder_inputs = np.array(encoder_inputs)
    decoder_targets = np.array(decoder_targets)

    if time_major:
        encoder_inputs = encoder_inputs.T
        decoder_targets = decoder_targets.T

    return encoder_inputs, lengths, decoder_targets


if __name__ == "__main__":
    main()
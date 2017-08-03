# tf_seq2seq
Some seq2seq experiments with tensorflow

Added a simple seq2seq example for Tensorflow 1.1.0.

Simple sequence prediction: <br />
A random sequence of integers as input with the goal for the model to predict the same sequence as output.

Simple translator: <br />
Input is a source language and target is to predict text into a target language. 
For using it, change paths in seq2seq_translator.py file and run it!

Bucketing: <br />
Comparison of model with and without bucketing with 500 epochs: <br />
* without bucketing: 478.1 seconds training time
* bucketing with 2 buckets: 453.8 seconds training time

Seems that bucketing is needless in TensorFlow, maybe because the TrainingHelper
 and dynamic_rnn for the encoder already have information about the 
 sequence length and stop after it is reached.


Credit to [ematvey](https://github.com/ematvey/tensorflow-seq2seq-tutorials) for giving a good starting point for this.
import sys
import tensorflow as tf
import numpy as np
import spacy
from simple_translator import Tokenizer


def get_data(src_data_path, target_data_path, num_known_words):
    max_seq_len = 30

    with open(src_data_path, encoding="utf-8") as f:
        src_data = f.readlines()

    with open(target_data_path, encoding="utf-8") as f:
        target_data = f.readlines()

    tokenizer_source = Tokenizer.Tokenizer(num_words=num_known_words)
    tokenizer_target = Tokenizer.Tokenizer(num_words=num_known_words, for_output=True)

    tokenizer_source.fit_on_texts(src_data)
    tokenizer_target.fit_on_texts(target_data)

    sequences_source, length_source = tokenizer_source.texts_to_sequences(src_data, max_len=max_seq_len)
    sequences_target, length_target = tokenizer_target.texts_to_sequences(target_data, max_len=max_seq_len)

    data_source = tf.contrib.keras.preprocessing.sequence.pad_sequences(sequences_source, maxlen=max_seq_len,
                                                                        padding='post')
    # Here we have also the <GO> and <EOS> token contained. So padding with max_seq_len+2
    data_target = tf.contrib.keras.preprocessing.sequence.pad_sequences(sequences_target, maxlen=max_seq_len+2,
                                                                        padding='post')

    # Split target sequences in target input and target output
    # Final data will have sequence length of max_seq_len + 1
    # Example for target input: <GO> Dies ist eine Übersetzung von Englisch zu Deutsch
    # Example for target output: Dies ist eine Übersetzung von Englisch zu Deutsch <EOS>
    data_target_input = data_target[:, :-1]
    data_target_output = data_target[:, 1:]

    # Because of <GO> and <EOS> in target data, add one to all length target numbers
    length_target += 1

    # pack final data into two lists
    source_data = [data_source, length_source, tokenizer_source.word_dict]
    target_data = [data_target_input, data_target_output, length_target, tokenizer_target.word_dict]

    return source_data, target_data


def get_embeddings(language):
    """
    Unused at the moment, later possibility to use pre-trained weights
    :param language:
    :return:
    """

    nlp = spacy.load(language, parser=False, tagger=False, entity=False)

    vocab = nlp.vocab

    max_rank = max(lex.rank for lex in vocab if lex.has_vector)
    vectors = np.ndarray((max_rank + 1, vocab.vectors_length), dtype='float32')
    for lex in vocab:
        if lex.has_vector:
            vectors[lex.rank] = lex.vector
        else:
            print("word", lex.text, "has no vector")
    return vectors


if __name__ == "__main__":
    """
    Only for testing the output
    """
    data_german, length_german, tokenizer_german_word_index, data_english, length_english, tokenizer_english_word_index = get_data()

    print(tokenizer_german_word_index)
    print(np.array(data_german))

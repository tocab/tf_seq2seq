from collections import Counter
import sys
import numpy as np


class Tokenizer:
    def __init__(self, num_words=None, for_output=False):
        self.num_words = num_words
        self.for_output = for_output

        self.word_dict = {'<PAD>': 0, '<UNK>': 1, '<GO>': 2, '<EOS>': 3}

    def fit_on_texts(self, text_list):
        words = []

        for i, line in enumerate(text_list):
            line_words = line.split()
            words += line_words

        most_common = Counter(words).most_common(self.num_words)

        for i, word_counts in enumerate(most_common):
            self.word_dict[word_counts[0]] = i + 4

    def texts_to_sequences(self, text_list, max_len):

        seq_texts = []
        len_texts = []

        for i, line in enumerate(text_list):
            line_words = line.split()
            line_words = line_words[:max_len]
            len_texts.append(len(line_words))
            line_seq = [self.word_dict[word] if word in self.word_dict else self.word_dict['<UNK>'] for word in
                        line_words]

            # Set GO and EOS token if this is a output sentence
            if self.for_output:
                line_seq = [self.word_dict['<GO>']] + line_seq + [self.word_dict['<EOS>']]

            seq_texts.append(line_seq)

        return seq_texts, np.array(len_texts)

import numpy as np
import nltk
import sys


def bleu_score(tgt_input, tgt_output, time_major=True):

    if time_major:
        tgt_input = tgt_input.T
        tgt_output = tgt_output.T

    unpadded_input = []
    unpadded_output = []

    PAD = 0
    EOS = 3

    for i in range(len(tgt_input)):
        # input always begins with an GO Symbol and ends with PAD
        # Find first pad
        first_pad = np.where(tgt_input[i] == PAD)

        # Take Sequence to first PAD, skip Go
        if len(first_pad[0]) > 0:
            min_index = np.min(first_pad)
            unpadded_input.append([tgt_input[i, 1:min_index]])
        else:
            unpadded_input.append([tgt_input[i, 1:]])

        # For output sequence find first EOS
        first_eos = np.where(tgt_output[i] == EOS)

        # Take everything from beginning to EOS
        if len(first_eos[0]) > 0:
            min_index = np.min(first_eos)
            unpadded_output.append(tgt_output[i, :min_index])
        else:
            unpadded_output.append(tgt_output[i])

    # Calculate BLEU score with NLTK
    bleu = nltk.translate.bleu_score.corpus_bleu(unpadded_input, unpadded_output)

    return bleu
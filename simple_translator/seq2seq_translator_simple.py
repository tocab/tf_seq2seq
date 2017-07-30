import sys
from simple_translator.load_language_data import get_data, get_embeddings
from simple_translator import seq2seq_model, seq2seq_model_infer
import pickle
import os.path

# Data paths, a link to download train.tok.clean.bpe.32000.en and train.tok.clean.bpe.32000.de can be found at
# https://google.github.io/seq2seq/nmt/
SAVE_PICKLE_PATH = "english_german_dataset/simple_model/data.p"
SOURCE_DATA_PATH = "english_german_dataset/wmt16_en_de/train.tok.clean.bpe.32000.en"
TARGET_DATA_PATH = "english_german_dataset/wmt16_en_de/train.tok.clean.bpe.32000.de"
WEIGHTS_PATH = "weights/seq2seq_model.ckpt"
MODE = "TRAIN"  # OR "INFER"
# Most frequent X words that are used from each corpus. The other words will be marked with <UNK> (unknown) Symbol
# None = use all words
MAX_WORDS = 15000

if __name__ == "__main__":

    if not os.path.isfile(SAVE_PICKLE_PATH):
        data = get_data(SOURCE_DATA_PATH, TARGET_DATA_PATH, MAX_WORDS)
        pickle.dump(data, open(SAVE_PICKLE_PATH, "wb"))
    else:
        data = pickle.load(open(SAVE_PICKLE_PATH, "rb"))

    # Unpack data
    source_data, target_data = data

    # Unpack source data
    src_data = source_data[0]
    src_data_len = source_data[1]
    src_data_wordindex = source_data[2]

    # Unpack target data
    tgt_data_input = target_data[0]
    tgt_data_output = target_data[1]
    tgt_data_len = target_data[2]
    tgt_data_wordindex = target_data[3]

    weights = WEIGHTS_PATH

    if MODE == "TRAIN":
        model = seq2seq_model.seq2seq_model(src_data_wordindex, tgt_data_wordindex, weights=WEIGHTS_PATH)
        model.train(10000, src_data, src_data_len, tgt_data_input, tgt_data_output, tgt_data_len)
    elif MODE == "INFER":
        model = seq2seq_model_infer.seq2seq_model_infer(src_data_wordindex, tgt_data_wordindex, weights=WEIGHTS_PATH)
        model.predict()

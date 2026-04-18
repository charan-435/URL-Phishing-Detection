import argparse
import sys
import os
from feature_extraction import FeatureExtractor
from models.brnn.brnn_base import BrnnBase
from models.brnn.brnn_complex import BrnnComplex

# train brnn locally
def run_train(args):
    # load small data
    fe = FeatureExtractor(char_index_path="dataset/char_index")
    fe.load_from_file("dataset/train/small_train.txt")

    x = fe.get_sequences(sequence_length=args.sequence_length)
    y = fe.get_labels()

    # pick model
    if args.model == "brnn_base":
        builder = BrnnBase(args.embed_dim, args.sequence_length)
    else:
        builder = BrnnComplex(args.embed_dim, args.sequence_length)

    model = builder.build(fe.tokener.word_index)
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

    # fitting
    model.fit(x, y, epochs=args.epochs, batch_size=args.batch_size, validation_split=0.2)
    model.save(f"models/brnn/{args.model}.keras")
    print("done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="brnn_base")
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--sequence_length", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    run_train(parser.parse_args())
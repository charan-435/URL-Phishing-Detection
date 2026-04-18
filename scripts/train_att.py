import argparse
import sys
import os
from feature_extraction import FeatureExtractor
from models.att.att_base import AttBase
from models.att.att_complex import AttComplex

# train attention models
def run_train(args):
    # load data
    fe = FeatureExtractor(char_index_path="dataset/char_index")
    fe.load_from_file("dataset/train/train.txt")

    x = fe.get_sequences(sequence_length=args.sequence_length)
    y = fe.get_labels()

    # choose model
    if args.model == "att_base":
        builder = AttBase(args.embed_dim, args.sequence_length)
    else:
        builder = AttComplex(args.embed_dim, args.sequence_length)

    model = builder.build(fe.tokener.word_index)
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

    # fitting
    model.fit(x, y, epochs=args.epochs, batch_size=args.batch_size, validation_split=0.2)
    model.save(f"models/att/{args.model}.keras")
    print("done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="att_base")
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--sequence_length", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    run_train(parser.parse_args())
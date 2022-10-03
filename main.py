# -*- coding: utf-8 -*-
import argparse
import os
import random
import string
import sys

import config
from config import slices_path, slice_size, batch_size, files_per_genre
from config import validation_ratio, test_ratio
from Source.model import create_model, load_model
from Source.song_to_data import create_slices_from_audio
from Source.test_functions import test_3
from Source.train_functions import train_3

parser = argparse.ArgumentParser()
parser.add_argument("mode", help="Trains or tests the CNN", nargs='+', choices=["train", "test", "slice"])
args = parser.parse_args()

print("--------------------------")
print("| ** Config ** ")
print("| Validation ratio: {}".format(validation_ratio))
print("| Test ratio: {}".format(test_ratio))
print("| Slices per genre: {}".format(files_per_genre))
print("| Slice size: {}".format(slice_size))
print("--------------------------")

if __name__ == "__main__":
    if "slice" in args.mode:
        create_slices_from_audio()
        sys.exit()

    # List genres
    genres = os.listdir(slices_path)
    genres = [filename for filename in genres if os.path.isdir(slices_path + filename)]
    nb_classes = len(genres)

    if config.no_cuda:
        print('Disabling CUDA support for TensorFlow...')
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    # Create model
    model = create_model(nb_classes, slice_size)

    # Noe: We can do both train and test at once
    if "train" in args.mode:
        # Define run id for graphs
        run_id = "MusicGenres - " + str(batch_size) + " " + ''.join(
            random.SystemRandom().choice(string.ascii_uppercase) for _ in range(10))

        # Train the model
        print("[+] Training the model...")
        train_3()
        print("    Model trained! ✅")

    if "test" in args.mode:
        load_model(model)
        test_3()
        # test_2
        # CPU: 55.5s
        # CUDA: 17.5s

        # test_3
        # CUDA: 7.2s
        # CPU: 52.4s

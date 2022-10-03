# -*- coding: utf-8 -*-
import multiprocessing
import os
import pickle
import time
from random import shuffle

# Remove logs
import eyed3
import numpy as np
from PIL import Image

from Source import CustomGenerator

eyed3.log.setLevel("ERROR")

# from tools import get_image_data
from config import dataset_path, slices_path, selected_genres, training_splits, slice_size, \
    test_splits, batch_size


### Audio tools ###
def is_mono(filename):
    audiofile = eyed3.load(filename)
    return audiofile.info.mode == 'Mono'


def get_bitrate(filename):
    audiofile = eyed3.load(filename)
    return audiofile.info.bit_rate[1]


### Image tools ###
def get_processed_data(img, img_size):
    """Returns numpy image at size img_size*img_size"""
    img = img.resize((img_size, img_size), resample=Image.ANTIALIAS)
    img_data = np.asarray(img, dtype=np.uint8).reshape(img_size, img_size, 1)
    img_data = img_data / 255.
    return img_data


def get_image_data(filename, img_size):
    """Returns numpy image at size img_size * img_size"""
    img = Image.open(filename)
    img_data = get_processed_data(img, img_size)
    return img_data


### Dataset tools ###
def get_dataset_name(nb_per_genre, slice_size):
    """Creates name of dataset from parameters"""
    name = "{}".format(nb_per_genre)
    name += "_{}".format(slice_size)
    return name


def get_dataset(nb_per_genre, genres, slice_size, validation_ratio, test_ratio, mode):
    """Creates or loads dataset if it exists, note: Mode is train or test"""
    print("[+] Dataset name: {}".format(get_dataset_name(nb_per_genre, slice_size)))
    if not os.path.isfile(dataset_path + "train_X_" + get_dataset_name(nb_per_genre, slice_size) + ".p"):
        print("[+] Creating dataset with {} slices of size {} per genre... ⌛️".format(nb_per_genre, slice_size))
        create_dataset_from_slices(nb_per_genre, genres, slice_size, validation_ratio, test_ratio)
    else:
        print("[+] Using existing dataset")

    return load_dataset(nb_per_genre, genres, slice_size, mode)


def load_dataset(nb_per_genre, genres, slice_size, mode):
    # Load existing
    dataset_name = get_dataset_name(nb_per_genre, slice_size)
    if mode == "train":
        print("[+] Loading training and validation datasets... ")
        train_X = pickle.load(open("{}train_X_{}.p".format(dataset_path, dataset_name), "rb"))
        train_y = pickle.load(open("{}train_y_{}.p".format(dataset_path, dataset_name), "rb"))
        validation_X = pickle.load(open("{}validation_X_{}.p".format(dataset_path, dataset_name), "rb"))
        validation_y = pickle.load(open("{}validation_y_{}.p".format(dataset_path, dataset_name), "rb"))
        print("    Training and validation datasets loaded! ✅")
        return train_X, train_y, validation_X, validation_y

    else:
        print("[+] Loading testing dataset... ")
        test_X = pickle.load(open("{}test_X_{}.p".format(dataset_path, dataset_name), "rb"))
        test_y = pickle.load(open("{}test_y_{}.p".format(dataset_path, dataset_name), "rb"))
        print("    Testing dataset loaded! ✅")
        return test_X, test_y


def save_dataset(train_X, train_y, validation_X, validation_y, test_X, test_y, nb_per_genre, genres, slice_size):
    # Create path for dataset if doesn't exist
    os.makedirs(os.path.dirname(dataset_path), exist_ok=True)

    # save_dataset
    print("[+] Saving dataset... ")
    dataset_name = get_dataset_name(nb_per_genre, slice_size)
    pickle.dump(train_X, open("{}train_X_{}.p".format(dataset_path, dataset_name), "wb"))
    pickle.dump(train_y, open("{}train_y_{}.p".format(dataset_path, dataset_name), "wb"))
    pickle.dump(validation_X, open("{}validation_X_{}.p".format(dataset_path, dataset_name), "wb"))
    pickle.dump(validation_y, open("{}validation_y_{}.p".format(dataset_path, dataset_name), "wb"))
    pickle.dump(test_X, open("{}test_X_{}.p".format(dataset_path, dataset_name), "wb"))
    pickle.dump(test_y, open("{}test_y_{}.p".format(dataset_path, dataset_name), "wb"))
    print("    Dataset saved! ✅💾")


def create_dataset_from_slices(nb_per_genre, genres, slice_size, validation_ratio, test_ratio):
    """Creates and save dataset from slices"""
    data = []
    for genre in genres:
        print("-> Adding {}...".format(genre))
        # Get slices in genre subfolder
        filenames = os.listdir(slices_path + genre)
        filenames = [filename for filename in filenames if filename.endswith('.png')]
        if nb_per_genre != -1:
            filenames = filenames[:nb_per_genre]
        # Randomize file selection for this genre
        shuffle(filenames)

        # Add data (X,y)
        for filename in filenames:
            img_data = get_image_data(slices_path + genre + "/" + filename, slice_size)
            label = [1. if genre == g else 0. for g in genres]
            data.append((img_data, label))

    # Shuffle data
    shuffle(data)

    # Extract X and y
    X, y = zip(*data)

    # Split data
    validation_nb = int(len(X) * validation_ratio)
    test_nb = int(len(X) * test_ratio)
    train_nb = len(X) - (validation_nb + test_nb)

    # Prepare for Tflearn at the same time
    train_X = np.array(X[:train_nb]).reshape([-1, slice_size, slice_size, 1])
    train_y = np.array(y[:train_nb])
    validation_X = np.array(X[train_nb:train_nb + validation_nb]).reshape([-1, slice_size, slice_size, 1])
    validation_y = np.array(y[train_nb:train_nb + validation_nb])
    test_X = np.array(X[-test_nb:]).reshape([-1, slice_size, slice_size, 1])
    test_y = np.array(y[-test_nb:])
    print("    Dataset created! ✅")

    # Save
    save_dataset(train_X, train_y, validation_X, validation_y, test_X, test_y, nb_per_genre, genres, slice_size)

    return train_X, train_y, validation_X, validation_y, test_X, test_y


# Noi

def create_data_sets(nb_per_genre, validation_ratio, test_ratio):
    """Creates and save dataset from slices"""
    all_filenames_and_genres = []
    for genre in selected_genres:
        print("-> Adding {}...".format(genre))
        # Get slices in genre subfolder
        filenames_and_genres = os.listdir(slices_path + genre)
        filenames_and_genres = [(filename, genre) for filename in filenames_and_genres if filename.endswith('.png')]
        if nb_per_genre != -1:
            filenames_and_genres = filenames_and_genres[:nb_per_genre]
        all_filenames_and_genres += filenames_and_genres
    shuffle(all_filenames_and_genres)
    validation_nb = int(len(all_filenames_and_genres) * validation_ratio)
    test_nb = int(len(all_filenames_and_genres) * test_ratio)
    train_nb = len(all_filenames_and_genres) - (validation_nb + test_nb)
    file_type_to_paths = {'train': all_filenames_and_genres[:train_nb],
                          'validation': all_filenames_and_genres[train_nb:train_nb + validation_nb],
                          'test': all_filenames_and_genres[-test_nb:]}
    return file_type_to_paths


def compute_dnn_arrays(file_type_to_paths, mode):
    if mode == 'train':
        train_x_gen = file_type_to_paths['train']
        train_x_gen = split(train_x_gen, training_splits)
        validation_x_gen = file_type_to_paths['validation']
        validation_x_gen = split(validation_x_gen, training_splits)
        for _ in range(training_splits):
            train_x, train_y = get_x_and_y_from_pairs(next(train_x_gen))
            validation_x, validation_y = get_x_and_y_from_pairs(next(validation_x_gen))
            yield train_x, train_y, validation_x, validation_y
    elif mode == 'test':
        test_x_gen = file_type_to_paths['test']
        test_x_gen = split(test_x_gen, test_splits)
        for _ in range(test_splits):
            test_x, test_y = get_x_and_y_from_pairs(next(test_x_gen))
            yield test_x, test_y


def compute_dnn_generators(file_type_to_paths, mode, x_or_y):
    pairs = file_type_to_paths[mode]
    if x_or_y == 'x':
        return CustomGenerator.CustomGenerator(pairs, batch_size, input_map_func=get_image_data_v2)
    elif x_or_y == 'y':
        genres = [pair[1] for pair in pairs]
        return CustomGenerator.CustomGenerator(genres, batch_size, input_map_func=create_label_from_genre)
    else:
        raise f'x_or_y parameter should have "x" or "y" value...'


def get_x_and_y_from_pairs(pairs):
    y = np.array([create_label_from_genre(pair[1]) for pair in pairs])
    x = [get_image_data(slices_path + genre + "/" + filename, slice_size) for filename, genre in pairs]
    x = np.array(x).reshape([-1, slice_size, slice_size, 1])
    return x, y


def get_image_data_v2(pair_of_filename_and_genre):
    filename = pair_of_filename_and_genre[0]
    genre = pair_of_filename_and_genre[1]
    return get_image_data(slices_path + genre + '/' + filename, slice_size)


def create_label_from_genre(genre):
    return [1. if g == genre else 0. for g in selected_genres]


def save_data_sets(file_type_to_paths):
    pickle.dump(file_type_to_paths, open(f"{dataset_path}file_type_to_paths.p", "wb"))


def load_data_sets():
    return pickle.load(open(f"{dataset_path}file_type_to_paths.p", "rb"))


def split(iterable, n):
    k, m = divmod(len(iterable), n)
    return (iterable[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))


def execute_processes(target_funcs, args_iterable):
    processes = []
    for target, args in zip(target_funcs, args_iterable):
        process = multiprocessing.Process(target=target, args=args)
        processes.append(process)
        process.start()
    for p in processes:
        p.join()


def time_it(func):
    def wrapper(*args, **kwargs):
        begin = time.time()
        output = func(*args, **kwargs)
        end = time.time()
        print(f'Total time taken by {func.__name__}: {end - begin}')
        return output

    return wrapper

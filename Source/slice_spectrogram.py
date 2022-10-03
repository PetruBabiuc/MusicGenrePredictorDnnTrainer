# Import Pillow:
import os.path

from PIL import Image

from config import spectrograms_path, slices_path, spectrogram_splits_maker_process_number
from Source.tools import split, execute_processes


def create_slices_from_spectrograms(desired_size):
    """Batch slicing"""
    spectrograms = [f for f in os.listdir(spectrograms_path) if f.endswith(".png")]
    spectrograms = split(spectrograms, spectrogram_splits_maker_process_number)
    spectrograms = ((partition, desired_size) for partition in spectrograms)
    targets = (slice_maker_process_target for _ in range(spectrogram_splits_maker_process_number))
    execute_processes(targets, spectrograms)



def slice_maker_process_target(partition, desired_size):
    for spectrogram in partition:
        slice_spectrogram(spectrogram, desired_size)


# TODO Improvement - Make sure we don't miss the end of the song
def slice_spectrogram(filename, desired_size):
    """Creates slices from one spectrogram"""
    genre = filename.split("_")[0]  # Ex. Dubstep_19.png

    # Create path if doesn't exist
    slice_path = slices_path + "{}/".format(genre)
    os.makedirs(os.path.dirname(slice_path), exist_ok=True)
    create_slice(filename, spectrograms_path, desired_size, slice_path)


def create_slice(filename, file_dir, desired_size, destination_dir):
    # Load the full spectrogram
    img = Image.open(file_dir + filename)

    # Compute approximate number of 128x128 samples
    width, height = img.size
    nb_samples = int(width / desired_size)
    width - desired_size

    # For each sample
    for i in range(nb_samples):
        # print("Creating slice: ", i + 1, "/", nb_samples, "for", filename)
        # Extract and save 128x128 sample
        start_pixel = i * desired_size
        img_tmp = img.crop((start_pixel, 1, start_pixel + desired_size, desired_size + 1))
        img_tmp.save("{}/{}_{}.png".format(destination_dir, filename[:-4], i))

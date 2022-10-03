# -*- coding: utf-8 -*-
import os
from pathlib import Path
from subprocess import Popen, PIPE, STDOUT
import multiprocessing
# Remove logs
import eyed3

eyed3.log.setLevel("ERROR")

from Source.slice_spectrogram import create_slices_from_spectrograms
from Source.tools import is_mono, split, time_it, execute_processes, get_bitrate
from config import raw_data_path, spectrograms_path, pixel_per_second, slice_size, spectrogram_maker_processes_number, \
    genre_getter


def create_spectrogram(filename, new_filename, new_file_path, force_192k_bit_rate=False):
    """Creates spectrogram from mp3 files"""
    current_path = os.getcwd()

    filepath = filename
    if not filepath.startswith(raw_data_path):
        filepath = os.path.join(raw_data_path, filename)
    # Create temporary mono track if needed
    if is_mono(filepath):
        command = "cp '{}' '/tmp/{}.mp3'".format(filepath, new_filename)
    else:
        if force_192k_bit_rate and get_bitrate(filepath) != 192:
            command = "sox '{}' -C 192.02 '/tmp/{}.mp3' remix - ".format(filepath, new_filename)
        else:
            command = "sox -t mp3 '{}' -t mp3 '/tmp/{}.mp3' remix -".format(filepath, new_filename)
    p = Popen(command, shell=True, stdin=PIPE, stdout=PIPE, stderr=STDOUT, close_fds=True, cwd=current_path)
    output, errors = p.communicate()
    if errors:
        print(errors)

    # Create spectrogram
    spectrogram_filepath = os.path.join(new_file_path, new_filename)
    command = "sox '/tmp/{}.mp3' -n spectrogram -Y 200 -X {} -m -r -o '{}.png'".format(new_filename, pixel_per_second,
                                                                                       spectrogram_filepath)
    p = Popen(command, shell=True, stdin=PIPE, stdout=PIPE, stderr=STDOUT, close_fds=True, cwd=current_path)
    output, errors = p.communicate()
    if errors:
        print(errors)

    # Remove tmp mono track
    try:
        os.remove("/tmp/{}.mp3".format(new_filename))
    except FileNotFoundError as _:
        pass


def create_spectrograms_from_audio():
    """Creates .png whole spectrograms from mp3 files"""
    genres_id = multiprocessing.Manager().dict()

    # files = [file for file in os.listdir(raw_data_path) if file.endswith(".mp3")]
    files = [str(it).removeprefix(raw_data_path) for it in Path(raw_data_path).rglob("*.mp3")]

    # Create path if doesn't exist
    os.makedirs(os.path.dirname(spectrograms_path), exist_ok=True)
    args = split(list(enumerate(files)), spectrogram_maker_processes_number)
    args = ((partition, genres_id, len(files), genre_getter) for partition in args)
    targets = (spectrogram_maker_thread_target for _ in range(spectrogram_maker_processes_number))
    execute_processes(targets, args)


def spectrogram_maker_thread_target(partition, genres_id, total_len, genre_getter):
    for index, filename in partition:
        print("[Spectrogram maker process {}] Creating spectrogram for file {}/{}...".format(os.getpid(), index + 1,
                                                                                             total_len))
        file_genre = genre_getter.get_genre(raw_data_path + filename)
        if not file_genre:
            continue
        file_id = genres_id.get(file_genre, 0) + 1
        genres_id[file_genre] = file_id  # Increment counter
        new_filename = file_genre + "_" + str(file_id)
        create_spectrogram(filename, new_filename, spectrograms_path)


@time_it
def create_slices_from_audio():
    """Whole pipeline .mp3 -> .png slices"""
    print("Creating spectrograms...")
    create_spectrograms_from_audio()
    print("Spectrograms created!")

    print("Creating slices...")
    create_slices_from_spectrograms(slice_size)
    print("Slices created!")

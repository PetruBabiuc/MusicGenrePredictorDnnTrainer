# Define paths for files
from Source import GenreGetters

spectrograms_path = "Data/Spectrograms/"
slices_path = "Data/Slices/"
dataset_path = "Data/Dataset/"
raw_data_path = "Data/Raw/"

# Spectrogram resolution
pixel_per_second = 50

# Slice parameters
slice_size = 128

# Dataset parameters
files_per_genre = -1
validation_ratio = 0.3
test_ratio = 0.1

# Model parameters
batch_size = 110
learning_rate = 0.001
nb_epochs = 30

# La cele cu International
# selected_genres = ('Electronic', 'Folk', 'Hip-Hop', 'Instrumental', 'International', 'Pop', 'Rock')

# Electronic: 170000
# Folk: 41000
# Hip-Hop: 39000
# Instrumental: 43000
# Pop: 48000
# Rock: 142000

# La cele fara International, Experimental
selected_genres = ('Electronic', 'Folk', 'Hip-Hop', 'Instrumental', 'Pop', 'Rock')
spectrogram_maker_processes_number = 7
spectrogram_splits_maker_process_number = 7

# The entire [(slice_1_data, genre), (slice_2_data, genre), ..., (slice_2_data, genre)]
# consumes a lot of ram. If the RAM gets full, the process will most likely receive SIGKILL.
# The entire dataset is split: all the filenames are loaded into memory but the DNN is trained with
# partitions of all the dataset. The "training_splits" represents the number of how many partitions
# the dataset is split into.
training_splits = 12
training_splits_to_skip = 0

test_splits = 10
no_cuda = True

dnn_path = 'DNN saves/61% jamendo relu/musicDNN.tflearn6213'

genre_getter = GenreGetters.FmaGenreGetter('Data/Metadata/genres.csv', selected_genres)
# genre_getter = GenreGetters.JamendoGenreGetter('Data/Metadata/autotagging_genre.tsv')

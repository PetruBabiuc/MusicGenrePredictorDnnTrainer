import os
import glob
import config
from Source import song_to_data, slice_spectrogram, CustomGenerator, tools
from Source.model import create_model


def get_model():
    model = create_model(len(config.selected_genres), config.slice_size)
    model.load(config.dnn_path)
    return model


def get_all_songs_and_genres():
    # path = 'Data/Raw/fma_small'
    path = 'Data/Raw/petru'
    # path = 'Data/Raw/free_mp3'
    pairs = glob.glob(f'{path}/**/*.mp3', recursive=True)
    pairs = map(lambda file: (file, config.genre_getter.get_genre(file)), pairs)
    pairs = list(filter(lambda pair: pair[1], pairs))
    return pairs


def print_genre_stats_of_song(song, real_genre, model):
    song_to_data.create_spectrogram(song, 'spectrogram', 'Auxiliary spectrogram directory')
    slice_spectrogram.create_slice('spectrogram.png', 'Auxiliary spectrogram directory/', config.slice_size,
                                   'Auxiliary spectrogram directory')
    slices = glob.glob('Auxiliary spectrogram directory/spectrogram_*.png')
    x = CustomGenerator.CustomGenerator(slices,
                                        input_map_func=lambda it: tools.get_image_data(it, config.slice_size))
    results = model.predict(x)

    for slice_path in slices:
        os.remove(slice_path)
    os.remove('Auxiliary spectrogram directory/spectrogram.png')

    slices_with_high_probability_of_genre_prediction = 0
    genre_to_count = {genre: 0 for genre in config.selected_genres}
    for spectrogram_slice, result in zip(slices, results):
        ratio = result.max()
        # if ratio > 0.5:
        if True:
            slices_with_high_probability_of_genre_prediction += 1
            genre = config.selected_genres[result.argmax()]
            genre_to_count[genre] += 1
        # print(f'Slice "{spectrogram_slice}" => Genre: {genre}, probability: {ratio}')
    predicted_genre = max(genre_to_count, key=genre_to_count.get)
    if slices_with_high_probability_of_genre_prediction:
        ratio = max(genre_to_count.values()) / slices_with_high_probability_of_genre_prediction
        print(f'Song: {song}\t\tReal genre: {real_genre}\t\tPredicted genre: {predicted_genre}\t\tRatio: {ratio}')
        # print('\n\n\n')
        return real_genre == predicted_genre
    else:
        print(f'Song: {song}\t\tNo slice with high probability of genre prediction..')
        return False


@tools.time_it
def print_genre_stats_of_all_songs(model):
    if config.no_cuda:
        print('Disabling CUDA support for TensorFlow...')
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    songs = get_all_songs_and_genres()
    # filepath = 'Data/Raw/dataset/00/1100.mp3'
    # filepath = 'Data/Raw/free_mp3/Rest_Easy_-_Laurence_DaNova.mp3'
    # songs = [(filepath, config.genre_getter.get_genre(filepath))]
    correctly_predicted_genres = 0
    for song, real_genre in songs:
        if print_genre_stats_of_song(song, real_genre, model):
            correctly_predicted_genres += 1
    print(f'Song genre prediction accuracy: {correctly_predicted_genres / len(songs)}')


if __name__ == '__main__':
    dnn = get_model()
    print_genre_stats_of_all_songs(dnn)

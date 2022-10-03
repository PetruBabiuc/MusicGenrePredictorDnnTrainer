from abc import ABCMeta, abstractmethod
from dataclasses import dataclass

import eyed3
from functional import seq


class AbstractGenreGetter(metaclass=ABCMeta):
    @abstractmethod
    def get_genre(self, filename):
        pass


@dataclass
class GenreMetadata:
    genre_id: int
    genre_name: str
    top_level: int


class FmaGenreGetter(AbstractGenreGetter):
    def __init__(self, genres_csv_path, selected_genres):
        self.__selected_genres = selected_genres
        with open(genres_csv_path) as csvfile:
            lines = csvfile.read().splitlines()[1:]
        base = seq(lines) \
            .map(lambda it: it.split(','))
        self.__genre_name_to_metadata = base \
            .map(lambda it: (it[3], GenreMetadata(int(it[0]), it[3], int(it[4])))) \
            .to_dict()
        self.__top_level_genre_id_to_metadata = base \
            .filter(lambda it: it[0] == it[4]) \
            .map(lambda it: (int(it[0]), GenreMetadata(int(it[0]), it[3], int(it[4])))) \
            .to_dict()

    def get_genre(self, filename):
        audiofile = eyed3.load(filename)
        genre = audiofile.tag.genre
        # No genre
        if not genre:
            print(f'File {filename} had no genre...')
            return None
        genre = genre.name
        genre_metadata = self.__genre_name_to_metadata.get(genre)
        if not genre_metadata:
            print(f'Invalid genre "{genre}" of file {filename}...')
            return None
        top_level_genre = self.__top_level_genre_id_to_metadata[genre_metadata.top_level]
        top_level_genre = top_level_genre.genre_name
        if top_level_genre not in self.__selected_genres:
            print(f'The top level genre "{top_level_genre}" is not in the selected genres.')
            return None
        return top_level_genre


class JamendoGenreGetter(AbstractGenreGetter):
    def __init__(self, genres_tsv_path):
        genre_metadata_to_genre_name = {
            'genre---rock': 'Rock',
            'genre---pop': 'Pop',
            'genre---hiphop': 'Hip-Hop',
            'genre---electronic': 'Electronic',
            'genre---classical': 'Instrumental',
            'genre---folk': 'Folk',
        }
        with open(genres_tsv_path) as f:
            lines = f.read().splitlines()[1:]
        self.__filename_to_genre = seq(lines) \
            .map(lambda line: line.split('\t')) \
            .map(lambda line_components: (line_components[3], line_components[5:])) \
            .map(lambda pair: (pair[0], seq(pair[1]).find(lambda it: it in genre_metadata_to_genre_name))) \
            .filter(lambda pair: pair[1] is not None).map(lambda pair: (pair[0], genre_metadata_to_genre_name[pair[1]])) \
            .to_dict()
        pass

    def get_genre(self, filename):
        filename = filename.split('/')[-2:]
        filename = '/'.join(filename)
        return self.__filename_to_genre[filename]

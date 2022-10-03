from dataclasses import dataclass
from pathlib import Path

import numpy
import tensorflow
from tensorflow import initializers

from Source import CustomGenerator, tools
from config import raw_data_path
from Source.tools import split


def test1():
    i = initializers.glorot_uniform()
    if type(i) in [tensorflow.Tensor, numpy.ndarray, list]:
        print('OK')
    pass


def test2():
    result = list(str(Path(raw_data_path).rglob("*.mp3")))
    pass


@dataclass
class GenreMetadata:
    genre_id: int
    genre_name: str
    top_level: int

def test4():
    l = list(range(5732))
    rez = split(l, 7)
    for t in rez:
        print(t)
    pass

def test5():
    files = tools.create_data_sets(1500, 0.3, 0.1)
    tools.compute_dnn_arrays(files, 'train')

def test6():
    g = (i for i in range(6))
    e1 = next(g)
    e2 = next(g)
    pass

def test7():
    files = tools.create_data_sets(1500, 0.3, 0.1)

    for train_x, train_y, validation_x, validation_y in tools.compute_dnn_arrays(files, 'train'):
        pass
    pass

def test8():
    generator = CustomGenerator.CustomGenerator(list(range(100)), 10, input_map_func=lambda it: it * 2)
    x = generator[(0, 2, 4)]
    l = numpy.array([it*it for it in range(10)])
    x = l[[0, 4, 1, 9]]
    pass

def test9():
    tensorflow_keras_xavier = tensorflow.keras.initializers.glorot_uniform()
    tensorflow_xavier = initializers.GlorotUniform()
    tensorflow_keras_xavier_v2 = tensorflow.compat.v1.keras.initializers.glorot_uniform()
    pass

if __name__ == '__main__':
    test9()

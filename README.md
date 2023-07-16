# MusicGenrePredictorDnnTrainer
This is the repository for the CNN (convolutional neural network) training/validation/testing used in my bachelor's degree project. 
The other parts cand be found in the following repositories:
* [Back-end repository](https://github.com/PetruBabiuc/MusicGenreComputationMicroservices)
* [Front-end repository](https://github.com/PetruBabiuc/BachelorsDegreeFrontEnd)

This repository (part of the project) was not created from scratch, it was created from Julien Despois [repository](https://github.com/despoisj/DeepAudioClassification) and [Medium article](https://medium.com/@juliendespois/finding-the-genre-of-a-song-with-deep-learning-da8f59a61194#.yhemoyql0). The CNN obtained after the last training experiment can be downloaded from [here](https://drive.google.com/file/d/1ZVZNSOMlAHZFPhuFQ-aXEDd2vy7sLHVl/view?usp=drive_link).



The process of computing the song genre using the CNN has the following steps:
1. Compute the song's spectrogram (using [SoX](https://www.soundexchange.com))
2. Split the spectrogram in images of size 128x128 (using [Pillow](https://pillow.readthedocs.io/en/stable) library)
3. For each spectrogram slice, compute each genre's probability using the convolutional neural network (created using [TFLearn](http://tflearn.org) framework)
4. Find song's genre by picking the genre with the most spectrogram fragments

The convolutional neural network has the following structure (image from the [Medium article](https://medium.com/@juliendespois/finding-the-genre-of-a-song-with-deep-learning-da8f59a61194#.yhemoyql0)):
![CNN structure](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*FQyTMv3f7m2WHFWz5gCy9g.png)

## Training, validation, testing
For training, validation and testing I used two datasets:
* [FMA](https://github.com/mdeff/fma), the following sizes:
  * Small (7.2 GB)
  * Large (93 GB)
* [MTG-Jamendo](https://github.com/MTG/mtg-jamendo-dataset) (508 GB)

The datasets were split as follows:

| Split | Percentage |
| :---- | ----: |
| Training | 60% |
| Validation | 30% |
| Testing | 10% |

With the purpose of obtaining a performant CNN, the following table shows the training attempts I have made:
|    | Pixel/second | Files/Genre | Epochs | Initial weights | Activations | Accuracy (%) | Training time | RAM usage | Mentions |
| -  | ------------ | ----------- | ------ | --------------- | ----------- | ------------ | ------------- | --------- | -------- |
| 1  | 50           | 1000        | 30     | Glorot Uniform  | ELU         | 33           | ?             | ?         |          |
| 2  | 50           | 1000        | 30     | He Uniform      | ELU         | 41           | ?             | ?         |          |
| 3  | 50           | 1500        | 30     | He Uniform      | ELU         | 42           | ?             | ?         |          |
| 4  | 25           | 1500        | 30     | He Uniform      | ELU         | 45           | ?             | ?         |          |
| 5  | 25           | 1500        | 60     | He Uniform      | ELU         | 45           | ?             | ?         |          |
| 6  | 25           | 3150        | 30     | He Uniform      | ELU         | 44           | ?             | ?         |          |
| 7  | 50           | 6300        | 30     | He Uniform      | ELU         | 51           | ?             | 68%       |          |
| 8  | 100          | 14000       | 30     | He Uniform      | ELU         | 50           | ?             | 68%       |          |
| 9  | 50           | 6300        | 30     | He Uniform      | ELU         | 55           | ?             | 68%       | No "Experimental" genre |
| 10 | 50           | 6300        | 30     | Glorot Uniform  | ELU         | 56           | ?             | 68%       | No "Experimental" genre |
| 11 | 50           | 6300        | 30     | Lecun Uniform   | ELU         | 53           | ?             | 68%       | No "Experimental" genre  |
| 12 | 50           | 6300        | 30     | Glorot Uniform  | ReLU        | 55           | ?             | 68%       | No "Experimental" genre |
| 13 | 50           | 6300        | 30     | Glorot Uniform  | ELU         | 61           | ?             | 58%       | No "Experimental" genre, no partitions |
| 14 | 50           | 6300        | 30     | Glorot Uniform  | ReLU        | 64           | ?             | 58%       |No "Experimental" genre, no partitions |
| 15 | 50           | 6300        | 30     | Glorot Uniform  | ReLU        | 68           | ?             | 58%       |No "Experimental" genre, no partitions |
| 16 | 50           | 6300        | 30     | Glorot Uniform  | ReLU        | 70           | ?             | 58%       | No "Experimental" genre, no partitions, best checkpoint weights |
| 17 | 50           | 80500       | 30     | Glorot Uniform  | ReLU        | 61           | 1h            | 58%       | No "Experimental" and "International" genres |
| 18 | 50           | 80500       | 30     | Glorot Uniform  | ELU         | 60           | 10h           | 58%       | No "Experimental" and "International" genres |
| 19 | 50           | 80500       | 30     | Glorot Uniform  | ELU         | 59           | 10h           | 58%       | No "Experimental" and "International" genres |
| 20 | 50           | 57193       | 30     | Glorot Uniform  | ReLU        | 61           | 7.2h          | 58%       | MTG-Jamendo Dataset, same 6 top level genres |

## Original project additions/new features
When creating the spectrogram slices, both steps (the spectrogram creation and slicing) are parallelized. Each steps' inputs (songs and spectrogram paths) are partitioned and each partition is handled in a distinct process using the [Python Multiprocessing library](https://docs.python.org/3/library/multiprocessing.html). The parallelization sped up the spectrogram slices computation as following:
| Processes | Time(s) | Time(m) | CPU Usage (%) |
| --------- | ------- | ------- | ------------- |
| 1         | 2560    | 42.66   | 13            |
| 5         | 586     | 9.76    | 65            |
| 7         | 494     | 8.23    | 91            |

The songs genres are obtained from:
* the ID3 metadata (FMA dataset) using the [FmaGenreGetter class](https://github.com/PetruBabiuc/MusicGenrePredictorDnnTrainer/blob/main/Source/GenreGetters.py)
* the genre metadata files found [here](https://github.com/PetruBabiuc/MusicGenrePredictorDnnTrainer/tree/main/Data/Metadata) (MTG-Jamendo dataset) using the [JamendoGenreGetter class](https://github.com/PetruBabiuc/MusicGenrePredictorDnnTrainer/blob/main/Source/GenreGetters.py)

I have also created an [Early Stopping Callback](https://github.com/PetruBabiuc/MusicGenrePredictorDnnTrainer/blob/main/Source/EarlyStopCallback.py) that stops the training process when the validation accuracy and loss have satisfying values. It has the purpose of avoiding overfitting the model.

## Problems
I encountered the problem that an entire dataset can't be loaded entirely in the RAM memory. I found two solutions:
1. Create partitions for the dataset and train/validate/test the CNN for each partition
* It proved to be somewhat unstable (sometimes the accuracy in the training phases from a partition to another drops like reseting the weights)
* Uses more RAM
* Faster because the spectrogram slices are loaded only once
2. Create a [custom data structure](https://github.com/PetruBabiuc/MusicGenrePredictorDnnTrainer/blob/main/Source/CustomGenerator.py)
* Is constructed with:
    * the elements in a _low memory usage form_ (in our case spectrogram slices' paths)
    * a function that transforms the _low memory usage form_ elements in _real elements_ (in our case loads the spectrogram slices from the paths)
* Supports indexing
* Loads the _real elements_ on demand (in our case when the TFLearn CNN training/validation/testing algorithms requests them)
* Uses less RAM
* Slower because the training and validation slices are loaded every training epoch

import config
from config import files_per_genre, slice_size, validation_ratio, test_ratio, batch_size
from main import genres, model
from tools import get_dataset, time_it, load_data_sets, compute_dnn_arrays, compute_dnn_generators


def test_1():
    # Create or load new dataset
    test_X, test_y = get_dataset(files_per_genre, genres, slice_size, validation_ratio, test_ratio, mode="test")

    testAccuracy = model.evaluate(test_X, test_y)[0]
    print("[+] Test accuracy: {} ".format(testAccuracy))


@time_it
def test_2():
    print('Loading datasets...')
    file_type_to_path = load_data_sets()
    total_acc = 0
    split = 1
    for test_x, test_y in compute_dnn_arrays(file_type_to_path, 'test'):
        acc = model.evaluate(test_x, test_y)[0]
        print(f"Split {split}'s accuracy: {acc}")
        split += 1
        total_acc += acc
    total_acc /= config.test_splits
    print("[+] Test accuracy: {} ".format(total_acc))


@time_it
def test_3():
    print('Loading datasets...')
    file_type_to_path = load_data_sets()
    test_x = compute_dnn_generators(file_type_to_path, 'test', 'x')
    test_y = compute_dnn_generators(file_type_to_path, 'test', 'y')
    accuracy = model.evaluate(test_x, test_y, batch_size)[0]
    print("[+] Test accuracy: {} ".format(accuracy))

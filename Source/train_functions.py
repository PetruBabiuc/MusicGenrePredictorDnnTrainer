from Source import EarlyStopCallback
import config
from config import files_per_genre, slice_size, validation_ratio, test_ratio, nb_epochs, batch_size
from main import genres, model, run_id
from Source.model import save_model, load_model
from tools import get_dataset, time_it, create_data_sets, save_data_sets, compute_dnn_arrays, compute_dnn_generators


def train_1():
    # Create or load new dataset
    train_X, train_y, validation_X, validation_y = get_dataset(files_per_genre, genres, slice_size,
                                                               validation_ratio, test_ratio, mode="train")

    model.fit(train_X, train_y, n_epoch=nb_epochs, batch_size=batch_size, shuffle=True,
              validation_set=(validation_X, validation_y), snapshot_step=100, show_metric=True, run_id=run_id)
    save_model(model)


@time_it
def train_2():
    print('Creating datasets..')
    file_type_to_path = create_data_sets(files_per_genre, validation_ratio, test_ratio)
    print('Saving datasets...')
    save_data_sets(file_type_to_path)

    if config.training_splits_to_skip > 0:
        load_model(model)

    step = 1
    skipped = 0
    for train_x, train_y, validation_x, validation_y in compute_dnn_arrays(file_type_to_path, mode='train'):
        if skipped < config.training_splits_to_skip:
            skipped += 1
            continue

        print(f'-------------- Training step {step}/{config.training_splits} --------------')

        model.fit(train_x, train_y, n_epoch=nb_epochs, batch_size=batch_size, shuffle=True,
                  validation_set=(validation_x, validation_y), snapshot_step=500, show_metric=True, run_id=run_id)
        save_model(model)
        step += 1


@time_it
def train_3():
    print('Creating datasets..')
    file_type_to_path = create_data_sets(files_per_genre, validation_ratio, test_ratio)
    print('Saving datasets...')
    save_data_sets(file_type_to_path)
    train_x = compute_dnn_generators(file_type_to_path, 'train', 'x')
    train_y = compute_dnn_generators(file_type_to_path, 'train', 'y')
    validation_x = compute_dnn_generators(file_type_to_path, 'validation', 'x')
    validation_y = compute_dnn_generators(file_type_to_path, 'validation', 'y')
    early_stopping_callback = EarlyStopCallback.EarlyStoppingCallback(0.9, 0.1, '../logs.txt')
    model.fit(train_x, train_y, n_epoch=nb_epochs, shuffle=True, batch_size=batch_size,
              validation_set=(validation_x, validation_y),
              show_metric=True, run_id=run_id, callbacks=[early_stopping_callback])
    save_model(model)

import numpy as np


def split_datasets(target_dataset, test_fold=2, num_folds=4):
    assert 1 <= test_fold <= num_folds, f"test_fold must be between 1 and {num_folds}"

    datasets = target_dataset.copy()
    np.random.shuffle(datasets)
    fold_size = len(datasets) // num_folds
    folds = []
    for i in range(num_folds):
        if i == num_folds - 1:
            folds.append(datasets[i * fold_size:])
        else:
            folds.append(datasets[i * fold_size:(i + 1) * fold_size])
    test_datasets = folds[test_fold - 1]
    train_datasets = []
    for i in range(num_folds):
        if i != test_fold - 1:
            train_datasets.extend(folds[i])

    return train_datasets, test_datasets

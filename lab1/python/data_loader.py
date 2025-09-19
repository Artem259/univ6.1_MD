import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
from pathlib import Path


def load_data(
        data_dir: Path,
        val_fold: int,
        total_folds: int,
        class_names: list[str],
        batch_size: int,
        image_size: int
) -> tuple:
    def load_fold(fold_num):
        fold_dir = data_dir / f"cv{fold_num}"
        return image_dataset_from_directory(
            directory=fold_dir,
            labels='inferred',
            label_mode='int',
            class_names=class_names,
            color_mode="grayscale",
            batch_size=batch_size,
            image_size=(image_size, image_size),
            shuffle=True,
            seed=42,
        )

    val_data = load_fold(val_fold)

    train_folds_data = []
    for fold in range(1, total_folds + 1):
        if fold != val_fold:
            train_folds_data.append(load_fold(fold))

    train_data = train_folds_data[0]
    for ds in train_folds_data[1:]:
        train_data = train_data.concatenate(ds)

    train_data = train_data.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    val_data = val_data.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return train_data, val_data

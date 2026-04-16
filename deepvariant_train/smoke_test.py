from __future__ import annotations

import argparse

import tensorflow as tf

from deepvariant_train.data import (
    DEFAULT_TRAIN_PATTERN,
    build_dataset,
    infer_input_shape,
    list_tfrecords,
)
from deepvariant_train.models import MODEL_BUILDERS, build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a lightweight project smoke test.")
    parser.add_argument("--data-dir", default="tfrecords")
    parser.add_argument("--train-pattern", default=DEFAULT_TRAIN_PATTERN)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-classes", type=int, default=3)
    parser.add_argument(
        "--models",
        nargs="*",
        default=list(MODEL_BUILDERS),
        choices=list(MODEL_BUILDERS),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_shape = infer_input_shape(args.data_dir, args.train_pattern)
    train_files = list_tfrecords(args.data_dir, args.train_pattern)
    dataset = build_dataset(
        train_files,
        input_shape=input_shape,
        batch_size=args.batch_size,
        training=True,
        shuffle=True,
        shuffle_buffer=64,
    )
    images, labels = next(iter(dataset.take(1)))
    print(f"Batch image shape: {tuple(images.shape)}")
    print(f"Batch label shape: {tuple(labels.shape)}")
    print(f"Batch labels: {labels.numpy().tolist()}")

    for model_name in args.models:
        tf.keras.backend.clear_session()
        model = build_model(
            model_name,
            input_shape=input_shape,
            num_classes=args.num_classes,
        )
        logits = model(images[:1], training=False)
        print(
            f"{model_name}: output_shape={tuple(logits.shape)}, "
            f"params={model.count_params():,}"
        )


if __name__ == "__main__":
    main()


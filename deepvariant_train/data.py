from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import tensorflow as tf

DEFAULT_INPUT_SHAPE = (100, 221, 6)
DEFAULT_TRAIN_PATTERN = "*train*.tfrecord-*-of-*.gz"
DEFAULT_VAL_PATTERN = "*val*.tfrecord-*-of-*.gz"
DEFAULT_TEST_PATTERN = "*test*.tfrecord-*-of-*.gz"


def parse_input_shape(value: str | Sequence[int] | None) -> tuple[int, int, int]:
    if value is None:
        return DEFAULT_INPUT_SHAPE
    if isinstance(value, str):
        parts = [int(part.strip()) for part in value.split(",") if part.strip()]
    else:
        parts = [int(part) for part in value]
    if len(parts) != 3:
        raise ValueError("input shape must have 3 integers: height,width,channels")
    return tuple(parts)  # type: ignore[return-value]


def list_tfrecords(data_dir: str | Path, pattern: str) -> list[str]:
    root = Path(data_dir)
    files = sorted(
        str(path)
        for path in root.glob(pattern)
        if path.is_file() and not path.name.endswith(".json")
    )
    if not files:
        raise FileNotFoundError(f"No TFRecord files found in {root} with pattern {pattern!r}")
    return files


def load_example_info(data_dir: str | Path, pattern: str = DEFAULT_TRAIN_PATTERN) -> dict:
    root = Path(data_dir)
    tfrecord_files = [Path(path) for path in list_tfrecords(root, pattern)]
    candidate_paths: list[Path] = []
    for tfrecord_path in tfrecord_files:
        candidate_paths.append(Path(str(tfrecord_path) + ".example_info.json"))
    candidate_paths.extend(sorted(root.glob("*.example_info.json")))

    for path in candidate_paths:
        if path.exists():
            with path.open("r", encoding="utf-8") as handle:
                return json.load(handle)

    return {"shape": list(DEFAULT_INPUT_SHAPE)}


def infer_input_shape(data_dir: str | Path, pattern: str = DEFAULT_TRAIN_PATTERN) -> tuple[int, int, int]:
    info = load_example_info(data_dir, pattern)
    return parse_input_shape(info.get("shape"))


def make_example_parser(
    input_shape: Sequence[int],
    *,
    image_key: str = "image/encoded",
    label_key: str = "label",
    shape_key: str = "image/shape",
    normalize: bool = True,
):
    input_shape = tuple(int(dim) for dim in input_shape)
    expected_size = int(np.prod(input_shape))

    feature_spec = {
        image_key: tf.io.FixedLenFeature([], tf.string),
        label_key: tf.io.FixedLenFeature([], tf.int64),
    }
    if shape_key:
        feature_spec[shape_key] = tf.io.FixedLenFeature(
            [3], tf.int64, default_value=list(input_shape)
        )

    def parse(serialized: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        features = tf.io.parse_single_example(serialized, feature_spec)
        image = tf.io.decode_raw(features[image_key], out_type=tf.uint8)
        image = tf.ensure_shape(image, [expected_size])
        image = tf.reshape(image, input_shape)
        image = tf.cast(image, tf.float32)
        if normalize:
            image = image / 255.0

        label = tf.cast(features[label_key], tf.int32)
        return image, label

    return parse


def build_dataset(
    files: Iterable[str | Path],
    *,
    input_shape: Sequence[int],
    batch_size: int,
    training: bool,
    shuffle: bool = True,
    shuffle_buffer: int = 8192,
    seed: int = 42,
    compression_type: str = "GZIP",
    image_key: str = "image/encoded",
    label_key: str = "label",
    shape_key: str = "image/shape",
    normalize: bool = True,
    drop_remainder: bool = False,
    parallel_reads: int = 8,
) -> tf.data.Dataset:
    file_list = [str(Path(path)) for path in files]
    if not file_list:
        raise ValueError("files must not be empty")

    file_dataset = tf.data.Dataset.from_tensor_slices(file_list)
    if training and shuffle:
        file_dataset = file_dataset.shuffle(
            buffer_size=len(file_list),
            seed=seed,
            reshuffle_each_iteration=True,
        )

    cycle_length = max(1, min(int(parallel_reads), len(file_list)))
    dataset = file_dataset.interleave(
        lambda filename: tf.data.TFRecordDataset(
            filename,
            compression_type=compression_type,
        ),
        cycle_length=cycle_length,
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=not (training and shuffle),
    )

    if training and shuffle:
        dataset = dataset.shuffle(
            buffer_size=shuffle_buffer,
            seed=seed,
            reshuffle_each_iteration=True,
        )

    parser = make_example_parser(
        input_shape,
        image_key=image_key,
        label_key=label_key,
        shape_key=shape_key,
        normalize=normalize,
    )
    dataset = dataset.map(
        parser,
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=not (training and shuffle),
    )
    dataset = dataset.batch(batch_size, drop_remainder=drop_remainder)
    return dataset.prefetch(tf.data.AUTOTUNE)


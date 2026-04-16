from __future__ import annotations

import argparse
from pathlib import Path

import tensorflow as tf

from deepvariant_train.data import DEFAULT_TRAIN_PATTERN, list_tfrecords


def describe_feature(feature: tf.train.Feature) -> str:
    kind = feature.WhichOneof("kind")
    if kind == "bytes_list":
        values = feature.bytes_list.value
        preview = f", first_len={len(values[0])}" if values else ""
        return f"bytes_list(count={len(values)}{preview})"
    if kind == "float_list":
        values = feature.float_list.value
        return f"float_list(count={len(values)}, first={list(values[:5])})"
    if kind == "int64_list":
        values = feature.int64_list.value
        return f"int64_list(count={len(values)}, first={list(values[:10])})"
    return "empty"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect one DeepVariant TFRecord example.")
    parser.add_argument("--data-dir", default="tfrecords", help="Folder containing TFRecord shards.")
    parser.add_argument("--pattern", default=DEFAULT_TRAIN_PATTERN, help="Glob pattern to choose a shard.")
    parser.add_argument("--file", default=None, help="Specific TFRecord file to inspect.")
    parser.add_argument("--compression-type", default="GZIP", help="TFRecord compression type.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    file_path = Path(args.file) if args.file else Path(list_tfrecords(args.data_dir, args.pattern)[0])
    dataset = tf.data.TFRecordDataset(str(file_path), compression_type=args.compression_type)
    raw_example = next(iter(dataset.take(1))).numpy()
    example = tf.train.Example.FromString(raw_example)

    print(f"File: {file_path}")
    print("Features:")
    for key in sorted(example.features.feature):
        print(f"  {key}: {describe_feature(example.features.feature[key])}")


if __name__ == "__main__":
    main()


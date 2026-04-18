from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import tensorflow as tf

from deepvariant_train.data import (
    DEFAULT_TRAIN_PATTERN,
    DEFAULT_VAL_PATTERN,
    build_dataset,
    infer_input_shape,
    list_tfrecords,
    parse_input_shape,
)
from deepvariant_train.models import MODEL_BUILDERS, build_model, normalize_model_name


def configure_runtime(mixed_precision: bool) -> None:
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass

    if mixed_precision:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")

    if gpus:
        gpu_names = ", ".join(gpu.name for gpu in gpus)
        print(f"GPUs visible to TensorFlow: {gpu_names}")
    else:
        print("GPUs visible to TensorFlow: none")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train image classifiers on DeepVariant TFRecord shards."
    )
    parser.add_argument("--data-dir", default="tfrecords", help="Folder containing TFRecord shards.")
    parser.add_argument(
        "--model",
        default="all",
        help=(
            "Model to train: all, inceptionv3, convnextv2_tiny, efficientnetv2_s, "
            "vit_tiny. Multiple models can be comma-separated."
        ),
    )
    parser.add_argument("--output-dir", default="runs", help="Folder for checkpoints and logs.")
    parser.add_argument("--run-name", default=None, help="Run folder name. Defaults to timestamp.")
    parser.add_argument("--train-pattern", default=DEFAULT_TRAIN_PATTERN)
    parser.add_argument("--val-pattern", default=DEFAULT_VAL_PATTERN)
    parser.add_argument("--input-shape", default=None, help="Example: 100,221,6. Defaults to metadata.")
    parser.add_argument("--num-classes", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--steps-per-epoch", type=int, default=None)
    parser.add_argument("--validation-steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout-rate", type=float, default=0.2)
    parser.add_argument("--drop-path-rate", type=float, default=0.1)
    parser.add_argument("--vit-patch-size", type=int, default=16)
    parser.add_argument("--shuffle-buffer", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--compression-type", default="GZIP")
    parser.add_argument("--image-key", default="image/encoded")
    parser.add_argument("--label-key", default="label")
    parser.add_argument("--shape-key", default="image/shape")
    parser.add_argument("--parallel-reads", type=int, default=8)
    parser.add_argument("--no-normalize", action="store_true", help="Do not divide images by 255.")
    parser.add_argument("--no-train-shuffle", action="store_true", help="Disable train shuffle.")
    parser.add_argument("--drop-remainder", action="store_true")
    parser.add_argument("--mixed-precision", action="store_true")
    parser.add_argument("--patience", type=int, default=0, help="EarlyStopping patience. 0 disables it.")
    parser.add_argument("--verbose", type=int, default=2, choices=[0, 1, 2])
    return parser.parse_args()


def make_optimizer(learning_rate: float, weight_decay: float) -> tf.keras.optimizers.Optimizer:
    if weight_decay > 0:
        return tf.keras.optimizers.AdamW(
            learning_rate=learning_rate,
            weight_decay=weight_decay,
        )
    return tf.keras.optimizers.Adam(learning_rate=learning_rate)


def make_callbacks(run_dir: Path, monitor: str) -> list[tf.keras.callbacks.Callback]:
    return [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(run_dir / "best.keras"),
            monitor=monitor,
            mode="min",
            save_best_only=True,
        ),
        tf.keras.callbacks.CSVLogger(str(run_dir / "history.csv")),
        tf.keras.callbacks.TensorBoard(log_dir=str(run_dir / "tensorboard")),
    ]


def write_model_summary(model: tf.keras.Model, path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        model.summary(print_fn=lambda line: handle.write(line + "\n"))


def train_one_model(
    *,
    model_name: str,
    args: argparse.Namespace,
    input_shape: tuple[int, int, int],
    train_files: list[str],
    val_files: list[str],
    run_dir: Path,
) -> None:
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(args.seed)

    train_dataset = build_dataset(
        train_files,
        input_shape=input_shape,
        batch_size=args.batch_size,
        training=True,
        shuffle=not args.no_train_shuffle,
        shuffle_buffer=args.shuffle_buffer,
        seed=args.seed,
        compression_type=args.compression_type,
        image_key=args.image_key,
        label_key=args.label_key,
        shape_key=args.shape_key,
        normalize=not args.no_normalize,
        drop_remainder=args.drop_remainder,
        parallel_reads=args.parallel_reads,
    )
    if args.steps_per_epoch is not None:
        train_dataset = train_dataset.repeat()

    val_dataset = build_dataset(
        val_files,
        input_shape=input_shape,
        batch_size=args.batch_size,
        training=False,
        shuffle=False,
        seed=args.seed,
        compression_type=args.compression_type,
        image_key=args.image_key,
        label_key=args.label_key,
        shape_key=args.shape_key,
        normalize=not args.no_normalize,
        drop_remainder=False,
        parallel_reads=args.parallel_reads,
    )
    if args.validation_steps is not None:
        val_dataset = val_dataset.repeat()

    model = build_model(
        model_name,
        input_shape=input_shape,
        num_classes=args.num_classes,
        dropout_rate=args.dropout_rate,
        drop_path_rate=args.drop_path_rate,
        patch_size=args.vit_patch_size,
    )
    model.compile(
        optimizer=make_optimizer(args.learning_rate, args.weight_decay),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[
            tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
        ],
    )

    run_dir.mkdir(parents=True, exist_ok=True)
    write_model_summary(model, run_dir / "model_summary.txt")
    with (run_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                **vars(args),
                "model": model_name,
                "input_shape": list(input_shape),
                "train_files": train_files,
                "val_files": val_files,
            },
            handle,
            indent=2,
        )

    monitor = "val_loss"
    callbacks = make_callbacks(run_dir, monitor=monitor)
    if args.patience > 0:
        callbacks.append(
            tf.keras.callbacks.EarlyStopping(
                monitor=monitor,
                mode="min",
                patience=args.patience,
                restore_best_weights=True,
            )
        )

    print(f"\n=== Training {model_name} ===")
    print(f"Run dir: {run_dir}")
    model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        validation_steps=args.validation_steps,
        callbacks=callbacks,
        verbose=args.verbose,
    )
    model.save(run_dir / "last.keras")


def main() -> None:
    args = parse_args()
    model_names = [normalize_model_name(name) for name in args.model.split(",") if name.strip()]
    if not model_names:
        raise ValueError("--model must not be empty")
    if "all" in model_names and len(model_names) > 1:
        raise ValueError("--model all cannot be combined with other model names")

    configure_runtime(args.mixed_precision)

    input_shape = (
        parse_input_shape(args.input_shape)
        if args.input_shape
        else infer_input_shape(args.data_dir, args.train_pattern)
    )
    train_files = list_tfrecords(args.data_dir, args.train_pattern)
    val_files = list_tfrecords(args.data_dir, args.val_pattern)

    model_names = list(MODEL_BUILDERS) if model_names == ["all"] else model_names
    run_name = args.run_name or datetime.now().strftime("%Y%m%d-%H%M%S")
    base_run_dir = Path(args.output_dir) / run_name

    print(f"Input shape: {input_shape}")
    print(f"Train shards: {len(train_files)}")
    print(f"Validation shards: {len(val_files)}")
    print(f"Train shuffle: {not args.no_train_shuffle} (buffer={args.shuffle_buffer})")
    print(f"Models: {', '.join(model_names)}")

    for model_name in model_names:
        train_one_model(
            model_name=model_name,
            args=args,
            input_shape=input_shape,
            train_files=train_files,
            val_files=val_files,
            run_dir=base_run_dir / model_name,
        )


if __name__ == "__main__":
    main()

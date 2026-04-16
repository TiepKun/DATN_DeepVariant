from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package="deepvariant_train")
class StochasticDepth(tf.keras.layers.Layer):
    def __init__(self, drop_rate: float = 0.0, **kwargs):
        super().__init__(**kwargs)
        self.drop_rate = float(drop_rate)

    def call(self, inputs: tf.Tensor, training: bool | None = None) -> tf.Tensor:
        if self.drop_rate == 0.0:
            return inputs
        if not training:
            return inputs

        keep_prob = 1.0 - self.drop_rate
        rank = tf.rank(inputs)
        shape = tf.concat([[tf.shape(inputs)[0]], tf.ones([rank - 1], dtype=tf.int32)], axis=0)
        random_tensor = keep_prob + tf.random.uniform(shape, dtype=inputs.dtype)
        binary_tensor = tf.floor(random_tensor)
        return tf.divide(inputs, keep_prob) * binary_tensor

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({"drop_rate": self.drop_rate})
        return config


@tf.keras.utils.register_keras_serializable(package="deepvariant_train")
class GlobalResponseNorm(tf.keras.layers.Layer):
    def __init__(self, epsilon: float = 1e-6, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = float(epsilon)

    def build(self, input_shape):
        channels = int(input_shape[-1])
        self.gamma = self.add_weight(
            name="gamma",
            shape=(1, 1, 1, channels),
            initializer="zeros",
            trainable=True,
        )
        self.beta = self.add_weight(
            name="beta",
            shape=(1, 1, 1, channels),
            initializer="zeros",
            trainable=True,
        )

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        spatial_norm = tf.norm(inputs, ord=2, axis=(1, 2), keepdims=True)
        response_norm = spatial_norm / (
            tf.reduce_mean(spatial_norm, axis=-1, keepdims=True) + self.epsilon
        )
        return self.gamma * (inputs * response_norm) + self.beta + inputs

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({"epsilon": self.epsilon})
        return config


@tf.keras.utils.register_keras_serializable(package="deepvariant_train")
class AddClassToken(tf.keras.layers.Layer):
    def build(self, input_shape):
        embed_dim = int(input_shape[-1])
        self.cls_token = self.add_weight(
            name="cls_token",
            shape=(1, 1, embed_dim),
            initializer="zeros",
            trainable=True,
        )

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        batch_size = tf.shape(inputs)[0]
        cls_tokens = tf.tile(self.cls_token, [batch_size, 1, 1])
        return tf.concat([cls_tokens, inputs], axis=1)


@tf.keras.utils.register_keras_serializable(package="deepvariant_train")
class PositionEmbedding(tf.keras.layers.Layer):
    def build(self, input_shape):
        num_tokens = input_shape[1]
        embed_dim = int(input_shape[-1])
        if num_tokens is None:
            raise ValueError("PositionEmbedding requires a static token count")
        self.position_embedding = self.add_weight(
            name="position_embedding",
            shape=(1, int(num_tokens), embed_dim),
            initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02),
            trainable=True,
        )

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        return inputs + self.position_embedding


@tf.keras.utils.register_keras_serializable(package="deepvariant_train")
class TakeClassToken(tf.keras.layers.Layer):
    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        return inputs[:, 0]


def _classifier_head(
    x: tf.Tensor,
    *,
    num_classes: int,
    dropout_rate: float,
    name: str,
) -> tf.Tensor:
    if dropout_rate > 0:
        x = tf.keras.layers.Dropout(dropout_rate, name=f"{name}_dropout")(x)
    return tf.keras.layers.Dense(num_classes, dtype="float32", name=f"{name}_logits")(x)


def build_inceptionv3(
    input_shape: Sequence[int],
    num_classes: int,
    *,
    dropout_rate: float = 0.2,
    **_: object,
) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=tuple(input_shape), name="image")
    base = tf.keras.applications.InceptionV3(
        include_top=False,
        weights=None,
        input_tensor=inputs,
        pooling=None,
    )
    x = tf.keras.layers.GlobalAveragePooling2D(name="avg_pool")(base.output)
    outputs = _classifier_head(
        x,
        num_classes=num_classes,
        dropout_rate=dropout_rate,
        name="classifier",
    )
    return tf.keras.Model(inputs=inputs, outputs=outputs, name="inceptionv3")


def build_efficientnetv2_s(
    input_shape: Sequence[int],
    num_classes: int,
    *,
    dropout_rate: float = 0.2,
    **_: object,
) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=tuple(input_shape), name="image")
    base = tf.keras.applications.EfficientNetV2S(
        include_top=False,
        weights=None,
        input_tensor=inputs,
        pooling=None,
        include_preprocessing=False,
    )
    x = tf.keras.layers.GlobalAveragePooling2D(name="avg_pool")(base.output)
    outputs = _classifier_head(
        x,
        num_classes=num_classes,
        dropout_rate=dropout_rate,
        name="classifier",
    )
    return tf.keras.Model(inputs=inputs, outputs=outputs, name="efficientnetv2_s")


def _convnextv2_block(
    x: tf.Tensor,
    dim: int,
    drop_path_rate: float,
    name: str,
) -> tf.Tensor:
    shortcut = x
    x = tf.keras.layers.DepthwiseConv2D(
        kernel_size=7,
        padding="same",
        name=f"{name}_dwconv",
    )(x)
    x = tf.keras.layers.LayerNormalization(epsilon=1e-6, name=f"{name}_norm")(x)
    x = tf.keras.layers.Dense(4 * dim, name=f"{name}_pwconv1")(x)
    x = tf.keras.layers.Activation("gelu", name=f"{name}_gelu")(x)
    x = GlobalResponseNorm(name=f"{name}_grn")(x)
    x = tf.keras.layers.Dense(dim, name=f"{name}_pwconv2")(x)
    if drop_path_rate > 0:
        x = StochasticDepth(drop_path_rate, name=f"{name}_drop_path")(x)
    return tf.keras.layers.Add(name=f"{name}_add")([shortcut, x])


def build_convnextv2_tiny(
    input_shape: Sequence[int],
    num_classes: int,
    *,
    dropout_rate: float = 0.2,
    drop_path_rate: float = 0.1,
    **_: object,
) -> tf.keras.Model:
    depths = (3, 3, 9, 3)
    dims = (96, 192, 384, 768)
    total_blocks = sum(depths)
    drop_rates = np.linspace(0.0, drop_path_rate, total_blocks)
    block_index = 0

    inputs = tf.keras.Input(shape=tuple(input_shape), name="image")
    x = tf.keras.layers.Conv2D(
        dims[0],
        kernel_size=4,
        strides=4,
        padding="same",
        name="stem_conv",
    )(inputs)
    x = tf.keras.layers.LayerNormalization(epsilon=1e-6, name="stem_norm")(x)

    for stage_index, (depth, dim) in enumerate(zip(depths, dims)):
        for local_index in range(depth):
            x = _convnextv2_block(
                x,
                dim=dim,
                drop_path_rate=float(drop_rates[block_index]),
                name=f"stage{stage_index + 1}_block{local_index + 1}",
            )
            block_index += 1

        if stage_index < len(depths) - 1:
            x = tf.keras.layers.LayerNormalization(
                epsilon=1e-6,
                name=f"downsample{stage_index + 1}_norm",
            )(x)
            x = tf.keras.layers.Conv2D(
                dims[stage_index + 1],
                kernel_size=2,
                strides=2,
                padding="same",
                name=f"downsample{stage_index + 1}_conv",
            )(x)

    x = tf.keras.layers.LayerNormalization(epsilon=1e-6, name="final_norm")(x)
    x = tf.keras.layers.GlobalAveragePooling2D(name="avg_pool")(x)
    outputs = _classifier_head(
        x,
        num_classes=num_classes,
        dropout_rate=dropout_rate,
        name="classifier",
    )
    return tf.keras.Model(inputs=inputs, outputs=outputs, name="convnextv2_tiny")


def _transformer_block(
    x: tf.Tensor,
    *,
    embed_dim: int,
    num_heads: int,
    mlp_dim: int,
    dropout_rate: float,
    name: str,
) -> tf.Tensor:
    residual = x
    x = tf.keras.layers.LayerNormalization(epsilon=1e-6, name=f"{name}_attn_norm")(x)
    x = tf.keras.layers.MultiHeadAttention(
        num_heads=num_heads,
        key_dim=embed_dim // num_heads,
        dropout=dropout_rate,
        name=f"{name}_attn",
    )(x, x)
    x = tf.keras.layers.Dropout(dropout_rate, name=f"{name}_attn_dropout")(x)
    x = tf.keras.layers.Add(name=f"{name}_attn_add")([residual, x])

    residual = x
    x = tf.keras.layers.LayerNormalization(epsilon=1e-6, name=f"{name}_mlp_norm")(x)
    x = tf.keras.layers.Dense(mlp_dim, activation="gelu", name=f"{name}_mlp_dense1")(x)
    x = tf.keras.layers.Dropout(dropout_rate, name=f"{name}_mlp_dropout1")(x)
    x = tf.keras.layers.Dense(embed_dim, name=f"{name}_mlp_dense2")(x)
    x = tf.keras.layers.Dropout(dropout_rate, name=f"{name}_mlp_dropout2")(x)
    return tf.keras.layers.Add(name=f"{name}_mlp_add")([residual, x])


def build_vit_tiny(
    input_shape: Sequence[int],
    num_classes: int,
    *,
    dropout_rate: float = 0.1,
    patch_size: int = 16,
    **_: object,
) -> tf.keras.Model:
    embed_dim = 192
    depth = 12
    num_heads = 3
    mlp_dim = embed_dim * 4

    inputs = tf.keras.Input(shape=tuple(input_shape), name="image")
    x = tf.keras.layers.Conv2D(
        embed_dim,
        kernel_size=patch_size,
        strides=patch_size,
        padding="same",
        name="patch_embedding",
    )(inputs)
    x = tf.keras.layers.Reshape((-1, embed_dim), name="flatten_patches")(x)
    x = AddClassToken(name="add_class_token")(x)
    x = PositionEmbedding(name="position_embedding")(x)
    x = tf.keras.layers.Dropout(dropout_rate, name="embedding_dropout")(x)

    for index in range(depth):
        x = _transformer_block(
            x,
            embed_dim=embed_dim,
            num_heads=num_heads,
            mlp_dim=mlp_dim,
            dropout_rate=dropout_rate,
            name=f"encoder{index + 1}",
        )

    x = tf.keras.layers.LayerNormalization(epsilon=1e-6, name="encoder_norm")(x)
    x = TakeClassToken(name="class_token")(x)
    outputs = _classifier_head(
        x,
        num_classes=num_classes,
        dropout_rate=dropout_rate,
        name="classifier",
    )
    return tf.keras.Model(inputs=inputs, outputs=outputs, name="vit_tiny")


MODEL_BUILDERS: dict[str, Callable[..., tf.keras.Model]] = {
    "inceptionv3": build_inceptionv3,
    "convnextv2_tiny": build_convnextv2_tiny,
    "efficientnetv2_s": build_efficientnetv2_s,
    "vit_tiny": build_vit_tiny,
}

MODEL_ALIASES = {
    "inception_v3": "inceptionv3",
    "inception-v3": "inceptionv3",
    "convnextv2-tiny": "convnextv2_tiny",
    "convnext_v2_tiny": "convnextv2_tiny",
    "efficientnetv2-s": "efficientnetv2_s",
    "efficientnet_v2_s": "efficientnetv2_s",
    "vit-tiny": "vit_tiny",
}


def normalize_model_name(name: str) -> str:
    normalized = name.strip().lower()
    normalized = MODEL_ALIASES.get(normalized, normalized)
    if normalized not in MODEL_BUILDERS and normalized != "all":
        valid = ", ".join(["all", *MODEL_BUILDERS.keys()])
        raise ValueError(f"Unknown model {name!r}. Valid values: {valid}")
    return normalized


def build_model(
    name: str,
    input_shape: Sequence[int],
    num_classes: int,
    *,
    dropout_rate: float = 0.2,
    drop_path_rate: float = 0.1,
    patch_size: int = 16,
) -> tf.keras.Model:
    model_name = normalize_model_name(name)
    if model_name == "all":
        raise ValueError("build_model expects one concrete model name, not 'all'")
    return MODEL_BUILDERS[model_name](
        input_shape=tuple(input_shape),
        num_classes=num_classes,
        dropout_rate=dropout_rate,
        drop_path_rate=drop_path_rate,
        patch_size=patch_size,
    )


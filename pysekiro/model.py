import os

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import MultiHeadAttention, LayerNormalization, GlobalAveragePooling3D


def transformer_encoder(inputs, head_size, num_heads):
    # 层标准化
    x = LayerNormalization(epsilon=1e-6)(inputs)

    # 多头自注意力层
    attention_output = MultiHeadAttention(
        num_heads=num_heads,
        key_dim=head_size,
        dropout=0.1)(x, x)

    # 跳跃连接后的再次层标准化
    x = tf.keras.layers.Add()([attention_output, x])
    x = LayerNormalization(epsilon=1e-6)(x)

    # 前馈网络
    x = tf.keras.layers.Dense(units=head_size, activation=tf.nn.relu)(x)
    x = tf.keras.layers.Dense(units=inputs.shape[-1])(x)

    # 第二次跳跃连接
    x = tf.keras.layers.Add()([x, inputs])
    return x

gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)
    print(tf.config.experimental.get_device_details(gpus[0])['device_name'])

def MODEL(in_depth, in_height, in_width, in_channels, outputs, lr, load_weights_path=None):

    Input = tf.keras.Input(shape=[in_depth, in_height, in_width, in_channels])

    out_dim = 32
    conv_0 = tf.keras.layers.Conv3D(filters=out_dim, kernel_size=(2, 1, 1), strides=(2, 1, 1), padding='same', activation=tf.nn.relu)(Input)

    # ---------- 借鉴 P3D-B ----------

    conv_1 = tf.keras.layers.Conv3D(filters=out_dim, kernel_size=(1, 1, 1), padding='same', activation=tf.nn.relu)(conv_0)
    conv_2_0 = tf.keras.layers.Conv3D(filters=out_dim, kernel_size=(1, 3, 3), padding='same', activation=tf.nn.relu)(conv_1)
    conv_2_1 = tf.keras.layers.Conv3D(filters=out_dim, kernel_size=(3, 1, 1), padding='same', activation=None)(conv_1)
    conv_2 = tf.keras.layers.Add()([conv_2_0, conv_2_1])
    conv_2_relu = tf.nn.relu(conv_2)
    conv_3 = tf.keras.layers.Conv3D(filters=out_dim, kernel_size=(1, 1, 1), padding='same', activation=None)(conv_2_relu)
    out = tf.keras.layers.Add()([conv_0, conv_3])
    out_relu = tf.nn.relu(out)

    # -------transformer
    # Transformer部分
    pooled_output = GlobalAveragePooling3D()(out_relu)
    reshaped = tf.keras.layers.Reshape((1, -1))(pooled_output)
    transformer_output = transformer_encoder(reshaped, head_size=64, num_heads=4)
    flat = tf.keras.layers.Flatten()(transformer_output)
    # --------------------
    dense = tf.keras.layers.Dense(16,activation=tf.nn.relu)(flat)
    dense = tf.keras.layers.BatchNormalization()(dense)

    output = tf.keras.layers.Dense(outputs,activation=tf.nn.softmax)(dense)

    model = tf.keras.Model(inputs=Input, outputs=output)

    model.compile(
        optimizer=tf.keras.optimizers.RMSprop(lr),
        loss=tf.keras.losses.MeanSquaredError(),
    )

    if load_weights_path:
        if os.path.exists(load_weights_path):
            model.load_weights(load_weights_path)
            print('Load ' + load_weights_path)
        else:
            print('Nothing to load')
    return model

# ---*---

def main():
    in_depth = 10
    in_height = 50
    in_width = 50
    in_channels = 1
    outputs = 5
    lr = 0.01
    model = MODEL(in_depth, in_height, in_width, in_channels, outputs, lr)
    model.summary()

    tensorboard = tf.keras.callbacks.TensorBoard(log_dir='../logs/')

    model.fit(
        np.zeros((in_depth, in_height, in_width, in_channels)).reshape(-1, in_depth, in_height, in_width, in_channels),
        np.array([[0, 0, 0, 0, 1]]),
        verbose=0,
        callbacks=[tensorboard]
    )

# ---*---

if __name__ == '__main__':
    main()
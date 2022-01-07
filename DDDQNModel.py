import tensorflow as tf
from rl.agents import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy
from rl.processors import MultiInputProcessor
import numpy as np


def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(1000, (2 * (i // 2)) / np.float32(d_model))
    return pos * angle_rates


def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis], np.arange(d_model)[np.newaxis, :],
                            d_model)  # (10000,1) (1,40) 40

    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    pos_encoding = angle_rads[np.newaxis, ...]
    return tf.cast(pos_encoding, dtype=tf.float32)


def build_dddqn_model(n_actions, h, w, win_len, activation, data_format, kernel_size_1, kernel_size_2, kernel_size_3,
                      neuron_size_1, neuron_size_2, neuron_size_3, neuron_size_4, neuron_size_5, embedding_type):
    global ins_out, ins_input, input_ids_layer, input_attention_layer
    img_input = tf.keras.layers.Input(shape=(win_len, h, w), name='img_input')
    x = tf.keras.layers.Conv2D(kernel_size_1, (4, 4), strides=(3, 3), activation=activation, data_format=data_format)(
        img_input)
    x = tf.keras.layers.Conv2D(kernel_size_2, (4, 4), strides=(2, 2), activation=activation)(x)
    x = tf.keras.layers.Conv2D(kernel_size_3, (3, 3), strides=(2, 2), activation=activation)(x)
    x = tf.keras.layers.Flatten()(x)
    img_out = x

    if embedding_type == "transformers":
        d_model = 32
        dff = 128
        maximum_position_encoding = 1000

        ins_input = tf.keras.layers.Input(shape=(win_len, 10), name="Transformer_input")
        y = tf.keras.layers.Flatten(name="Flatten_Transformer_input")(ins_input)
        y = tf.keras.layers.Embedding(200, d_model, name="Embedding_Transformer_input")(y)

        # positional encoding
        pos = positional_encoding(maximum_position_encoding, d_model)
        y = tf.keras.layers.Add(name="Add_Embedding_Transformer_input_positional_encoding")(
            [y, pos[:, :tf.shape(y)[1], :]])

        ## self-attention
        query = tf.keras.layers.Dense(d_model, name="Query_vector")(y)
        value = tf.keras.layers.Dense(d_model, name="value_vector")(y)
        key = tf.keras.layers.Dense(d_model, name="key_vector")(y)
        attention = tf.keras.layers.Attention(use_scale=True, name="Attention_with_scale")([query, value, key])
        attention = tf.keras.layers.Dense(d_model)(attention)
        y = tf.keras.layers.Add(name="Add_attention_Dense")([y, attention])  # residual connection
        y = tf.keras.layers.LayerNormalization(epsilon=1e-6)(y)

        ## Feed Forward
        dense = tf.keras.layers.Dense(dff, activation='relu')(y)
        dense = tf.keras.layers.Dense(d_model)(dense)
        y = tf.keras.layers.Add()([y, dense])  # residual connection
        y = tf.keras.layers.LayerNormalization(epsilon=1e-6)(y)
        y = tf.keras.layers.Flatten()(y)
        ins_out = y

    elif embedding_type == "lstm":
        ins_input = tf.keras.layers.Input(shape=(win_len, 10), name="LSTM_input")
        y = tf.keras.layers.Flatten()(ins_input)
        y = tf.keras.layers.Embedding(input_dim=200, output_dim=32)(y)
        y = tf.keras.layers.LSTM(32)(y)
        y = tf.keras.layers.Flatten()(y)
        ins_out = y

    elif embedding_type == "control_vector":
        ins_input = tf.keras.layers.Input(shape=(win_len, 10), name="Control_Vector")
        y = tf.keras.layers.Flatten()(ins_input)
        ins_out = y

    x = tf.keras.layers.concatenate([img_out, ins_out])
    #x = tf.keras.layers.Dense(256, activation=activation)(x)
    x = tf.keras.layers.Dense(256, activation=activation)(x)
    x = tf.keras.layers.Dense(128, activation=activation)(x)
    x = tf.keras.layers.Dense(128, activation=activation)(x)
    x = tf.keras.layers.Dense(64, activation=activation)(x)
    #x = tf.keras.layers.Dense(64, activation=activation)(x)
    #x = tf.keras.layers.Dense(64, activation=activation)(x)
    main_output = tf.keras.layers.Dense(n_actions, activation="linear")(x)

    if embedding_type == "transformers":
        model = tf.keras.models.Model(inputs=[img_input, ins_input], outputs=main_output)
    else:
        model = tf.keras.models.Model(inputs=[img_input, ins_input], outputs=main_output)
    model.summary()
    # exit()
    return model


def build_agent(model, action, win_len, attr, value_max, value_min, value_test, nb_steps, limit, target_model_update,
                enable_double_dqn, enable_dueling_network, dueling_type, nb_steps_warmup):
    policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr=attr, value_max=value_max, value_min=value_min,
                                  value_test=value_test,
                                  nb_steps=nb_steps)
    memory = SequentialMemory(limit=limit, window_length=win_len)
    processor = MultiInputProcessor(nb_inputs=2)
    dqn = DQNAgent(model=model, memory=memory, target_model_update=target_model_update, policy=policy,
                   enable_double_dqn=enable_double_dqn,
                   enable_dueling_network=enable_dueling_network,
                   dueling_type=dueling_type, nb_actions=action, nb_steps_warmup=nb_steps_warmup, processor=processor,
                   batch_size=32)
    return dqn

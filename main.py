import sys
import DDDQNModel as dddqn
from CustomeEnvironment_ import LunarLander_Env
import yaml
import tensorflow as tf
from rl.callbacks import FileLogger, ModelIntervalCheckpoint

if __name__ == "__main__":
    try:
        metric = sys.argv[1]
        embedding_type = sys.argv[2]
        curriculum_learning = sys.argv[3]

    except:
        print("No argument was passed, please write one of the following in command line:")
        print("1) main.py test transformers")
        print("2) main.py train transformers")
        print("3) main.py test lstm")
        print("4) main.py train lstm")
        print("5) main.py test control_vector")
        print("6) main.py train control_vector")
        metric = "test"
        embedding_type = "transformers"
        curriculum_learning = "False"

    if metric is not None and embedding_type is not None:
        if metric == "train":
            pass
            # wandb.login(key="b44de666e33753ad33891ecfeffebc4df34a7289")
            # wandb.init(project='ddqn', entity='vaizerd_grey')

        with open(r'config.yaml') as file:
            config = yaml.load(file, Loader=yaml.FullLoader)

        # Initializing Variables
        # From Config File
        if curriculum_learning == "True":
            curriculum_learning = True
        elif curriculum_learning == "False":
            curriculum_learning = False

        window_length = config["global_parameters"][0]["window_length"]
        # curriculum_learning = config["global_parameters"][1]["curriculum_learning"]
        no_ins = config["global_parameters"][2]["no_ins"]

        activation_function = config["model_parameters"][0]["activation_function"]
        data_format = config["model_parameters"][1]["data_format"]
        layer_1_kernel_size = config["model_parameters"][2]["conv2d"][0]["layer_1_kernel_size"]
        layer_2_kernel_size = config["model_parameters"][2]["conv2d"][1]["layer_2_kernel_size"]
        layer_3_kernel_size = config["model_parameters"][2]["conv2d"][2]["layer_3_kernel_size"]

        layer_1_neurons = config["model_parameters"][3]["dense"][0]["layer_1_neurons"]
        layer_2_neurons = config["model_parameters"][3]["dense"][1]["layer_2_neurons"]
        layer_3_neurons = config["model_parameters"][3]["dense"][2]["layer_3_neurons"]
        layer_4_neurons = config["model_parameters"][3]["dense"][3]["layer_4_neurons"]
        layer_5_neurons = config["model_parameters"][3]["dense"][4]["layer_5_neurons"]

        optimizer_lr = config["model_parameters"][4]["compile"][0]["optimizer_lr"]
        metrics = config["model_parameters"][4]["compile"][1]["metrics"]
        nb_steps_fit = config["model_parameters"][5]["fit"][0]["nb_steps"]
        visualize = config["model_parameters"][5]["fit"][1]["visualize"]
        verbose = config["model_parameters"][5]["fit"][2]["verbose"]

        attr = config["DQNAgent_parameters"][0]["policy"][0]["attr"]
        value_max = config["DQNAgent_parameters"][0]["policy"][1]["value_max"]
        value_min = config["DQNAgent_parameters"][0]["policy"][2]["value_min"]
        value_test = config["DQNAgent_parameters"][0]["policy"][3]["value_test"]
        nb_steps_policy = config["DQNAgent_parameters"][0]["policy"][4]["nb_steps"]
        limit = config["DQNAgent_parameters"][1]["memory"][0]["limit"]
        target_model_update = config["DQNAgent_parameters"][2]["DQNAgent_function"][0]["target_model_update"]
        enable_double_dqn = config["DQNAgent_parameters"][2]["DQNAgent_function"][1]["enable_double_dqn"]
        enable_dueling_network = config["DQNAgent_parameters"][2]["DQNAgent_function"][2]["enable_dueling_network"]
        dueling_type = config["DQNAgent_parameters"][2]["DQNAgent_function"][3]["dueling_type"]
        nb_steps_warmup = config["DQNAgent_parameters"][2]["DQNAgent_function"][4]["nb_steps_warmup"]

        print(f"file closed?: {file.closed}")

        env = LunarLander_Env(metric=metric, embedding_type=embedding_type, curriculum_learning=curriculum_learning,
                              no_ins=no_ins)
        print(f"Number of actions: {env.action_space.n} & observation_space_shape: {env.observation_space.shape}")
        actions = env.action_space.n

        h, w, = env.observation_space.shape
        print(h, w)

        model = dddqn.build_dddqn_model(n_actions=actions, h=h, w=w, win_len=window_length,
                                        activation=activation_function, data_format=data_format,
                                        kernel_size_1=layer_1_kernel_size, kernel_size_2=layer_2_kernel_size,
                                        kernel_size_3=layer_3_kernel_size,
                                        neuron_size_1=layer_1_neurons, neuron_size_2=layer_2_neurons,
                                        neuron_size_3=layer_3_neurons, neuron_size_4=layer_4_neurons,
                                        neuron_size_5=layer_5_neurons,
                                        embedding_type=embedding_type)

        dqn = dddqn.build_agent(model=model, action=actions, win_len=window_length, attr=attr, value_max=value_max,
                                value_min=value_min, value_test=value_test, nb_steps=nb_steps_policy, limit=limit,
                                target_model_update=target_model_update, enable_double_dqn=enable_double_dqn,
                                enable_dueling_network=enable_dueling_network, dueling_type=dueling_type,
                                nb_steps_warmup=nb_steps_warmup)

        dqn.compile(tf.keras.optimizers.Adam(lr=optimizer_lr), metrics=[metrics])

        log_filename = f"Logs/dqn_log_{no_ins}_{embedding_type}_{curriculum_learning}.json"
        checkpoint_weight_name = f"checkpoint_model/curriculum_learning_{curriculum_learning}/{embedding_type}/dqn_weights_checkpoint.h5f"
        callback = [ModelIntervalCheckpoint(checkpoint_weight_name, interval=100000)]
        callback += [FileLogger(log_filename, interval=100)]

        if metric == "train":
            print(f"you selected training for {embedding_type}")
            history = dqn.fit(env, action_repetition=1, nb_steps=nb_steps_fit, visualize=visualize, verbose=verbose,
                              nb_max_episode_steps=1500,
                              callbacks=callback)

            dqn.save_weights(
                f"run_model/{no_ins}/curriculum_learning_{curriculum_learning}/{embedding_type}/LunarLander_preBuild_with_labels.h5f",
                overwrite=True)

            env.close()

        elif metric == "test":
            print(f"you selected testing for {embedding_type}")
            #dqn.load_weights(f"run_model/{no_ins}/curriculum_learning_{curriculum_learning}/{embedding_type}_5_million/LunarLander_preBuild_with_labels.h5f")

            dqn.load_weights("checkpoint_model/dqn_weights_checkpoint.h5f")

            dqn.test(env, nb_episodes=8, visualize=False, nb_max_episode_steps=1500)

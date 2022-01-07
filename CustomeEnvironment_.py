import random

import cv2
import numpy as np
from gym import Env
from gym.spaces import Box
from sklearn.model_selection import train_test_split

from CustomeLunarLander import LunarLander
from Reward_file import reward_shaping

import imageio
import nltk

import tensorflow as tf


def preprocess_frame(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    scale_percent = 15  # percent of original size
    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    dim = (width, height)

    # resize image
    frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

    normalized_frame = frame / 255.0
    return normalized_frame


def preprocess_sent(sent, flag):
    tokenze_word = nltk.tokenize.word_tokenize(sent)
    sent1 = []
    if flag:
        for w in tokenze_word:
            if w == "|":
                break
            else:
                sent1.append(w)
    else:
        k_index = tokenze_word.index("|")
        for i in range(k_index + 1, len(tokenze_word)):
            sent1.append(tokenze_word[i])

    modified_sent = " ".join(sent1)

    # print(f"modified sentence---------- {modified_sent}")

    return modified_sent


def sentence_embd(sent):
    docs = [sent]
    vocab_size = 200
    encoded_docs = [tf.keras.preprocessing.text.one_hot(d, vocab_size) for d in docs]
    # print(encoded_docs)
    # pad documents to a max length of 4 words
    max_length = 10
    padded_docs = tf.keras.preprocessing.sequence.pad_sequences(encoded_docs, maxlen=max_length, padding='post')
    # print(padded_docs)
    padded_docs = np.array(padded_docs[0])
    return padded_docs


class LunarLander_Env(Env):
    def __init__(self, metric, embedding_type, curriculum_learning, no_ins):
        self.vertical = 0
        self.horizontal = 0
        self.l = 0
        self.score = 0
        self.prev_shaping = None
        self.phase = 1
        self.dropdown_flag = False
        self.env_lunar = LunarLander()
        # self.episode_first = True
        self.action_space = self.env_lunar.action_space

        self.state = self.env_lunar.reset()
        self.state_img = self.env_lunar.render(mode="rgb_array")
        self.state_img = preprocess_frame(self.state_img)
        h, w = self.state_img.shape
        self.observation_space = Box(low=0, high=255, shape=(h, w), dtype=np.uint8)

        # self.trans_model = SentenceTransformer('paraphrase-distilroberta-base-v1')
        self.curriculum_learning = curriculum_learning
        self.curriculum_learning_counter = 0

        no_instructions = no_ins
        self.embd = embedding_type

        self.complex_instructions_labels_train = np.load(
            f"dataset/{no_instructions}/instructions_label_numpy_train.npy")
        self.complex_instructions_sent_train = np.load(f"dataset/{no_instructions}/instructions_sent_numpy_train.npy")
        self.complex_instructions_labels_test = np.load(f"dataset/{no_instructions}/instructions_label_numpy_test.npy")
        self.complex_instructions_sent_test = np.load(f"dataset/{no_instructions}/instructions_sent_numpy_test.npy")

        if no_instructions == 4 or no_instructions == 16 or no_instructions == 24 or no_instructions == 8:
            self.X_train = self.complex_instructions_sent_train
            self.y_train = self.complex_instructions_labels_train
            self.X_test = self.complex_instructions_sent_test
            self.y_test = self.complex_instructions_labels_test
        else:
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                self.complex_instructions_sent_train, self.complex_instructions_labels_train, test_size=0.25,
                random_state=42,
                shuffle=True)

        print(
            f"self.X_train.shape: {self.X_train.shape}, self.X_test.shape: {self.X_test.shape}, self.y_train.shape: {self.y_train.shape}, self.y_test.shape: {self.y_test.shape}")

        # self.sentence_embeddings = np.load("sentence_embeddings_numpy.npy")
        # self.labels = np.load("labels_numpy.npy")
        # self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.sentence_embeddings, self.labels, test_size=0.05, random_state=42)
        # print(self.X_train.shape, self.X_test.shape, self.y_train.shape, self.y_test.shape)
        self.metric_ = metric
        if metric == "train":
            print("selected Training split")
            if self.curriculum_learning:
                self.X = []
                self.y = []
                rows, _ = self.y_train.shape
                for i in range(rows):
                    if self.y_train[i][1] == 5:
                        self.X.append(self.X_train[i])
                        self.y.append(self.y_train[i])
                print(self.X, self.y)
                self.X = np.array(self.X)
                self.y = np.array(self.y)
                print(f"self.X.shape: {self.X.shape}, self.y.shape: {self.y.shape}")
            else:
                self.X = self.X_train
                self.y = self.y_train
            self.flag = False

            self.rows, _ = self.y.shape
            print(f"self_rows: {self.rows}")
            self.random_number = random.randint(0, (self.rows - 1))
            self.random_complex_instructions = self.X[self.random_number]
            self.random_complex_instructions_labels = self.y[self.random_number]
            self.random_complex_ins_embd = None
            self.process_sent_flag = False
            self.random_number_counter = -1

        elif metric == "test":
            print("selected testing split")

            self.X = self.X_test
            self.y = self.y_test
            self.counter = 0
            self.flag = True
            self.image_gif = []

            self.rows, _ = self.y.shape
            print(f"self_rows: {self.rows}")
            self.random_number_counter = -1
            self.random_number = self.random_number_counter % (self.rows)
            self.random_complex_instructions = self.X[self.random_number]
            self.random_complex_instructions_labels = self.y[self.random_number]
            self.random_complex_ins_embd = None
            self.process_sent_flag = False

    def step(self, action):

        labels = self.y[self.random_number]

        label = labels[self.l]
        next_state, _, done, _, lander_awake, game_over = self.env_lunar.step(action)

        if self.phase == 2 and float(next_state[1]) > 0.45:
            self.dropdown_flag = True

        reward, self.prev_shaping = reward_shaping(action=action, label=label, state=next_state,
                                                   prev_shaping=self.prev_shaping,
                                                   lander_awake=lander_awake, game_over=game_over, phase=self.phase,
                                                   dropdown=self.dropdown_flag)

        next_state_img = self.env_lunar.render(mode="rgb_array")

        if self.flag:
            self.image_gif.append(next_state_img)
            if done:
                imageio.mimsave(f"Landing_gif\\landing_{self.counter}_{label}.gif", self.image_gif)
                self.counter += 1

        next_state_img = preprocess_frame(next_state_img)

        if done:
            self.vertical = ((float(0.0) + 0.0) * 6.666666666666667) + 3.9333333333333336
            self.horizontal = (float(next_state[0]) * 10) + 10
            self.l = self.l + 1
            self.phase = self.phase + 1
            self.env_lunar.reset_(self.horizontal, self.vertical)
            done = False
            # self.episode_first = False
            if self.l == 2 or labels[1] == 5:
                # self.episode_first = True
                # print("no phase 2")
                done = True
            else:
                sec_sent = preprocess_sent(self.random_complex_instructions, False)
                self.random_complex_ins_embd = sentence_embd(sec_sent)

        self.curriculum_learning_counter += 1

        return [next_state_img, self.random_complex_ins_embd], reward, done, _

    def render(self, mode='human'):
        self.env_lunar.render(mode=mode)

    def reset(self):
        self.dropdown_flag = False

        self.state = self.env_lunar.reset()

        self.state_img = self.env_lunar.render(mode="rgb_array")

        self.state_img = preprocess_frame(self.state_img)

        if self.metric_ == "train":
            '''
            self.random_number = random.randint(0, (self.rows - 1))
            self.random_complex_instructions = self.X[self.random_number]
            self.random_complex_instructions_labels = self.y[self.random_number]
            print(f"Labels - {self.random_complex_instructions_labels}")
            '''
            self.random_number_counter += 1
            self.random_number = self.random_number_counter % (self.rows)
            self.random_complex_instructions = self.X[self.random_number]
            self.random_complex_instructions_labels = self.y[self.random_number]
            # print(f"Labels - {self.random_complex_instructions_labels}")
        else:
            self.random_number_counter += 1
            self.random_number = self.random_number_counter % (self.rows)
            self.random_complex_instructions = self.X[self.random_number]
            self.random_complex_instructions_labels = self.y[self.random_number]
            print(f"Sentence - {self.random_complex_instructions}")

        if self.random_complex_instructions_labels[1] == 5:
            self.random_complex_ins_embd = sentence_embd(self.random_complex_instructions)
        else:
            start_sent = preprocess_sent(self.random_complex_instructions, True)
            self.random_complex_ins_embd = sentence_embd(start_sent)

        self.vertical = 0
        self.horizontal = 0
        self.l = 0
        self.phase = 1
        self.score = 0

        self.prev_shaping = None
        labels = self.y[self.random_number]
        # self.episode_first = False

        self.image_gif = []

        if self.curriculum_learning and self.curriculum_learning_counter >= 50_000:
            print("Increasing difficulty")
            self.curriculum_learning = False
            self.X = None
            self.y = None
            self.X = self.X_train
            self.y = self.y_train
            print(self.X, self.y)
            print(f"self.X.shape: {self.X.shape}, self.y.shape: {self.y.shape}")
            print(
                f"self.curriculum_learning: {self.curriculum_learning} and self.curriculum_learning_counter: {self.curriculum_learning_counter}")
            self.rows, _ = self.y.shape

        return [self.state_img, self.random_complex_ins_embd]

    def close(self):
        self.env_lunar.close()

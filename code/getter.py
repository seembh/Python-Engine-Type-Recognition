"""
    Gherman Sebastian-Costin
    Politehnica University of Bucharest, CS 
    Deep Learning Algorithm to identify car engine types
    
    Based on the following article:
    Spectral features for audio based vehicle and engine classification
    Alicja Wieczorkowska, Elżbieta Kubera, Tomasz Słowik & Krzysztof Skrzypiec 
        Journal of Intelligent Information Systems 

    The majority of the code is based on Paul Mora's approach to determining 
    differences between two Formula 1 engine manufacturers
    https://becominghuman.ai/signal-processing-engine-sound-detection-a88a8fa48344

    Code improvements and completion based on Valerio Velardo's Deep Learning (for Audio)
    with Python course on Youtube (great course)
    https://www.youtube.com/playlist?list=PL-wATfeyAMNrtbkCNsLcpoAyBBRJZVlnf

    Massive thanks to stackoverflow.com and https://www.tensorflow.org/
"""
# Packages
import librosa, librosa.display
import numpy as np
import matplotlib.pyplot as plt
import os# Paths
from pathlib import Path
import sys

# Bulk processing packages
from pydub import AudioSegment
from pydub.utils import make_chunks
import math
import re
from tqdm import tqdm
import json
import copy

# Importing packages
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras
from sklearn.metrics import accuracy_score

code_path = Path.cwd()
main_path = code_path.parent.absolute()
raw_path = main_path.joinpath("raw")
data_path = main_path.joinpath("data")
output_path = main_path.joinpath("output")

gas_path = raw_path.joinpath("gasoline")
diesel_path = raw_path.joinpath("diesel")
test_path = raw_path.joinpath("test")
#Specifying plotting information
### EXAMPLE OUTPUT PHASE ###
if (sys.argv[1] == '1'):
    print("Running: EXAMPLE OUTPUT PHASE")
    dict_example = {
        "gasoline":{
        "file": gas_path.joinpath("gasoline2.wav"),
        "color": "green"
        },

        "diesel":{
        "file": diesel_path.joinpath("diesel1.wav"),
        "color": "brown"
        }
    }


    fig, axs = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(20, 10))
    SR = 22_050
    axs = axs.ravel()
    for num, engine_type in enumerate(dict_example):
        signal, sr = librosa.load(dict_example[engine_type]["file"], sr=SR)
        dict_example[engine_type]["signal"] = signal
        librosa.display.waveplot(signal, sr=sr, ax=axs[num],
                                 color=dict_example[engine_type]["color"])
        axs[num].set_title(engine_type, {"fontsize":18})
        axs[num].tick_params(axis="both", labelsize=16)
        axs[num].set_ylabel("Amplitude", fontsize=18)
        axs[num].set_xlabel("Time", fontsize=18)
    fig.savefig("{}/waveplot.png".format(output_path),
                bbox_inches="tight")

    fig, axs = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(20, 10))
    axs = axs.ravel()
    for num, engine_type in enumerate(dict_example):
        # Calculating the fourier transform
        signal = dict_example[engine_type]["signal"]
        fft = np.fft.fft(signal)
        magnitude = np.abs(fft)
        frequency = np.linspace(0, sr, len(magnitude))
        left_frequency = frequency[:int(len(frequency)/2)]
        left_magnitude = magnitude[:int(len(frequency)/2)]
    # Plotting results
        axs[num].plot(left_frequency, left_magnitude)
        axs[num].set_title(engine_type, {"fontsize":18})
        axs[num].tick_params(axis="both", labelsize=16)
        axs[num].set_ylabel("Magnitude", fontsize=18)
        axs[num].set_xlabel("Frequency", fontsize=18)
        axs[num].plot(left_frequency, left_magnitude,
                      color=dict_example[engine_type]["color"])
    fig.savefig("{}/powerspectrum.png".format(output_path),
                bbox_inches="tight")



    n_fft = 2048  # Window for single fourier transform
    hop_length = 512  # Amount for shifting to the right
    fig, axs = plt.subplots(nrows=1, ncols=2, sharey=True,
                            figsize=(20, 10))
    axs = axs.ravel()
    for num, engine_type in enumerate(dict_example):
        signal = dict_example[engine_type]["signal"]
        stft = librosa.core.stft(signal, hop_length=hop_length, 
                                 n_fft=n_fft)
        spectogram = np.abs(stft)
        log_spectogram = librosa.amplitude_to_db(spectogram)
        plot = librosa.display.specshow(log_spectogram, sr=sr,
                                        hop_length=hop_length, 
                                        ax=axs[num])
        axs[num].tick_params(axis="both", labelsize=16)
        axs[num].set_title(engine_type, {"fontsize":18})
        axs[num].set_ylabel("Frequency", fontsize=18)
        axs[num].set_xlabel("Time", fontsize=18)
    cb = fig.colorbar(plot)
    cb.ax.tick_params(labelsize=16)
    fig.savefig(r"{}/short_fourier.png".format(output_path),
                bbox_inches="tight")

    fig, axs = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(20, 10))
    axs = axs.ravel()
    for num, engine_type in enumerate(dict_example):
        signal = dict_example[engine_type]["signal"]
        MFCCs = librosa.feature.mfcc(signal, n_fft=n_fft, hop_length=hop_length,
                                         n_mfcc=13)
        plot = librosa.display.specshow(MFCCs, sr=sr, hop_length=hop_length,
                                            ax=axs[num])
        axs[num].tick_params(axis="both", labelsize=16)
        axs[num].set_title(engine_type, {"fontsize":18})
        axs[num].set_ylabel("Time", fontsize=18)
        axs[num].set_xlabel("Frequency", fontsize=18)
    cb = fig.colorbar(plot)
    cb.ax.tick_params(labelsize=16)
    fig.savefig(r"{}/mfccs.png".format(output_path),
                bbox_inches="tight")

    print("Finished: EXAMPLE OUTPUT PHASE")
### BUILDING DATASET PHASE ###
if ((sys.argv[1] == '2') or (sys.argv[1] == '3')):
    print("Running: BUILDING DATASET PHASE")
    raw_files = {
        "gasoline": gas_path,
        "diesel": diesel_path
    }
    for engine_type in [
    "gasoline",
    "diesel"
    ]:
        wav_files = os.listdir("{}/{}".format(raw_path, engine_type))
        for file in wav_files:
            if not file.startswith("."):
                file_name = "{}/{}/{}".format(raw_path, engine_type, file)
                myaudio = AudioSegment.from_file(file_name, "wav")
                chunk_length_ms = 1000
                chunks = make_chunks(myaudio, chunk_length_ms)
                chunks = chunks[:-1]
                for i, chunk in enumerate(chunks):
                    padding = 3 - len(str(i))
                    number = padding*"0" + str(i)
                    chunk_name = "{}_{}".format(re.split(".wav", file)[0], number)
                    #print ("exporting", chunk_name)
                    chunk.export("{}/{}/{}.wav".format(data_path, engine_type, chunk_name), format="wav")

    data = {
         "train": {"mfcc": [], "labels": [], "category": []},
         "test": {"mfcc": [], "labels": [], "category": []}
    }
    test_types = [
    "gasoline",
    "diesel"
    ]

    SAMPLE_RATE = 22050
    n_mfcc = 13
    n_fft = 2048
    hop_length = 512
    expected_num_mfcc_vectors = math.ceil(SAMPLE_RATE / hop_length)
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(data_path)):
    # ensure that we are not at the root level
        if dirpath is not data_path:
            # save the engine type information
            dirpath_components = dirpath.split("/")
            label = dirpath_components[-1]
            
            # looping over the wav files
            for f in tqdm(filenames):
                if not f.startswith("."):
                    if not ("test" not in f and sys.argv[1] == '3'):
                        file_path = os.path.join(dirpath, f)
                        signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)
                        # extract the mfcc from the sound snippet
                        mfcc = librosa.feature.mfcc(signal, sr=sr, 
                                                    n_fft=n_fft,
                                                    n_mfcc=n_mfcc,
                                                    hop_length=hop_length)
                        mfcc = mfcc.T.tolist()
                        # to ensure that all snippets have the same length
                        if len(mfcc) == expected_num_mfcc_vectors:
                            if ("test" in f):
                                data["test"]["mfcc"].append(mfcc)
                                data["test"]["labels"].append(i-1)
                                data["test"]["category"].append(label)
                            else:
                                data["train"]["mfcc"].append(mfcc)
                                data["train"]["labels"].append(i-1)
                                data["train"]["category"].append(label)
                
    # saving json with the results
    with open("{}/processed_data.json".format(data_path), "w") as fp:
        json.dump(data, fp, indent=4)

    print("Finished: BUILDING DATASET PHASE")
### TRAINING PHASE ###
    if (sys.argv[1] == '2'):
        print("Running: TRAINING PHASE")
        # load and convert data
        with open("{}/processed_data.json".format(data_path), "r") as fp:
            data = json.load(fp)

        inputs = np.array(data["train"]["mfcc"])
        targets = np.array(data["train"]["labels"])

        # turn data into train and testset
        (inputs_train, inputs_test,
         target_train, target_test) = train_test_split(inputs, targets,
                                                       test_size=0.3)
        # build the network architecture
        model = keras.Sequential([
            # input layer
            keras.layers.Flatten(input_shape=(inputs.shape[1],
                                              inputs.shape[2])),
        # 1st hidden layer
            keras.layers.Dense(512, activation="relu"),
            keras.layers.Dropout(0.3),
        # 2nd hidden layer
            keras.layers.Dense(256, activation="relu"),
            keras.layers.Dropout(0.3),
        # 3rd hidden layer
            keras.layers.Dense(64, activation="relu"),
            keras.layers.Dropout(0.3),
        # output layer
            keras.layers.Dense(1, activation="sigmoid")
        ])
        # compiling the network
        optimizer = keras.optimizers.Adam(learning_rate=0.0001)
        model.compile(optimizer=optimizer, loss="binary_crossentropy",
                      metrics=["accuracy"])
        model.summary()
        # train the network
        history = model.fit(inputs_train, target_train,
                            validation_data=(inputs_test, target_test),
                            epochs=300,
                            batch_size=32)




        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(20, 10))
        # create accuracy subplot
        axs[0].plot(history.history["accuracy"], label="train_accuracy")
        axs[0].plot(history.history["val_accuracy"], label="test_accuracy")
        axs[0].set_ylabel("Accuracy", fontsize=18)
        axs[0].legend(loc="lower right", prop={"size": 16})
        axs[0].set_title("Accuracy evaluation", fontsize=20)
        axs[0].tick_params(axis="both", labelsize=16)
        # create error subplot
        axs[1].plot(history.history["loss"], label="train error")
        axs[1].plot(history.history["val_loss"], label="test error")
        axs[1].set_ylabel("Error", fontsize=18)
        axs[1].legend(loc="upper right", prop={"size": 16})
        axs[1].set_title("Error evaluation", fontsize=20)
        axs[1].tick_params(axis="both", labelsize=16)
        fig.savefig("{}/accuracy_error.png".format(output_path),
                    bbox_inches="tight")
        #plt.show()

        print("Saving Model")
        model.save("my_model")
        print("Finished: TRAINING PHASE")
### TESTING PHASE ###
if (sys.argv[1] == '3'):
    print("Running: TESTING PHASE")
    reconstructed_model = keras.models.load_model("my_model")
    print("Testing predictions")
    test_inputs = np.array(data["test"]["mfcc"])
    test_targets = np.array(data["test"]["labels"])
    predictions = (reconstructed_model.predict(test_inputs) > 0.5).astype("int32")

    output_data = np.column_stack((test_targets,predictions))
    output_file = open("predictions_output", 'w')
    np.savetxt(output_file,output_data, fmt="%d", delimiter=",")
    #print(output_data)

    total_diesel = success_diesel = total_gasoline = success_gasoline = 0

    for (valid_label, predicted) in zip(test_targets, predictions):
        # DIESEL CATEGORY
        if valid_label == 0:
            total_diesel += 1
            if predicted == valid_label:
                success_diesel +=1
        # GASOLINE CATEGORY
        elif valid_label == 1:
            total_gasoline += 1
            if predicted == valid_label:
                success_gasoline +=1
            
    print("Diesel Successfully Prediction Rate: {:.2f}%".format(success_diesel/total_diesel * 100))
    print("Gasoline Successfully Prediction Rate: {:.2f}%".format(success_gasoline/total_gasoline * 100))

    print("Finished: TESTING PHASE")
# Python-Engine-Type-Recognition

Gherman Sebastian-Costin
    Politehnica University of Bucharest, CS, 
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


    Usage:
    python3 getter.py argv {> output}
    argv :
    1 - only examples to run on 2 chosen diesel-gasoline input files;
    
        Outputs the following:
            /output/mfccs.png
            /output/powerspectrum.png
            /output/short_fourier.png
            /output/waveplot.png

    2 - train algorithm
    Firstly it creates the dataset in /data/diesel and /data/diesel in .wav format
    data must be provided in /raw/diesel and /raw/gasoline in .wav format

        Outputs the following:
            output/accuracy_error.png
            data/processed_data.json
            /code/my_model - as it saves the trained model for future use


    3 - test algorithm
    Must have in /raw/diesel and /raw/gasoline test files with the name test{number}.wav

        Outputs the following:
            code/predictions_output
    
    L.E Uploaded requirements.txt

# A-Feature-Extraction-Framework-for-Measuring-Auditory-Similarity-Between-Sounds
Code related to the master's thesis titled "A Feature Extraction Framework for Measuring Auditory Similarity Between Sounds"

## Dataset

Files in the dataset folder contains five files that show the similarity sounds in the ESC-50 dataset. The covered classes are:

* Can opening
* Church bells
* Dog
* Pouring water
* Thunderstorm

## Setup

To run the code in this repo you need to clone the ESC-50 dataset from [here](https://github.com/karolpiczak/ESC-50) and place it as `../datasets/ESC-50-master`. Place the files in the `dataset` folder in this repo into the meta folder.

Make sure that you have these packages installed:

* PyTorch
* torchvision
* Fastai
* Fastaudio
* lshashpy3
* numpy

## Python code

Execute `train.py` to train the model. It relies on `model_wrap.py`, `config.py`, `tripletfolder.py`, and `utils.py`.

* `model_wrap.py` is a wrapper class that makes training the model easier
* `config.py` is the file where configuration for the model, as well as spectrogram extraction is done.
* `tripletfolder.py` is required when using the `TripletMarginLoss` loss function. It makes sure the spectrograms are presented to the model in triplets.
* `utils.py` is a set of functions used by multiple files.

To evaluate the model run `test.py`. 

To get what the random values from the dataset would be, run rand_test.py

To get the normalization values for the spectrograms, run the `get_normal_values.py`. It will output the mean and std of the dataset.

# Readme

This is the code repository that accompany the master thesis by [Gustav Madslund](https://github.com/gustavmadslund) and [Mikkel Møller Brusen](https://github.com/mikkelbrusen).

The goal of the project was to train recurrent neural netowrks to translate text between different languages.

## Software Requirements

The software is coded in Python 3.6 using the Pytorch 1.1 version. 
To run the software smoothly, it is recommended to use those versions.

The code was made to be run on a CUDA GPU but can run on CPU too, although this will take forever... 
In order to run configurations that utilize the bi-direction pre-trained embeddings, at least 17GB RAM.

## Setup & Data

In order to run all configurations the following datasets are needed:

+ **Deeploc** which is the deeploc dataset encoded as profiles
+ **Deeploc_raw** which is the deeploc dataset without encoding (raw sequences)
+ **SecPred** which is the filtered CullPDB dataset encoded as profiles. Files with _no_x have X replaced by A.   
+ **SecPred_raw** which is the CB513 dataset without encoding (raw sequences). X has been replaced by A.

all of which can be [downloaded here](https://drive.google.com/drive/folders/1-qPOetLSYrrlFvcjmt2lSAwKoR-_AXFm?usp=sharing)

The datasets should then be positioned in the [`data/` directory](data/) similarly to the already included Deeploc_raw dataset.

## Training models

Model architecture and other settings are controlled by config files in the [`configs/{task}` directory](configs/). Each config is task specific, such that subcellular localization configurations can be found in [`configs/subcel` directory](configs/subcel/) and secondary structure prediction configurations can be found in [`configs/secpred` directory](configs/secpred/).

To start training a model, we need to first give the task as argument when running `main.py` e.g. `subcel` and then choose a configuration with `--config`. For example, if we want to train with the configuration `configs/subcel/deeploc_raw`, we should use the following command:

    python3 main.py subcel --config deeploc_raw

All configurations are created such that no hyperparameters needs to be specified, although they are possible if you want to do experiments with a specific configuration. For a list of all avaiable arguments the following commands are usefull:

    python3 main.py --help
    python3 main.py subcel --help
    python3 main.py secpred --help

The best models based on validation performance will be saved under `save/{task}/{config_name}/` where task can be subcel or secpred and config_name is the configuration that is training.

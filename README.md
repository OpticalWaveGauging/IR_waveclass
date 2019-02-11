
### About
Code and data to implement Buscombe & Carini (2019) to classify wave breaking using deep neural networks

> Buscombe and Carini (in review) A data-driven approach to classifying wave breaking in infrared imagery. Submitted to Coastal Engineering 

Software and data for training deep convolutional neural network models to classify wave breaker type from IR images of breaking waves in the surf zone

### Folder structure

* \conda_env contains yml files for setting up conda environment
* \conf contains configuration files using for training models
* \train contains files using for training models
* \test contains files using for testing models
* \out contains output files from model training
* \keras_mods contains modified keras applications files

### Setting up computing environments

These task requires modifications to keras libraries. The easiest way to deal with two different keras installs is to use conda environments

First, some conda housekeeping

```
conda clean --packages
conda update -n base conda
```

1. Create a new conda environment called ```classification```

```
conda env create -f conda_env/classification.yml
```
C:\Users\ddb265\github_clones\IR_wavegauge\

2. Copy the contents of the ```keras_mods\classification\tf_python_keras_applications``` folder into the ```tensorflow\python\keras\applications``` site package in your new conda env path. For example: 

```
C:\Users\user\AppData\Local\Continuum\anaconda3\envs\classification\Lib\site-packages\tensorflow\python\keras\applications
```

Be sure to keep a copy of the existing files there in case something goes wrong.

3. Copy the contents of the ```keras_mods\classification\tf_keras_applications``` folder into the ```tensorflow\keras\applications``` site package in your new conda env path. For example: 

```
C:\Users\user\AppData\Local\Continuum\anaconda3\envs\classification\Lib\site-packages\tensorflow\keras\applications
```

4. Activate environment:

```
conda activate classification
```


Deactivate environment when finished:

```
conda deactivate
```


## Training classification models

### Extract image features 

The following has been tested with the following models: MobileNetV1, MobileNetV2, Xception, InceptionV3, InceptionResnet2, and VGG19

1. Run the feature extractor using the MobileNetV2 model, with augmented images, running ```extract_features_imaug.py``` and the configuration file ```conf/conf_mobilenet.json```:

```
python extract_CNN_features_imaug.py -c conf_mobilenet
```

2. Run the feature extractor using the Xception model, without augmented images, running ```extract_features.py``` and the configuration file ```conf/conf_xception.json```:

```
python extract_CNN_features.py -c conf_xception
```

### Train and save model

```
python train_test_model.py -c conf_mobilenet
```





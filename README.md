# Final Project: ViT-AugReg Model and Comparison

### "The Sentinels" Contributors:
Drew Sypnieski<br/>
Aaron Blume<br/>
Angela Duran<br/>
Byron Fong<br/>

![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![PyTorch Lightning](https://img.shields.io/badge/Lightning-792DE4?style=for-the-badge&logo=lightning&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![Weights and Biases](https://img.shields.io/badge/Weights_&_Biases-FFBE00?style=for-the-badge&logo=WeightsAndBiases&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black)


## Description

In 2021, IEEE-GRSS hosted the [Data Fusion Contest](https://www.grss-ieee.org/community/technical-committees/2021-ieee-grss-data-fusion-contest-track-dse/) where competitors were meant to detect human settlements in Sub-Saharan Africa using multimodal and multitemporal remote sensing data.

In this repository, we've implemented three baseline models to complete this task:
- U-Net, located in `src/models/supervised/unet.py`
- A basic convolutional neural network to perform segmentation, located in `src/models/supervised/segmentation_cnn.py`
- A modified FCN-ResNet model, located in `src/models/supervised/resnet_transfer.py`

We also developed a fourth model: a pretrained AugReg Vision Transformer (ViT) which has been modified to this task's requirements.


## Getting Started

### Virtual Environment

To ensure that you have the proper packages setup in your environment, create a [virtual environment](https://docs.python.org/3/library/venv.html) so that this project runs with the right dependencies independently of other Python projects. To get started, follow these steps:

1. Navigate to the project directory in the terminal.

2. Create a virtual environment:
   `python3 -m venv esdenv`
3. Activate the virtual environment:
   * On macOS and Linux:
        `source esdenv/bin/activate`
   * On Windows:
        `.\esdenv\Scripts\activate`
4. Install the required packages:
    `pip install -r requirements.txt`

To deactivate the virtual environment, type `deactivate`.

### The Dataset

Please download and unzip the `dfc2021_dse_train.zip` saving the `Train` directory into the `data/raw` directory. The zip file is available at the following [link](https://drive.google.com/file/d/1mVDV9NkmyfZbkSiD5lkskv_MwOuYxiog/view).

You may also find it useful to create a `data/processed` and a `data/predictions` directory.

### Weights and Biases

You can log all necessary information with respect to model training and validation by creating an account with Weights and Biases. To create an account, go to: [wandb.ai](https://wandb.ai/).


## Usage

### Training the Models

Run the `train.py` script using the default `ESDConfig` values or custom command line argument values. The model results will be logged using Weights & Biases, so feel free to change the name of the project by modifying the `wandb.init` line in `train.py`:

`wandb.init(project = <your project name>, ...)`

There is also a `sweeps.yml` file which can be used to customize and automate your model training with different combinations of hyperparameters. Run this with `train_sweeps.py` to automate your model training. More information about setting up the YAML file for Weights & Biases sweeps can be found [here](https://docs.wandb.ai/guides/sweeps/define-sweep-configuration).

Weights & Biases will log different TorchMetrics results for each model training, including `MulticlassAccuracy`, `JaccardIndex`, `IntersectionOverUnion`, and `F1Score`. This is done for both the training and validation datasets.

### Evaluating the Models

Like with training, evaluating the models is as easy as running the `evaluate.py` script. You can also use the default `EvalConfig` values or customize them with command line arguments.

This script will train the specified model and run a validation loop with that model. It will then obtain the validation satellite tiles and plot the raw RGB satellite image alongside its restitched ground truth and the model's prediction. This will allow you to visually compare the model's accuracy at classifying the regions in the original satellite image.


## License

Distributed under the MIT License. See `LICENSE` for more information.


## Contact

- Drew Sypnieski - [Github](https://github.com/Drew-1771) - asypnies@uci.edu; drewsyp@gmail.com
- Aaron Blume - [LinkedIn](https://www.linkedin.com/in/aaron-blume/) - [GitHub](https://github.com/aaronist) - amblume@uci.edu


## Acknowledgments

These are the resources we used to help develop this project.

* [Overview of Semantic Segmentation](https://www.jeremyjordan.me/semantic-segmentation/)
* [Cook Your First UNet in PyTorch](https://towardsdatascience.com/cook-your-first-u-net-in-pytorch-b3297a844cf3)
* [How to train your ViT? Data, Augmentation, and Regularization in Vision Transformers](https://arxiv.org/abs/2106.10270)
* [Tuning Hyperparameters with Weights & Biases](https://docs.wandb.ai/guides/sweeps)

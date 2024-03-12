# Final Project: DINO Model Monitoring and Comparison

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

You may also find it useful to create a `data/raw/processed` and a `data/raw/predictions` directory.

### Weights and Biases

You can log all necessary information with respect to model training and validation by creating an account with Weights and Biases. To create an account, go to: [wandb.ai](https://wandb.ai/).


## Usage

TODO


## License

Distributed under the MIT License. See `LICENSE` for more information.


## Contact

- Drew Sypnieski - [Github](https://github.com/Drew-1771) - asypnies@uci.edu; drewsyp@gmail.com
- Aaron Blume - [LinkedIn](https://www.linkedin.com/in/aaron-blume/) - [GitHub](https://github.com/aaronist) - amblume@uci.edu


## Acknowledgments

These are the resources we used to help develop this project.

* [Overview of Semantic Segmentation](https://www.jeremyjordan.me/semantic-segmentation/)
* [Cook Your First UNet in PyTorch](https://towardsdatascience.com/cook-your-first-u-net-in-pytorch-b3297a844cf3)
* [Lightly AI](https://github.com/lightly-ai/lightly)
* [Tuning Hyperparameters with Weights & Biases](https://docs.wandb.ai/guides/sweeps)

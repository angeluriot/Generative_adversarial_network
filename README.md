# Generative Adversarial Network (GAN)

![Release](https://img.shields.io/badge/Release-v1.2-blueviolet)
![Language](https://img.shields.io/badge/Language-Python-f2cb1b)
![Libraries](https://img.shields.io/badge/Libraries-Keras-00cf2c)
![Size](https://img.shields.io/badge/Size-514Mo-f12222)
![Open Source](https://badges.frapsoft.com/os/v2/open-source.svg?v=103)

<br/>

This project is a deep learning model that can create high quality images by training on a dataset *(here an example with a dataset of animals 🐶)*.

<br/>

<p align="center">
	<img src="resources/misc/thumbnail.png" width="750">
</p>
<p align="center"><b>(These animals do not exist)</b></p>

<br/>

# Summary

* **[Summary](#summary)**
* **[Dependencies](#dependencies)**
* **[Training](#training)**
* **[Testing](#testing)**
* **[Model](#model)**
* **[Tests](#tests)**
* **[Credits](#credits)**

<br/>

# Dependencies

* [**Python 3.10**](https://www.python.org/downloads/release/python-31011/)
* [**Numpy**](https://numpy.org/)
* [**Matplotlib**](https://matplotlib.org/)
* [**Pillow**](https://pillow.readthedocs.io/)
* [**Tensorflow 2.10**](https://www.tensorflow.org/)
* [**Keras 2.10**](https://keras.io/)
* [**Tensorflow addons**](https://www.tensorflow.org/addons)

<br/>

Run the following command to install the dependencies:
```shell
$ pip install -r requirements.txt
```

<br/>

# Training

* First, you need to find and download a dataset of images *(at least 5,000 images but the more the better)*. You can find a lot of datasets on [**Kaggle**](https://www.kaggle.com/datasets)

* Then, in the `gan/settings.py` file:
	* Specify the **path** to the dataset
	* Set the **size** of the images
	* Lower the **batch size**, the **min filters** or the **image size** if you don't have enough VRAM *(ResourceExhaustedError)*

* Run the `training.ipynb` file *(you can stop the training at any time and resume it later thanks to the checkpoints)*

<br/>

# Testing

* Specify the path to the model in the beginning of the `testing.ipynb` file

* Then, run the `testing.ipynb` file

<br/>

# Model

The model structure is very similar to [**StyleGAN2**](https://doi.org/10.48550/arXiv.1912.04958) *(from [**Nvidia**](https://www.nvidia.com/))* with a few differences:

* **Path Length Regularization** is missing

* I added a **tanh** activation to the generator output

* I set the **gain** of the **equalized learning rate layers** to 1.2 (instead of 1)

* I implemented **Adaptive Discriminator Augmentation (ADA)** from the paper [**Training Generative Adversarial Networks with Limited Data**](https://doi.org/10.48550/arXiv.2006.06676) *(from [**Nvidia**](https://www.nvidia.com/))* but the augmentation probability is constant *(instead of being trained)*

<br/>

# Tests

<p align="center"><b>Animal faces (256*256) <a href="https://twitter.com/DIMENSION_YT/status/1621922138054688773">[see more]</a></b></p>
<p align="center">
	<img src="resources/misc/animals.png" width="650">
</p>

<br/>

<p align="center"><b>Anime faces (256*256) <a href="https://twitter.com/DIMENSION_YT/status/1619377981159538688">[see more]</a></b></p>
<p align="center">
	<img src="resources/misc/anime.png" width="650">
</p>

<br/>

# Credits

* [**Angel Uriot**](https://github.com/angeluriot) : Creator of the project.

# Detection of Atrial Fibrillation using Temporal features in deep neural networks
> This is a final project submission for [**CS236781 — Deep Learning on Computational Accelerators**](https://vistalab-technion.github.io/cs236781)
>
> Date: September, 2020. 
## Authors

* Dan Kalifa @ [Technion](mailto:kalifadan@campus.technion.ac.il)
* Michael Mendelson-Mints @ [Technion](mailto:michael.me@campus.technion.ac.il)

## Environment setup

This project is built with `PyTorch v1.3`. All dependencies are listed in 
`environment.yml`, and below are methods to setup your local development 
environment.

### Anaconda/Miniconda

Run the following from the project's root folder:
```bash
conda env create -f environment.yml
conda activate cs236781-project
```

## Datasets

The [MIT-BIH Atrial Fibrillation Dataset](https://physionet.org/content/afdb/1.0.0/) was used for running the
experiments in this project. For a complete description of using the data see [here](./data/README.md).

## Structure

### Code

All code is implemented in the `src/` directory, which includes:
* `model/`: includes the model implementations used for running the experiments: 
[Baseline (baseline.py)](src/model/baseline.py) for the baseline model, [TCNNet (tcnnet.py)](src/model/tcnnet.py) for
the Temporal Convolutional model. [Blocks (blocks.py)](src/model/blocks.py) contains useful building blocks
used inside the model. See the paper for more details on implementation.
* [data.py](src/data.py) contains implementations of datasets used in this project, and code to handle parsing MIT-BIH
datasets using the `wfdb` library. 
  - `AFECGDataset` loads the MIT-BIH data and splits it into sequences of ECG windows.
  - `SecondDataset` loads the MIT-BIH data as independent samples of given size, for visualization and experimentation
purposes. 
  - `WaveletTransform` implements for Wavelet decomposition for data preprocessing and can optionally be used
in each dataset. 

  All datasets include a `load()` function for preparing the preprocessed datasets, as well as
backing them up for faster loading the next time. \
See code for parameters and usage documentation
* [dsp.py](src/dsp.py): Includes some utility functions used by the `WaveletTransform` module to perform
signal processing using *scipy* and *pycwt*.
* [training.py](src/training.py): Contains train/test functions for running the experiments.

## Notebooks

This project includes 2 notebooks with several experiments on utilities:

1. [Input.ipynb](src/Input.ipynb): Contains several examples on using the data and Wavelet decomposition implemented 
in this project
2. [Model.ipynb](src/Model.ipynb): Contains data loading and train/test experiments for baseline and advanced models.

## Acknowledgements

TBD



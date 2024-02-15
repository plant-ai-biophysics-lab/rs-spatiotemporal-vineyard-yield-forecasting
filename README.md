# Large-scale Spatio-temporal Yield Estimation via Deep Learning

This repository is the official implementation of the research presented in the paper "Large-scale spatio-temporal yield estimation via deep learning using satellite and management data fusion in vineyards". This project presents a novel deep learning model combining 2D U-Net and ConvLSTM for yield estimation in vineyards using satellite imagery and management practice data. The model leverages 15 weeks of Sentinel-2 observations across 4 to 6 channels (RGB, NIR, and optionally encoded time and historical yield averages). It also incorporates embedded management practice information, including cultivar types, trellis types, row spacing, and canopy distance, to improve estimation accuracy.

![Paper Figure](https://ars.els-cdn.com/content/image/1-s2.0-S016816992300827X-gr4.jpg)

## Paper Reference
For in-depth details, refer to our [paper](https://www.sciencedirect.com/science/article/pii/S016816992300827X).

## Requirements

Before running the code, ensure you have the following dependencies installed:

- Python 3.11
- PyTorch > 2.0
- NumPy
- Pandas
- Matplotlib
- Seaborn (for visualization)


## Validation Scenarios

![Validation Scenarios](https://ars.els-cdn.com/content/image/1-s2.0-S016816992300827X-gr6.jpg)

Validation scenario 1 is called “pixel-hold-out”, a common strategy in which the validation year and block are part of training distribution. Validation scenario 2, sometimes called “year-hold-out”, can be used when there are time-series observations for each block, and the train and validation datasets are split by year. Although the unseen block is used for both train and validation datasets, management practice and site conditions vary from year-to-year, leading to a more general evaluation task. Third, we investigate validation scenario 3 to address the limitations of pixel-hold-out and year-hold-out, which is called “block-hold-out”. In this scenario, the historical observations of the test block have never been used in the training process but its cultivar type has been seen by different blocks. This scenario evaluates the model’s capacity to generalize to a new block which has no historical yield observations.


## Usage
The current dataloader is designed for local data and directory structures specific to our development environment. To effectively use the model with your data, you will need to implement a custom dataloader tailored to your local file directories and dataset structure. We plan to release a sample dataset along with detailed instructions on how to adapt the model for your needs in the near future, so please stay tuned to our GitHub repository for updates.


```bash
python run.py --exp_name my_experiment --batch_size 64 --scenario block-hold-out --in_channels 6 --dropout 0.1  --lr 0.0001 --wd 0.0001 --epochs 50 



## Citing 

@article{kamangir2024large,
  title={Large-scale spatio-temporal yield estimation via deep learning using satellite and management data fusion in vineyards},
  author={Kamangir, Hamid and Sams, Brent S and Dokoozlian, Nick and Sanchez, Luis and Earles, J Mason},
  journal={Computers and Electronics in Agriculture},
  volume={216},
  pages={108439},
  year={2024},
  publisher={Elsevier}
}

Kamangir, H., Sams, B.S., Dokoozlian, N., Sanchez, L. and Earles, J.M., 2024. Large-scale spatio-temporal yield estimation via deep learning using satellite and management data fusion in vineyards. Computers and Electronics in Agriculture, 216, p.108439.


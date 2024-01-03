# Large-scale Spatio-temporal Yield Estimation via Deep Learning

This repository is the official implementation of the research presented in the paper "Large-scale spatio-temporal yield estimation via deep learning using satellite and management data fusion in vineyards". Our study leverages deep learning techniques to estimate vineyard yields, integrating satellite and management data for comprehensive analysis.

![Paper Figure](https://ars.els-cdn.com/content/image/1-s2.0-S016816992300827X-gr12.jpg)

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


```bash
python run.py --exp_name my_experiment --batch_size 64 --scenario block-hold-out --in_channels 4 --dropout 0.1  --lr 0.0001 --wd 0.0001 --epochs 50 

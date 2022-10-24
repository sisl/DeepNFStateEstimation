# UnderCurrent

## Installation
All the required Python packages are saved in `requirements.txt` and can be installed via

```
pip3 install -r requirements.txt
```
The code is based on PyTorch 1.14 dev. Since the nightly builds of Pytorch can't (at least to my knowledge) easily installed through a requirement file, executing the `install_pytorch_nightly.sh` file should install the nighlty build of pytorch as well the `nflows` package.

```
bash install_pytorch_nightly.sh
```

A new experiment can be started from the `normalizing_flows.py` file. 

```
EXPERIMENT_NAME = "tanh_max_likelihood"
TRAIN = True
```

The above configuration starts a new experiment (with training) under the name "tanh_max_likelihood". If only the evaluation of a model is required, setting `TRAIN=FALSE` only evaluates the model and creates some plots which can be found in the `figs/experiments` directory. 
## Estimated pdf after 1000 iterations
![UnderCurrent with MLP after 1000 iterations](./figs/general/first_deep_current6_999.png)

## Mapping from $\mathbf{X}$ to $\mathbf{Z}$ of Input Space
![Mapping X to Z](./figs/general/mapping_x_z6.png)

## Mapping from $\mathbf{X}$ to $\mathbf{Z}$ of Dataset
![Mapping X to Z](./figs/general/mapped_points6_999.png)

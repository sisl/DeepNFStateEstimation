# UnderCurrent
CS230 Final Project - UnderCurrent

## Installation
All the required Python packages are saved in `requirements.txt` and can be installed via

```
pip3 install -r requirements.txt
```
The code is based on PyTorch 1.14 dev. In case there are issues with the automatic installation of 1.14 dev, the pip installation (for the CPU version) can be started with 

```
pip3 install --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cpu
```

A new experiment can be started from the `normalizing_flows.py` file. 

```
EXPERIMENT_NAME = "tanh_max_likelihood"
TRAIN = True
```

The above configuration starts a new experiment (with training) under the name "tanh_max_likelihood". If only the evaluation of a model is required, setting `TRAIN=FALSE` only evaluates the model and creates some plots which can be found in the `figs/experiments` directory. 
## Estimated pdf after 1000 iterations
![UnderCurrent with MLP after 1000 iterations](first_deep_current6_999.png)

## Mapping from $\mathbf{X}$ to $\mathbf{Z}$ of Input Space
![Mapping X to Z](mapping_x_z6.png)

## Mapping from $\mathbf{X}$ to $\mathbf{Z}$ of Dataset
![Mapping X to Z](mapped_points6_999.png)

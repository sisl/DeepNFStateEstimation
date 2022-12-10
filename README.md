# UnderCurrent

## Installation
All the required Python packages are saved in `requirements.txt` and can be installed via

```
pip3 install -r requirements.txt
```
The code is based on PyTorch 1.14 dev. Since the nightly builds of Pytorch can't (at least to my knowledge) easily installed through a requirement file, executing the `install_pytorch_nightly.sh` file should install the nighlty build of pytorch as well the `nflows` package. *Important: We use our own fork of the `nflows` package for our custom conditioner functions  (i.e., Transformers)*.

```
bash install_pytorch_nightly.sh
```

## Running the nflows code
All of the training files have descriptive names. `nflows_transformer.py`, e.g., is the file for the training of normalizing flows with a Transformer embedding network. Other code follows the same structure.

## Results
When running either of the `nflows_*.py` files, a gif showing the performance of the trained normalzing flow will be saved in `./figs/experiments/*/seq.gif` where * is a placeholder for the experiment name. To demonstrate how well normalizing flows work, even for the harder bimodal dataset conditioning on a sequence of noisy observations, we include a gif showing the predictions made using a normalizing flow with a Transformer-embedding.

![](https://github.com/MarcSchlichting/UnderCurrent/figs/exepriments/transformer/seq6.gif)

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

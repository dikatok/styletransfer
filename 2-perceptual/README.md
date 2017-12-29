# Implementation of "Perceptual Losses for Real-Time Style Transfer and Super-Resolution"

## Pre-requisites
#### Libraries needed to run the notebook/python scripts
- tensorflow (1.4.0 at the time of writing)
- matplotlib
- numpy
- jupyter notebook
#### Downloadables needed (for training)
- [tensorflow vgg weights](http://www.cs.toronto.edu/~frossard/post/vgg16/)
- [MS COCO 2014 training dataset](http://cocodataset.org/#download)
- [test images](https://drive.google.com/open?id=1-OEv8ELX-RDB1DvY7i8qEJzziIxJ9T3B)

## How to run
#### Notebook (includes the training and inference part)
- for training, download the downloadables and extract them
- start up the notebook in the root/project directory then open the corresponding notebook in browser
- or open the notebook via IDE (eg. PyCharm)
#### Python scripts (inference part only)
- unzip the model available (models.zip) which is trained using Piet Mondrian's composition as style image
```
python generate.py <model_dir> <content_img_path>
```
- available options can be found via
```
python generate.py -h
```

## Differences from the original paper
- instance norm instead of normal batch norm
- upsample convolution instead of deconvolution/transposed convolution to avoid checkerboard artifact in which total variation loss can be omitted [reading 1](https://distill.pub/2016/deconv-checkerboard/), [reading2](http://forums.fast.ai/t/how-to-avoid-the-patterned-artifacts-in-image-generation/1681)

## QOL used
- [miniconda](https://conda.io/miniconda.html)
- [jupyter extension - codefolding](https://github.com/ipython-contrib/jupyter_contrib_nbextensions)
- [PyCharm](https://www.jetbrains.com/pycharm/)
- [tensorboard](https://www.tensorflow.org/get_started/summaries_and_tensorboard)

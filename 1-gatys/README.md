# Implementation of "A Neural Algorithm of Artistic Style"

## Pre-requisites
#### Libraries needed to run the notebook/python scripts
- tensorflow (1.4.0 at the time of writing)
- matplotlib
- numpy
- jupyter notebook
#### Downloadables needed
- [tensorflow vgg weights](http://www.cs.toronto.edu/~frossard/post/vgg16/)

## How to run
#### Notebook
- start up the notebook in the root/project directory then open the corresponding notebook in browser
- or open the notebook via IDE (eg. PyCharm)
#### Python scripts
```
python generate.py <style_img_path> <content_img_path>
```
- available options can be found via
```
python generate.py -h
```

## Differences from the original paper
- relu3_3 as content features instead of relu4_2
- relu1_2, relu2_2, relu3_3, relu4_3 as style features instead of relu1_1, relu2_1, relu3_1, relu4_1, relu5_1

## QOL used
- [miniconda](https://conda.io/miniconda.html)
- [jupyter extension - codefolding](https://github.com/ipython-contrib/jupyter_contrib_nbextensions)
- [PyCharm](https://www.jetbrains.com/pycharm/)
- [tensorboard](https://www.tensorflow.org/get_started/summaries_and_tensorboard)

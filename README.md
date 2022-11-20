<p align="center">
  <img width=100% src="https://ollieboyne.github.io/FIND/image/find/find-logo.png">
</p>

Training and inference code for

> **FIND: An Unsupervised Implicit 3D Model of Articulated Human Feet** \
> Oliver Boyne, [James Charles](http://www.jjcvision.com), and [Roberto Cipolla](https://mi.eng.cam.ac.uk/~cipolla/) \
> [[arXiv]](https://arxiv.org/abs/2210.12241#) [[project page]](https://ollieboyne.github.io/FIND/) [[demo]](https://ollieboyne.github.io/FIND/3d-viewer)

[To be released by BMVC - 21st November 2022]


## Installation

1) `git clone --recurse-submodules https://github.com/OllieBoyne/FIND`
2) `pip install -r requirements.txt`
3) Download [Foot3D](https://github.com/OllieBoyne/Foot3D) dataset
4) Edit `src/cfg.yaml` with dataset locations

## Quickstart

To train a new model,

`python src/train/run_exmpt.py --exp_name train_3d`

## Acknowledgement

We acknowledge the collaboration and financial support of [Trya Srl](https://snapfeet.io).

If you make use of our Foot3D dataset or the FIND model, please cite our work:

```
@inproceedings{boyne2022find,
            title={FIND: An Unsupervised Implicit 3D Model of Articulated Human Feet},
            author={Boyne, Oliver and Charles, James and Cipolla, Roberto},
            booktitle={British Machine Vision Conference (BMVC)},
            year={2022}
}
```


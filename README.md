<p align="center">
  <img width=70% src="https://ollieboyne.github.io/FIND/image/find/find-logo.png">
</p>

Official training and evaluation code for

> **FIND: An Unsupervised Implicit 3D Model of Articulated Human Feet**  \
> British Machine Vision Conference 2022 \
> [Oliver Boyne](https://ollieboyne.github.io), [James Charles](http://www.jjcvision.com), and [Roberto Cipolla](https://mi.eng.cam.ac.uk/~cipolla/) \
> [[arXiv]](https://arxiv.org/abs/2210.12241#) [[project page]](https://ollieboyne.github.io/FIND/) [[demo]](https://ollieboyne.github.io/FIND/3d-viewer)


## Installation

(Tested on macOS 12.6, Ubuntu 16.04)

1) `git clone --recurse-submodules https://github.com/OllieBoyne/FIND`
2) Install [PyTorch](https://pytorch.org/get-started/locally) and [PyTorch3D](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md)
3) `pip install -r requirements_<mac_linux/windows>.txt`
4) Download the [Foot3D](https://github.com/OllieBoyne/Foot3D) dataset
5) [Download the models](#model-downloads)
6) Edit `src/cfg.yaml` with dataset locations

## Usage

### Training models (3D)

Training parameters are defined in a `.yaml` file in the `cfgs` directory. `train_3d.yaml` is there as an example, which trains a FIND model, as well as a baseline [SUPR](https://github.com/ahmedosman/SUPR) model for comparison. To run this training:

```
python src/train/run_expmt.py --exp_name train_3d
```

### Evaluating models (3D)

Models are evaluated for 3D metrics (chamfer and keypoint error) under the `exp` folder, which they will be saved to after training. An entire directory of models within `exp` is evaluated at once, for example:

```
python src/eval/eval_3d.py --exp_name 3D_only
```

Evaluation outputs are saved to `eval_export/eval_3d`

### Part-based learning - training and evaluation

[Coming soon]

## Model downloads

- Pre-trained FIND models can be [downloaded from here](https://drive.google.com/drive/folders/13C8xftuvJTicMCsdyg-B8ZgmU_CA07iv?usp=sharing)
- To compare against [SUPR](https://github.com/ahmedosman/SUPR), please download their left_foot .npy files from their website to `src/model/SUPR/models`

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

### License


<sub>(c) Oliver Boyne, James Charles and Roberto Cipolla. Department of Engineering, University of Cambridge 2022</sub>

<sub>This software and annotations are provided by the copyright holders and contributors "as is" and any express or implied warranties, including, but not limited to, the implied warranties of merchantability and fitness for a particular purpose are disclaimed. In no event shall the copyright owner or contributors be liable for any direct, indirect, incidental, special, exemplary, or consequential damages (including, but not limited to, procurement of substitute goods or services; loss of use, data, or profits; or business interruption) however caused and on any theory of liability, whether in contract, strict liability, or tort (including negligence or otherwise) arising in any way out of the use of this software, even if advised of the possibility of such damage.</sub>


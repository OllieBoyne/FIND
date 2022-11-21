<p align="center">
  <img width=100% src="https://ollieboyne.github.io/FIND/image/find/find-logo.png">
</p>

Official training and evaluation code for

> **FIND: An Unsupervised Implicit 3D Model of Articulated Human Feet**  \
> British Machine Vision Conference 2022 \
> [Oliver Boyne](https://ollieboyne.github.io), [James Charles](http://www.jjcvision.com), and [Roberto Cipolla](https://mi.eng.cam.ac.uk/~cipolla/) \
> [[arXiv]](https://arxiv.org/abs/2210.12241#) [[project page]](https://ollieboyne.github.io/FIND/) [[demo]](https://ollieboyne.github.io/FIND/3d-viewer)


## Installation

1) `git clone --recurse-submodules https://github.com/OllieBoyne/FIND`
2) `pip install -r requirements_<mac_linux/windows>.txt`
   1) on Windows, install PyTorch3D via `python src/utils/install_pytorch3d.py`
3) Download [Foot3D](https://github.com/OllieBoyne/Foot3D) dataset
4) Edit `src/cfg.yaml` with dataset locations

## Quickstart

- 3D model training: `python src/train/run_exmpt.py --exp_name train_3d`
- Unsupervised part learning: [Coming soon]

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


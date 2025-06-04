# Implicit Neural Surface Deformation with Explicit Velocity Fields

**ICLR 2025**

[Lu Sang](https://sangluisme.github.io/), [Zehranaz Canfes](), [Dongliang Cao](https://dongliangcao.github.io/), [Florian Bernard](https://scholar.google.com/citations?user=9GrQ2KYAAAAJ&hl=en), [Daniel Cremers](https://scholar.google.com/citations?user=cXQciMEAAAAJ&hl=en)

Technical University of Munich, Munich Center for Machine Learning, 
University of Bonn

[📄 PAPER](https://arxiv.org/abs/2501.14038)


![teaser](assets/teaser.png)


![teaser](assets/shrec16.png)

##  🖊️ Intro

In this repo, we offer official code for the paper 

- [**Implicit Neural Surface Deformation with Explicit Velocity Fields**]()

and re-implementation of 3D LipMLP

- [**Learning Smooth Neural Functions via Lipschitz Regularization**](https://github.com/ml-for-gp/jaxgptoolbox/tree/main/demos/lipschitz_mlp)


## 🛠️ Setup

install the package using
```
pip install -r requirements.txt
```
Please test if the `jax` successfully with `cudnn`. 

Install jax version 0.4.25 matching your CUDA version as described here. For example for CUDA 12:
```
pip install -U "jax[cuda12]"
```
Other jax versions may also work, but have not been tested.

**Trouble shooting**

If `natsort` fail, delete the `natsort==8.4.0` in `requirements.txt`, after install the reset, run
```
pip install natsort
```

## 📏 Data Preparation

We offer 2 different dataset:

- ### Shape matching data where the correspondences are obtained from method [**Unsupervised Learning of Robust Spectral Shape Matching**](https://github.com/dongliangcao/unsupervised-learning-of-robust-spectral-shape-matching)

Please download the example [data](https://drive.google.com/file/d/1BCv3Jr1DIDxg6qiiaF4kZSj_wioEjd-e/view?usp=sharing) and extract it. We offer 2 datasets with their correspondences `Faust_r` and `shrec16_cuts`.

then run 
```
python ./datasets/preprocessing.py --data_root <TO YOUR DATA FOLDER> --corr_root <THE EXTRACT NPY FILES> --save_dir ./data/ --data_type matching
```

for example:

```
python ./datasets/preprocessing.py --data_root ./download_data/FAUST_r --corr_root ./download_data/faust_p2p --save_dir ./data/faust_r --data_type matching
```

- ### Shape matching datasets where ground truth correspondences are offered.

Please download the example [data](https://drive.google.com/file/d/1BCv3Jr1DIDxg6qiiaF4kZSj_wioEjd-e/view?usp=sharing) and extract it. We offer partial example dataset in `SMAL`.

datasets such as original SMAL and FAUST are template datasets that the vertices are ordered. To deal these datasets, please run


```
python ./datasets/preprocessing.py --data_root <TO YOUR DATA FOLDER> --save_dir ./data/
--data_type template
```

for example:

```
python ./datasets/preprocessing.py --data_root ./download_data/smal --save_dir ./data/smal --data_type template
```



## 💻 Deformation Training and Evaluation

```
python train.py --conf <CONFIG FILE> --savedir <SAVE PATH> --expname <NAME OF EXPERIMENT> --index <INDEX OF SOURCE SHAPE> --subindex <INDEX OF TARGET SHAPE> --reset
```
for example, train for pair in `FAUST_r`:

```
python train.py --conf ./conf/faust.conf --savedir ./exp --expname faust_r --index 0 --subindex 4 --reset
```

Evaluation

```
python eval.py --modeldir <SAVED TRAINING FOLDER> --steps <TIME STEP> --mc_resolution <MARCHING CUBES RESOLUTION>
```
for example:

```
python eval.py --modeldir ./exp/faust/2024_12_12_12_12_12/ --steps 5 --me_resolution 256
```
### Change settings

- You can change setting on `./conf/`.  Especially when errors such as out of memory happens, please reduce the batch_size, or MLP layers, or T. 
- We recommend time step $T_e$ during evaluation is not larger than twice of T during training, for better results. I.e., $T_e$ < 2T.  

## 📺 LipMLP Training and Evaluation

```
python train_lipmlp.py --conf <CONFIG FILE> --savedir <SAVE PATH> --expname <NAME OF EXPERIMENT> --index <INDEX OF SOURCE SHAPE> --subindex <INDEX OF TARGET SHAPE>
```
for example, train for pair in `FAUST_r`:

```
python train_lipmlp.py --conf ./conf/faust_lip.conf --savedir ./exp --expname faust_r --index 0 --subindex 4 --reset
```

Evaluation

```
python eval_lipmlp.py --modeldir <SAVED TRAINING FOLDER> --steps <TIME STEP> --mc_resolution <MARCHING CUBES RESOLUTION>
```
for example:

```
python eval_lipmlp.py --modeldir ./exp/faust/2024_12_12_12_12_12/ --steps 5 --me_resolution 256
```

note: for LipMLP please use MLP with node 512 and more than 6 layers. The Lipschitz loss weight is hardcoded with $10^{-10}$, since we find only this small value works.


### Cite
```
@inproceedings{sang2025implicit,
  title = {Implicit Neural Surface Deformation with Explicit Velocity Fields},
  author = {Sang, Lu and Canfes, Zehranaz and Cao, Dongliang and Bernard, Florian and Cremers, Daniel},
  year = {2025},
  booktitle = {ICLR},
}
```

### Check our other work

[**4Deform: Neural Surface Deformation for Robust Shape Interpolation**](https://4deform.github.io/)


[**TwoSquared: 4D Generation from 2D Image Pairs**](https://sangluisme.github.io/TwoSquared/)
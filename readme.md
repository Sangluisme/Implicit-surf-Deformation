### Implicit Neural Surface Deformation with Explicit Velocity Fields

##  🖊️ Intro

In this repo, we offer official code for the paper 

- **Implicit Neural Surface Deformation with Explicit Velocity Fields**

and re-implementation of 3D LipMLP

- [**Learning Smooth Neural Functions via Lipschitz Regularization**](https://github.com/ml-for-gp/jaxgptoolbox/tree/main/demos/lipschitz_mlp)


## 🛠️ Setup

install the package using
```
pip install -r requirements.txt
```

## 📏 Data Preparation

We offer 3 different dataset:

- Temporal Sequence Data, such as [**4D-DRESS**](https://eth-ait.github.io/4d-dress/)

Please download the dataset from the website and get the folder has structure such as:

    |--- _4D-DRESS_00135_Outer_2
        |--Take19
            |--Capture
            |--Meshes_pkl
            |--Semantic
            |--SMPL
            ....

then run 
```
python ./datasets/preprocessing.py --data_root <TO YOUR DATA FOLDER> --seq_num <YOUR SEQUENCE NUMER> --save_dir ./data/
```
for example
```
python ./datasets/preprocessing.py --data_root ./_4D-DRESS_00135_Outer_2 --seq_num 19 --save_dir ./data/
```

It will create a `Take19` folder under `data` folder, containing
    
    |-Take19
        |--mesh
        |--ptc
        |--smpl
        |--train

- Shape matching data where the correspondences are obtained from method [**Unsupervised Learning of Robust Spectral Shape Matching**](https://github.com/dongliangcao/unsupervised-learning-of-robust-spectral-shape-matching)

Please download the [data](https://drive.google.com/file/d/1zbBs3NjUIBBmVebw38MC1nhu_Tpgn1gr/view) evaluation files from [this link]() and extract it.

then run 
```
python ./datasets/preprocessing.py --data_root <TO YOUR DATA FOLDER> --corr_root <THE EXTRACT NPY FILES> --save_dir ./data/
```

for example:

```
python ./datasets/preprocessing.py --data_root ./FAUST_r --corr_root ./fmnet_p2p/faust_p2p --save_dir ./data/faust_r
```

- Shape matching datasets where ground truth correspondences are offered.

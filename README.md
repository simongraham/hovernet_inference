# HoVer-Net Inference Code

HoVer-Net Tile and WSI processing code for simultaneous nuclear segmentation and classification in histology images. <br />

[Link](https://www.sciencedirect.com/science/article/abs/pii/S1361841519301045?via%3Dihub) to Medical Image Analysis paper.  <br />

**NEWS:** Our model achieved the best performance in the [MoNuSAC challenge](https://monusac-2020.grand-challenge.org/).  <br />

We will now be primarily supporting the PyTorch version of this code, which also enables model training. For more information please refer to [this repository](https://github.com/vqdang/hover_net).  <br />


## Set up envrionment

```
conda create --name hovernet python=3.6
conda activate hovernet
pip install -r requirements.txt
```

Glymur requires OpenJPEG as a dependency. If this is not installed, use `conda install -c conda-forge openjpeg`.

## Running the code

Before running the code, download the HoVer-Net weights [here](https://drive.google.com/drive/folders/1NwnF71vuhMB3QeV9R5CFDyYFBGaexf_f?usp=sharing). There are two checkpoint files that are available to use: `pannuke.npz` and `monusac.npz`, which correspond to the dataset that they were trained on. See below for licensing details. 

Usage:
```
  python run.py [--gpu=<id>] [--mode=<mode>] [--model=<path>] [--input_dir=<path>] [--output_dir=<path>] \
      [--cache_dir=<path>] [--batch_size=<n>] [--inf_tile_shape=<n>] [--proc_tile_shape=<n>] \
      [--postproc_workers=<n>] [--return_probs]
  python run.py (-h | --help)
  python run.py --version
```
Options:
```
  -h --help                  Show this string.
  --version                  Show version.
  --gpu=<id>                 GPU list. [default: 0]
  --mode=<mode>              Inference mode. 'tile' or 'wsi'. [default: tile]
  --model=<str>              Choose either `pannuke` or `monusac` to use model trained on corresponding dataset. [default: pannuke]
  --input_dir=<path>         Directory containing input images/WSIs.
  --output_dir=<path>        Directory where the output will be saved. [default: output/]
  --cache_dir=<path>         Cache directory for saving temporary output. [default: cache/]
  --batch_size=<n>           Batch size. [default: 25]
  --inf_tile_shape=<n>       Size of tiles for inference (assumes square shape). [default: 10000]
  --proc_tile_shape=<n>      Size of tiles for post processing (assumes square shape). [default: 2048]
  --postproc_workers=<n>     Number of workers for post processing. [default: 10]
  --return_probs             Whether to return the class probabilities for each nucleus
```

Example:
```
python run.py --gpu='0' --mode='roi' --model='pannuke.npz' --input_dir='tile_dir' --output_dir='output'
python run.py --gpu='0' --mode='roi' --model='monusac.npz' --input_dir='tile_dir' --output_dir='output'
python run.py --gpu='0' --mode='wsi' --model='pannuke.npz' --input_dir='wsi_dir' --output_dir='output'
python run.py --gpu='0' --mode='wsi' --model='monusac.npz' --input_dir='wsi_dir' --output_dir='output' --return_probs
```

There are two modes for running this code: `tile` and `wsi`.

* `tile`
    * **Input**: standard image file - for example, `.jpg` or `.png`
    * **Output 1**: Overlaid results on image (`.png` file)
    * **Output 2**: Instance segmentation map (`.npy` file)
    * **Output 3**: Instance dictionary (`.json` file)

* `wsi`
    * **Input**: Whole-slide image - for example, `.svs`, `.ndpi`, `.tiff`, `.mrxs`, `.jp2`
    * **Output 1**: Low resolution thumbnail (`.png` file)
    * **Output 2**: Binary tissue mask at the same resolution of the thumbnail (`.png` file)
    * **Output 3**: Instance dictionary (`.json` file)

In `wsi` mode, the WSI is broken into tiles and each tile is processed independently. `--inf_tile_shape` may be used to alter the size of tiles if needed. Similary, the post processing tile can be modified at `--proc_tile_shape`. Using tiles during post processing speeds up overall time and prevents any potential memory errors. <br />

To access the `.json` file, use: 
```
with open(json_path) as json_file:
    data = json.load(json_file)
    for inst in data:
        inst_info = data[inst]
        inst_centroid = inst_info['centroid']
        inst_contour = inst_info['contour']
        inst_type = inst_info['type']
        inst_prob = inst_info['probs']
```

Here, `centroid` and `contour` are the coordinates of the centroid and contours coordinates of each instance. `type` is the prediction of the nuclear type, which is an integer from 0 to `N`, where `N` is the number of classes. The instance will be labelled as 0 if all nuclear pixels have been predicted as the background class. `probs` is the per class probabilities of each nucleus. The probability of each class is determined by the proportion of pixels assigned to each class in each nucleus.

## Datasets

In this repository, we provide checkpoints trained on two datasets:

- [PanNuke Dataset](https://warwick.ac.uk/fac/sci/dcs/research/tia/data/pannuke)
- [MoNuSAC Challenge Dataset](https://monusac-2020.grand-challenge.org/)

The network will output an intefer value for each nuclear instance denoting the class prediction. The meaning of these values for each dataset is provided below: <br />

**PanNuke:**
- 0: Background
- 1: Neoplastic
- 2: Inflammatory
- 3: Connective
- 4: Dead
- 5: Non-Neoplastic Epithelial

**MoNuSAC:**
- 0: Background
- 1: Epithelial
- 2: Lymphocyte
- 3: Macrophage
- 4: Neutrophil

Note, in the MoNuSAC dataset the positive classes do not span **all** nuclei. For example, fibroblasts are treated as background.

## Citation 

If any part of this code is used, please give appropriate citation. <br />

BibTex entry: <br />
```
@article{graham2019hover,
  title={Hover-net: Simultaneous segmentation and classification of nuclei in multi-tissue histology images},
  author={Graham, Simon and Vu, Quoc Dang and Raza, Shan E Ahmed and Azam, Ayesha and Tsang, Yee Wah and Kwak, Jin Tae and Rajpoot, Nasir},
  journal={Medical Image Analysis},
  pages={101563},
  year={2019},
  publisher={Elsevier}
}

@inproceedings{gamper2019pannuke,
  title={PanNuke: an open pan-cancer histology dataset for nuclei instance segmentation and classification},
  author={Gamper, Jevgenij and Koohbanani, Navid Alemi and Benet, Ksenija and Khuram, Ali and Rajpoot, Nasir},
  booktitle={European Congress on Digital Pathology},
  pages={11--19},
  year={2019},
  organization={Springer}
}

@article{gamper2020pannuke,
  title={PanNuke Dataset Extension, Insights and Baselines},
  author={Gamper, Jevgenij and Koohbanani, Navid Alemi and Graham, Simon and Jahanifar, Mostafa and Benet, Ksenija and Khurram, Syed Ali and Azam, Ayesha and Hewitt, Katherine and Rajpoot, Nasir},
  journal={arXiv preprint arXiv:2003.10778},
  year={2020}
}

@article{verma2021monusac2020,
  title={MoNuSAC2020: A Multi-organ Nuclei Segmentation and Classification Challenge},
  author={Verma, Ruchika and Kumar, Neeraj and Patil, Abhijeet and Kurian, Nikhil Cherian and Rane, Swapnil and Graham, Simon and Vu, Quoc Dang and Zwager, Mieke and Raza, Shan E Ahmed and Rajpoot, Nasir and others},
  journal={IEEE Transactions on Medical Imaging},
  year={2021},
  publisher={IEEE}
}
```

## Extra Notes

In this repository, we use 3x3 valid convolution in the decoder, as opposed to 5x5 convolution in the original paper. This leads to a slightly larger output and consequently speeds up inference, which is especially important for WSI processing. For further information on how to run the models, refer to the `usage.ipynb` jupyter notebook. <br />

Models were trained on data at ~40x objective magnification. Therefore, for tile processing, ensure that your data is also at this magnification level. For WSI processing, we ensure patches are processed at 40x. For this, if the slide is scanned < 40x, we scale each patch before before input to HoVer-Net.

## License

Note that the PanNuke dataset is licensed under [Attribution-NonCommercial-ShareAlike 4.0 International](http://creativecommons.org/licenses/by-nc-sa/4.0/), therefore the derived weights for HoVer-Net are also shared under the same license. Please consider the implications of using the weights under this license on your work and it's licensing. 

## Authors and Contributors

Authors:

- [Simon Graham](https://github.com/simongraham)
- [Quoc Dang Vu](https://github.com/vqdang)

See the list of [contributors](https://github.com/simongraham/hovernet_inference/graphs/contributors) who participated in this project.


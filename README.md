# HoVer-Net Inference Code

HoVer-Net ROI and WSI processing code for simultaneous nuclear segmentation and classification in histology images. <br />
If you require the model to be trained, refer to the [original repository](https://github.com/vqdang/hover_net).  <br />

[Link](https://www.sciencedirect.com/science/article/abs/pii/S1361841519301045?via%3Dihub) to Medical Image Analysis paper. 

## Repository Structure

- `src/` contains executable files used to run the model
- `misc/`contains util scripts
- `model/` contains scripts that define the architecture of the segmentation models
- `postproc/` contains post processing utils
- `config.py` is the configuration file. Paths need to be changed accordingly
- `infer.py` is the main inference file
- `JP2Image.m` and `read_region.m` are matlab scripts for processing `.jp2` WSIs


## Running the code

Before running the code:
+ [Download](https://drive.google.com/file/d/1k1GSsQkFkSjYY0eXi2Kx7Hlj8AGrhOOP/view?usp=sharing) HoVer-Net weights
+ In `config.py`, set: <br />
`self.inf_model_path`: location of `hovernet.npz` weights file <br />
`self.inf_output_dir`: directory where results are saved
+ If processing WSIs, set: <br />
`self.inf_wsi_ext` : WSI file extension <br />
`self.inf_wsi_dir` : directory where WSIs are located
+ If processing ROIs, set: <br />
`self.inf_imgs_ext` : ROI file extension <br />
`self.inf_data_dir` : directory where ROIs are located

Note, other hyperparameters such as batch size and WSI processing level can be tuned. Refer to comments in `config.py.` <br />

To run, use: <br />

`python infer.py --gpu=<gpu_list> --mode=<inf_mode>` <br />

`<gpu_list>` is a comma separated list indicating the GPUs to use. <br />
`<inf_mode>` is a string indicating the inference mode. Use either:

- `'roi'`
- `'wsi'`

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
```

## Extra Information

The network in this repository was trained on the Pecan dataset (under review at CVPR), where images are of size 256x256. This explains the slight difference in the input size of HoVer-Net compared to the original paper. In this repository, we also use 3x3 valid convolution in the decoder, as opposed to 5x5 convolution in the paper. This speeds up inference, which is especially important for WSI processing.

Please note, the Pecan dataset is not currently publicly available and until acceptance of the paper, the model weights trained on this dataset must be only circulated within the lab. 

## Task list

- [x] ROI segmentation code
- [x] WSI segmentation code
- [ ] Give environment instructions

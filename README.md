# HoVer-Net Inference Code

HoVer-Net inference code for simultaneous nuclear segmentation and classification

[Download HoVer-Net weights](https://drive.google.com/file/d/1k1GSsQkFkSjYY0eXi2Kx7Hlj8AGrhOOP/view?usp=sharing)

To run, use: <br\>

`python infer.py --gpu=<gpu_list> --mode=<inf_mode>` <br\>

`<gpu_list>` is a comma separated list indicating the GPUs to use.
`<inf_mode>` is a string indicating the inference mode. Use either:

- `'roi_seg'`
- `'wsi_coords'`

# Benchmark modern historical map vectorization processes

![Data and object instances](dataset/fig/map_animation.gif)

## Abstract

Shape vectorization is a key stage of the digitization of high-scale historical maps, especially city maps. Having access to digitized buildings, building blocks, street networks and other typical from historical content opens many ways for historical studies: change tracking, morphological analysis, density estimations. In the context of the digitization of Paris atlases produced of the course of the 19th and early 20th centuries, we designed a processing pipeline capable of extracting closed shaped from historical maps, formalized as an instance segmentation problem, in an attempt to reduce the amount of manual work required. This pipeline relies on an edge filtering stage using deep filters, a closed shape extraction stage using a watershed transform, and a raster-to-vector conversion stage.

The following contributions are introduced:

- a public dataset over which an extensive benchmark is performed;
- a comparison of the performance of state-of-the-art deep edge detectors, among which vision transformers, and several deep and classical watershed approaches;
- a joint optimization of the edge detection and shape extraction stages;
- a study of the effects of augmentation techniques.

Results are made reproducible and reusable thanks to public data, code and results.

## Usage 

- **Download** 

The dataset can be downloaded through [Zenodo]()
The pretrained weight can be downloaded through [Zenodo]()

<!-- [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.6519817.svg)](https://zenodo.org/record/6519817#.Yq30V3VBzRL) -->

Download the dataset and put it into the folder [Dataset](./dataset/)

- **Data loading** 

This repository contains: 

```markdown
📂IJGIS_benchmark_vector
 ┣ 📂benchmark         # Benchmark scripts for both datasets
 ┣ 📂config            # Config files for vision image transformers
 ┣ 📂data              # Dataset loader
 ┣ 📂dataset           # Datasets files including images and ground truths
 ┣ 📂demo              # Demos for the algorithms used in this paper
 ┣ 📂evaluation        # Evaluation code for pixel and topology evaluation
 ┃ ┣ 📂all_eval        # Evaluation code for historical map vectorization task
 ┃ ┃ ┣ 📂pixel_eval    # Pixel evaluation code
 ┃ ┃ ┣ 📂topo_eval     # Topology evaluation code 
 ┣ 📂inference         # Inferencing maps with trained models
 ┣ 📂licenses          # Licenses
 ┣ 📂loss              # Pixel and topology losses
 ┣ 📂model             # Pytorch models
 ┣ 📂pretrain_weight   # Pretrained weight for training
 ┣ 📂training          # Training scripts
 ┣ 📂utils             # Some utility files
 ┣ 📂watershed         # Watershed segmentation code {binary files + python version}
 ```

- **Smart dataloader**

In order to process maps with large size, we introduce [Smart data loader](./data/smart_data_loader.py) which can divide the image into (overlap or non-overlap) patches, then feed it into pytorch dataloader. 

- **Training the maps**

We seperated the training [with topology losses](./training/train_loss.py) and [without topology losses](./training/train_model_only.py) into two different training files.

To start training:

```bat

cd training
python train_model_only.py --model_type <model_type> --cuda --gpu <gpu> --lr <learning_rate>

```

or

```bat

cd training
python train_loss.py --model_type <model_type> --cuda --gpu <gpu> --lr <learning_rate>

```

The training results is saved in the folder: [training_info](./training_info/)

```markdown
📂training_info
 ┣ 📂 params                  # Saving every weights in the training
 ┣ 📂 reconstruction_png      # Reconstruct patch predictions into full maps
 ┣ <model>.txt                # Training logs
```


- **Joint optimization validation maps** 

Joint optimization by grid search every predictions of epochs with Meyer waterhsed segmentation:

```bat
cd benchmark/
python run_bench_mws.py --image_dir ./training_info/<model_name>/reconstruct_png
```

or with binarization of EPM + edge filteirng (waterhsed segmentation with area=0, dynamics=0):

```bat
cd benchmark/
python run_bench_ccfilter.py --image_dir ./training_info/<model_name>/reconstruct_png
```

- **Evaluation test maps** 

Evaluation the test maps and return pixel and topology evaluation results and save it into .json file.

```bat
cd inferencing/
python test_mws.py --cuda --gpu <gpu> --model_type <model type> -d <dyanmic value> -a <area value> --model <best model weight .pth file> 
```

- **Infrencing unseen historical maps with watershed segmentation** 

```bat
cd inferencing/
python test_mws.py --cuda --unseen --gpu <gpu> --model_type <model type> -d <dyanmic value> -a <area value> --model <best model weight .pth file> --original_image <original_image_path>
```

## Benchmark table

- **Joint optimization of EPM with Meyer watershed with area and dynamics filtering**

|   model                 |   area |   dynamic |   pq_val |   rq_val |   sq_val |   B_f1 |   B_p |   B_r |   ClDice |   Complete |   Correct |   Quality |   pq_test |   rq_test |   sq_test |
|:------------------------|-------:|----------:|---------:|---------:|---------:|-------:|------:|------:|---------:|-----------:|----------:|----------:|----------:|----------:|----------:|
| unet (JO+MWS)           |     50 |        10 |    60.35 |    88.15 |    68.47 |   0.79 |  0.56 |  1.29 |    51.3  |      97.23 |     88.64 |     86.46 |     47.12 |     54.3  |     86.77 |
| unet_hws (JO+HWS)       |    100 |         0 |    59.2  |    87.3  |    67.8  |   0.89 |  0.61 |  1.59 |    49.08 |      96.78 |     89.25 |     86.67 |     46.31 |     54.44 |     85.07 |
| mini-unet (JO+MWS)      |    100 |        10 |    56.66 |    87.72 |    64.59 |   1.01 |  0.67 |  2.03 |    51.54 |      97.98 |     87.65 |     86.1  |     45.11 |     52.48 |     85.96 |
| vit (JO+MWS)            |    500 |        10 |    38.64 |    80.9  |    47.77 |   0.75 |  0.55 |  1.21 |    33.82 |      98.45 |     92.36 |     91.04 |     34.71 |     43.15 |     80.43 |
| pvt (JO+MWS)            |    400 |         9 |    45.72 |    85.42 |    53.52 |   0.63 |  0.48 |  0.92 |    45.79 |      97.38 |     89.1  |     87.01 |     36.65 |     44.17 |     82.97 |
| bal (JO+MWS)            |     50 |         1 |    63.11 |    87.63 |    72.02 |   0.84 |  0.59 |  1.46 |    49.8  |      98.3  |     85.44 |     84.2  |     45.65 |     52.9  |     86.3  |
| topo (JO+MWS)           |    100 |         6 |    59.91 |    88.08 |    68.02 |   1.24 |  0.76 |  3.23 |    50.87 |      92.59 |     87.48 |     81.75 |     36.87 |     43.78 |     84.22 |
| mosin (JO+MWS)          |     50 |         1 |    57.68 |    88.32 |    65.31 |   0.09 |  0.08 |  0.09 |    55.85 |      94.22 |     90.29 |     85.55 |     36.01 |     41.19 |     87.44 |
| hed (JO+MWS)            |    400 |        10 |    47.64 |    86.85 |    54.85 |   1.06 |  0.69 |  2.27 |    50.24 |      99.35 |     84.02 |     83.56 |     40.75 |     47.92 |     85.02 |
| hed_pretrain (JO+MWS)   |    400 |        10 |    51.81 |    87.45 |    59.24 |   0.82 |  0.58 |  1.38 |    53.96 |      95.69 |     90.69 |     87.13 |     43.66 |     50.67 |     86.15 |
| bdcn (JO+MWS)           |    400 |        10 |    43.53 |    85.54 |    50.89 |   0.99 |  0.66 |  1.95 |    44.89 |      98.39 |     77.7  |     76.72 |     36.96 |     43.76 |     84.46 |
| bdcn_pretrain (JO+MWS)  |    400 |         9 |    54.96 |    88.55 |    62.07 |   0.87 |  0.61 |  1.54 |    55.87 |      98.4  |     90.72 |     89.39 |     46.97 |     53.79 |     87.32 |
| unet_bri (JO+MWS)       |    100 |         6 |    57.29 |    88.15 |    64.99 |   0.84 |  0.59 |  1.44 |    50    |      97.48 |     89.19 |     87.18 |     47.19 |     54.41 |     86.73 |
| unet_aff (JO+MWS)       |    100 |         9 |    60.98 |    87.88 |    69.39 |   1.11 |  0.71 |  2.47 |    51.23 |      98.2  |     87.02 |     85.65 |     47.66 |     55.13 |     86.46 |
| unet_bri_aff (JO+MWS)   |    100 |        10 |    61.14 |    88.12 |    69.38 |   1.05 |  0.69 |  2.19 |    53.22 |      98.57 |     89.42 |     88.27 |     50.74 |     58.48 |     86.77 |
| unet_hom (JO+MWS)       |    200 |        10 |    58.44 |    87.88 |    66.5  |   0.96 |  0.65 |  1.84 |    52.63 |      98.01 |     87.47 |     85.95 |     49.62 |     57.12 |     86.88 |
| unet_bri_hom (JO+MWS)   |    200 |        10 |    59.47 |    88.22 |    67.41 |   1    |  0.67 |  2    |    52.75 |      98.71 |     88.65 |     87.63 |     50.43 |     58.2  |     86.66 |
| unet_tps (JO+MWS)       |    100 |        10 |    59.81 |    88.28 |    67.75 |   0.93 |  0.63 |  1.73 |    52.71 |      98.1  |     87.47 |     86.01 |     47.87 |     55.07 |     86.92 |
| unet_bri_tps (JO+MWS)   |    100 |         7 |    59.57 |    88.2  |    67.54 |   0.9  |  0.62 |  1.64 |    55.08 |      98.29 |     89.41 |     88.05 |     51.08 |     58.82 |     86.84 |
| deep_watershed (JO+MWS) |      0 |         0 |    53.95 |    87.41 |    61.73 |   0.65 |  0.49 |  0.97 |    49.28 |      89.35 |     89.87 |     81.18 |     28.49 |     33.55 |     84.92 |

- **Joint optimization of EPM with CC-labelling and edge thinning**

|  model              |   pq_val |   rq_val |   sq_val |   B_f1 |   B_p |   B_r |   ClDice |   Complete |   Correct |   Quality |   pq_test |   rq_test |   sq_test |
|:---------------------------|---------:|---------:|---------:|-------:|------:|------:|---------:|-----------:|----------:|----------:|----------:|----------:|----------:|
| unet (CC=0.5,ET)           |    46.82 |    87.48 |    53.52 |   0.69 |  0.51 |  1.05 |    53.58 |      97    |     88.02 |     85.69 |     41.16 |     48.22 |     85.36 |
| mini-unet (CC=0.5,ET)      |    51.75 |    87.25 |    59.31 |   1.39 |  0.82 |  4.56 |    36.78 |      99.66 |     61.82 |     61.69 |     31.28 |     38.41 |     81.44 |
| vit (CC=0.5,ET)            |    29.85 |    80.55 |    37.06 |   0.69 |  0.51 |  1.06 |    33.69 |      98.37 |     91.95 |     90.58 |     28.8  |     37.16 |     77.51 |
| pvt (CC=0.5,ET)            |    35.16 |    84.97 |    41.38 |   0.48 |  0.39 |  0.63 |    44.84 |      97.45 |     88.18 |     86.19 |     25.4  |     31.11 |     81.66 |
| bal (CC=0.5,ET)            |    57.96 |    87.63 |    66.14 |   0.87 |  0.61 |  1.55 |    47.06 |      98.13 |     79.37 |     78.18 |     43.43 |     51.44 |     84.42 |
| topo (CC=0.5,ET)           |    56.77 |    87.67 |    64.76 |   0.06 |  0.06 |  0.06 |    55.12 |      90.14 |     91.23 |     82.95 |     30.42 |     35.5  |     85.68 |
| mosin (CC=0.5,ET)          |    58.52 |    86.49 |    67.66 |   0.75 |  1.2  |  0.55 |    55.16 |      85.22 |     92.02 |     79.36 |     18.85 |     21.69 |     86.93 |
| hed (CC=0.5,ET)            |    52.19 |    86.76 |    60.16 |   0.87 |  0.61 |  1.55 |    51    |      98.46 |     84.86 |     83.75 |     42.7  |     50.11 |     85.22 |
| hed_pretrain (CC=0.5,ET)   |    32.44 |    86.99 |    37.3  |   0.89 |  0.62 |  1.61 |    50.77 |      99.01 |     85.65 |     84.93 |     44.54 |     52.27 |     85.21 |
| bdcn (CC=0.5,ET)           |    35    |    86.03 |    40.68 |   0.03 |  0.02 |  0.03 |    45    |      93.36 |     77.75 |     73.68 |     23.58 |     27.89 |     84.57 |
| bdcn_pretrain (CC=0.5,ET)  |    55.67 |    86.99 |    63.99 |   0.51 |  0.41 |  0.69 |    55.51 |      95.89 |     92.8  |     89.25 |     41.44 |     48.15 |     86.07 |
| unet_bri (CC=0.5,ET)       |    40.47 |    87.82 |    46.09 |   0.3  |  0.26 |  0.35 |    53.48 |      94.93 |     88.32 |     84.34 |     35.34 |     41.19 |     85.8  |
| unet_aff (CC=0.5,ET)       |    56.79 |    87.1  |    65.2  |   0.7  |  0.52 |  1.08 |    51.72 |      96.79 |     86.87 |     84.44 |     42.56 |     49.71 |     85.61 |
| unet_bri_aff (CC=0.5,ET)   |    59.25 |    87.57 |    67.66 |   0.66 |  0.5  |  0.99 |    53.46 |      97.01 |     90.47 |     88.02 |     46.74 |     54.32 |     86.03 |
| unet_hom (CC=0.5,ET)       |    58.43 |    86.82 |    67.3  |   0.69 |  0.51 |  1.05 |    53.3  |      96.4  |     88.88 |     86.03 |     45.78 |     53.66 |     85.32 |
| unet_bri_hom (CC=0.5,ET)   |    59.95 |    86.87 |    69.01 |   0.79 |  0.56 |  1.3  |    54.15 |      97.55 |     89.46 |     87.49 |     48.46 |     56.91 |     85.17 |
| unet_tps (CC=0.5,ET)       |    57.81 |    87.85 |    65.81 |   0.34 |  0.29 |  0.4  |    54.74 |      96.15 |     89.4  |     86.31 |     38.84 |     44.91 |     86.49 |
| unet_bri_tps (CC=0.5,ET)   |    56.87 |    87.81 |    64.77 |   0.44 |  0.36 |  0.56 |    55.8  |      96.59 |     90.7  |     87.89 |     44.2  |     51.09 |     86.52 |
| deep_watershed (CC=0.5,ET) |    53.96 |    87.4  |    61.73 |   0.65 |  0.49 |  0.97 |    49.28 |      89.35 |     89.87 |     81.18 |     28.49 |     33.55 |     84.92 |

## Citation

If you use this repository your work, please cite our [paper](path-to-the-paper-page):

```
...
```

## Credits

- This work was supported by ANR project SoDUCo ANR-18-CE38-0013.
- The source of the map [Historical_map_resource]().
- We thank Zenodo for hosting the dataset.

---
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

Model_name and its related model_type (download weight file (in release) into the folder ./pretrain_weight/ ):

| Model_name                     | model_type    |
|:------------------------------ |:------------- |
| hed_best_weight.pth            | hed           |
| hed_pretrain_best_weight.pth   | hed_pretrain  |
| bdcn_best_weight.pth           | bdcn          |
| bdcn_pretrain_best_weight.pth  | bdcn_pretrain |
| mini_unet_best_weight.pth      | mini-unet     |
| mosin_best_weight.pth          | mosin         |
| topo_best_weight.pth           | topo          |
| bal_best_weight.pth            | bal           |
| pathloss_best_weight.pth       | pathloss      |
| vit_best_weight.pth            | vit           |
| pvt_best_weight.pth            | pvt           |
| unet_best_weight.pth           | unet          |
| unet_hws_best_weight.pth       | unet          |
| unet_aff_best_weight.pth       | unet_aff      |
| unet_bri_aff_best_weight.pth   | unet_bri_aff  |
| unet_bri_best_weight.pth       | unet_bri      |
| unet_bri_hom_best_weight.pth   | unet_bri_hom  |
| unet_tps_best_weight.pth       | unet_tps      |
| unet_bri_tps_best_weight.pth   | unet_bri_tps  |
| unet_hom_best_weight.pth       | unet_hom      |
| deep_watershed_best_weight.pth | dws           |

Map inferencing:

```bat
cd inferencing/
python new_map_inference.py --unseen --cuda --gpu <gpu> --model_type <model type> --model <best model weight .pth file>  --input_map_path <image path .jpg/.png file>
```

For example:
```bat
cd inferencing/
python new_map_inference.py --unseen --cuda --gpu 1 --model_type unet --model ./pretrain_weight/unet_best_weight.pth  --input_map_path ./BHdV_PL_ATL20Ardt_1898_0004-TEST-INPUT_color_border.jpg --vectorization
```

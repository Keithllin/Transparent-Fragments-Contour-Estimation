# Transparent Fragments Contour Estimation via Visual–Tactile Fusion for Autonomous Reassembly

## Automated Batch Generation of Synthetic Dataset in Blender

Blender provides powerful visual simulation capabilities through physically-based rendering (PBR) and material reflection/refraction, producing results that closely mimic real-world interactions between light and materials. 

**Transparent objects**, being a special category, have refractive and transmissive material properties that make their visual features highly sensitive to environmental lighting and background. In real-world scenarios, collecting data of transparent objects with diverse backgrounds and lighting conditions is challenging, and annotations are prone to errors due to difficulties in recognition.

This project offers a Blender script for the **automated generation of synthetic datasets** containing transparent objects. Before each render, the scripts randomize scene elements, including background, lighting and camera angle, greatly enhancing the diversity and richness of the dataset. Using **ID masks**, accurate segmentation masks can be generated. The script supports batch dataset generation with **any scene in which objects are placed at a horizontal plane**. For implementation details, please refer to the [Automated_Batch_Generation_of_Synthetic_Datasets_in_Blender](https://github.com/Keithllin/Transparent-Fragments-Contour-Estimation/tree/main/Automated_Batch_Generation_of_Synthetic_Datasets_in_Blender)
 folder.
### TransFrag27K Dataset

This script was used to create **the first large-scale transparent fragment dataset, TransFrag27K**, which contains **27,000 images and masks** at a resolution of 640×480. 
The dataset covers **fragments of common everyday glassware** and incorporates more than **150 background textures** and **100 HDRI environment lightings**.  

📥 **Download:** The dataset [TransFrag27K on Hugging Face](https://huggingface.co/datasets/chenbr7/TransFrag27K) is available to download. 

## Mask Segmentation and Corner Extraction for Transparent Object Fragments Transparent objects
Robust Mask Segmentation and Corner Extraction for Transparent Object Fragments Transparent objects exhibit strong refraction and transmission, making their visual appearance highly sensitive to lighting and background. Collecting diverse, well-annotated real-world data is difficult, and segmentation of transparent fragments is especially error-prone yet crucial for downstream manipulation tasks.

This project provides a Swin Transformer–based segmentation pipeline that predicts high-quality binary masks from RGB images and derives corner endpoints along dominant edges via post-processing. The system includes careful preprocessing, illumination-robust normalization, optional test-time augmentation (TTA), and brightness/contrast/gamma enhancements to improve performance under challenging lighting. Evaluation metrics (IoU, Dice, pixel accuracy), visualization, and best-checkpoint selection are integrated.

The codebase offers an end-to-end training loop with checkpointing and logging, as well as a batch inference script that outputs masks, corner coordinates, and qualitative overlays. It supports datasets organized as PNG images with corresponding NumPy mask files and is configurable through a single settings module. For implementation details, please refer to train.py, inference.py, and config/settings.py.

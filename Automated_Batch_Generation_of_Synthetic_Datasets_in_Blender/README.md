# Blender-Based Script for Automated Synthetic Dataset Generation

This repository contains a Blender script and example `.blend` files that demonstrate an automated pipeline for batch generation of synthetic datasets.  

The script is compatible with Blender 4.2.3 LTS and does not require additional libraries.  

## Features
- During each rendering iteration, the script produces both **RGB images** and **binary ground-truth masks** using Blender’s compositor ID mask nodes.  
- The scene setup includes:
  - A plane (serving as a tabletop or ground surface, hereafter referred to as "tabletop"),  
  - Multiple target objects,  
  - An HDRI environment texture used as global lighting,  
  - Additional light sources.  

By parameterizing and randomizing scene elements before each render, the script enables fully automated batch dataset generation. This provides a scalable pipeline for synthetic datasets involving objects placed on horizontal surfaces.  

## Randomization Controls
The script automatically configures compositor nodes for mask generation and randomizes the following elements at each render iteration:
- Target object poses,  
- Tabletop position and scale,  
- HDRI lighting intensity,  
- Additional light source positions, size, and intensity,  
- Camera pose.

By manually setting parameter ranges (e.g., object scale, lighting levels), users can quickly generate dataset variants adapted to different scenarios—such as adjusting the proportion of objects within the frame or the overall brightness. Material libraries and HDRI templates allow for fast scene setup without remodeling, while retaining flexibility in dataset customization.  

## Special Consideration: Transparent Objects
Transparent objects pose unique challenges in dataset creation due to their refractive and transmissive properties, which are highly sensitive to lighting and background conditions. Real-world data collection for such cases is both costly and error-prone.  Using this synthetic approach circumvents these limitations, allowing for significantly greater diversity in backgrounds (tabletop materials) and lighting conditions. Overall, this method enables the creation of datasets that better reflect the variability encountered in real applications.

## Example Usage
An example `.blend` file is provided with a pre-configured scene.  
1. Open the file in Blender.  
2. Navigate to **Scripting** in the top menu.  
3. Set the output path, starting index, and number of images to generate
4. Run the script to quickly test batch dataset generation.  

## Limitations
This script does not fully automate dataset creation. Users would manually:  
- Replace object models, tabletop materials, and HDRI textures,  
- Define parameter ranges for randomization based on scene requirements.  
Tabletop material randomization is not included in the provided script because the shader nodes are wired differently across the materials in the built material library. If your material library follows a consistent node setup, you can extend the script to randomize tabletop materials and texture mappings.  

We still recommend familiarity with the following Blender features before use:  
- Object manipulation,  
- Asset library management (for materials, models, HDRIs),  
- HDRI environment textures,  
- Compositor nodes,  
- Material and shader node workflows.  

## Important Notes
- The Blender console does not support `print`. To use a conventional console, go to the top-left menu:  
  *Window → Toggle System Console*.  
- Once the script starts running, Blender’s UI becomes unresponsive until completion. We recommend running small test batches first.  

## Tips
To place irregular objects realistically on the tabletop:  
1. Select the object → *Physics Properties* → enable **Rigid Body** → set type to *Active*.  
2. Select the tabletop → *Physics Properties* → enable **Rigid Body** → set type to *Passive*.  
3. Play the animation to let the object fall naturally.  
4. At a suitable position, select the object → `Ctrl+A` → *Visual Transform*.  
5. Return the timeline to frame 0.  
   > **Note**: Randomization only takes effect at frame 0.  
## TransFrag27K Dataset

This script was used to create **the first large-scale transparent fragment dataset, TransFrag27K**, which contains **27,000 images and masks** at a resolution of 640×480. 
The dataset covers **fragments of common everyday glassware** and incorporates more than **150 background textures** and **100 HDRI environment lightings**.  

📥 **Download:** [TransFrag27K on Hugging Face](https://huggingface.co/datasets/chenbr7/TransFrag27K)  


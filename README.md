# Blended-Diffusion-for-Sound-driven-Editing-of-Natural-Images

**Sound driven Inpainting of Natural Images using Diffusion**

This code combination of [SoundCLIP(CVPR 2022)](https://openaccess.thecvf.com/content/CVPR2022/papers/Lee_Sound-Guided_Semantic_Image_Manipulation_CVPR_2022_paper.pdf) and [Blended-diffusion(CVPR 2022)](https://openaccess.thecvf.com/content/CVPR2022/papers/Avrahami_Blended_Diffusion_for_Text-Driven_Editing_of_Natural_Images_CVPR_2022_paper.pdf) in Pytorch implementation.

Original [soundclip code](https://github.com/kuai-lab/sound-guided-semantic-image-manipulation) and [blended-dissuion code](https://github.com/omriav/blended-diffusion) are provided.

<img src="bird tweeting/output_i_7_b_0.png" width="600px">

# Getting Started
## Installation
1. Create the virtual environment:

```bash
$ conda create --name blended-diffusion python=3.9
$ conda activate blended-diffusion
$ pip3 install timm ftfy regex matplotlib lpips kornia opencv-python torch==1.9.0+cu111 torchvision==0.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
$ pip3 install git+https://github.com/openai/CLIP.git
```

2. Create a `checkpoints` directory and download the pretrained diffusion model from [here](https://drive.google.com/file/d/145NpznbcwMeoX-v8U-bUpu8eXILh3n7Z/view?usp=sharing) to this folder.
3. Create a `pretrained_models` directory and download the pretrained diffusion model from [here](https://download.pytorch.org/models/resnet18-5c106cde.pth) to this folder. (or another [Resnet models](https://pytorch.org/vision/0.8/_modules/torchvision/models/resnet.html))

## Image generation
An example of text-driven multiple synthesis results:

```bash
$ python main.py -a "audio_example/bird tweeting.wav" -i "input_example/butterfly.png" --mask "input_example/butterfly_mask.png" --output_path "output"
```

The generation results will be saved in `output/ranked` folder, ordered by CLIP similarity rank. In order to get the best results, please generate a large number of results (at least 64) and take the best ones.

In order to generate multiple results in a single diffusion process, we utilized batch processing. If you get `CUDA out of memory` try first to lower the batch size by setting `--batch_size 1`.


## Acknowledgments
[SoundCLIP](https://openaccess.thecvf.com/content/CVPR2022/papers/Lee_Sound-Guided_Semantic_Image_Manipulation_CVPR_2022_paper.pdf)

[Blended-diffusion](https://openaccess.thecvf.com/content/CVPR2022/papers/Avrahami_Blended_Diffusion_for_Text-Driven_Editing_of_Natural_Images_CVPR_2022_paper.pdf)

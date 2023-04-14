# **分割万物** Segment Anything

**[Meta AI Research, FAIR](https://ai.facebook.com/research/)**

[Alexander Kirillov](https://alexander-kirillov.github.io/), [Eric Mintun](https://ericmintun.github.io/), [Nikhila Ravi](https://nikhilaravi.com/), [Hanzi Mao](https://hanzimao.me/), Chloe Rolland, Laura Gustafson, [Tete Xiao](https://tetexiao.com), [Spencer Whitehead](https://www.spencerwhitehead.com/), Alex Berg, Wan-Yen Lo, [Piotr Dollar](https://pdollar.github.io/), [Ross Girshick](https://www.rossgirshick.info/)


\[[`纸`](https://ai.facebook.com/research/publications/segment-anything/)]\[[`项目`](https://segment-anything.com/)]\[[`演示`](https://segment-anything.com/demo)]\[[`数据集`](https://segment-anything.com/dataset/index.html)]\[[`博客`](https://ai.facebook.com/blog/segment-anything-foundation-model-image-segmentation/)]\[[`BibTeX`](#citing-segment-anything)]

[![地对空导弹设计](/jumbojing/segment-anything/raw/main/assets/model_diagram.png?raw=true)](/jumbojing/segment-anything/blob/main/assets/model_diagram.png?raw=true)

**分割任何模型 （SAM**） 根据输入提示（如点或框）生成高质量的对象蒙版，并可用于为图像中的所有对象生成蒙版。它已经在 11 万张图像和 1 亿个掩码的[数据集](https://segment-anything.com/dataset/index.html)上进行了训练，并且在各种分割任务上具有强大的零镜头性能。

[![](/jumbojing/segment-anything/raw/main/assets/masks1.png?raw=true)](/jumbojing/segment-anything/blob/main/assets/masks1.png?raw=true) [![](/jumbojing/segment-anything/raw/main/assets/masks2.jpg?raw=true)](/jumbojing/segment-anything/blob/main/assets/masks2.jpg?raw=true)

## [](#installation)安装

代码需要 ，以及 和 。请按照[此处](https://pytorch.org/get-started/locally/)的说明安装 PyTorch 和 TorchVision 依赖项。强烈建议同时安装具有CUDA支持的PyTorch和TorchVision。`python>=3.8``pytorch>=1.7``torchvision>=0.8`

安装分段任何内容：

```
pip install git+https://github.com/facebookresearch/segment-anything.git
```

或在本地克隆存储库并安装

```
git clone git@github.com:facebookresearch/segment-anything.git
cd segment-anything; pip install -e .
```

以下可选依赖项对于掩码后处理、以 COCO 格式保存掩码、示例笔记本以及以 ONNX 格式导出模型是必需的。 还需要运行示例笔记本。`jupyter`

```
pip install opencv-python pycocotools matplotlib onnxruntime onnx
```

## [](#getting-started)[]()开始

首先下载[模型检查点](#model-checkpoints)。然后，只需几行即可使用该模型从给定提示中获取掩码：

```
from segment_anything import SamPredictor, sam_model_registry
sam = sam_model_registry["<model_type>"](checkpoint="<path/to/checkpoint>")
predictor = SamPredictor(sam)
predictor.set_image(<your_image>)
masks, _, _ = predictor.predict(<input_prompts>)
```

或为整个图像生成蒙版：

```
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
sam = sam_model_registry["<model_type>"](checkpoint="<path/to/checkpoint>")
mask_generator = SamAutomaticMaskGenerator(sam)
masks = mask_generator.generate(<your_image>)
```

此外，还可以从命令行为图像生成蒙版：

```
python scripts/amg.py --checkpoint <path/to/checkpoint> --model-type <model_type> --input <image_or_folder> --output <path/to/output>
```

有关更多详细信息，请参阅有关[使用带有提示的 SAM 和](/jumbojing/segment-anything/blob/main/notebooks/predictor_example.ipynb)[自动生成掩码](/jumbojing/segment-anything/blob/main/notebooks/automatic_mask_generator_example.ipynb)的示例笔记本。

[![](/jumbojing/segment-anything/raw/main/assets/notebook1.png?raw=true)](/jumbojing/segment-anything/blob/main/assets/notebook1.png?raw=true) [![](/jumbojing/segment-anything/raw/main/assets/notebook2.png?raw=true)](/jumbojing/segment-anything/blob/main/assets/notebook2.png?raw=true)

## [](#onnx-export)ONNX 导出

SAM 的轻量级掩码解码器可以导出为 ONNX 格式，以便它可以在任何支持 ONNX 运行时的环境中运行，例如[演示](https://segment-anything.com/demo)中显示的浏览器内。导出模型

```
python scripts/export_onnx_model.py --checkpoint <path/to/checkpoint> --model-type <model_type> --output <path/to/output>
```

有关如何将通过 SAM 主干进行的图像预处理与使用 ONNX 模型的掩模预测相结合的详细信息，请参阅[示例笔记本](https://github.com/facebookresearch/segment-anything/blob/main/notebooks/onnx_model_example.ipynb)。建议使用最新的稳定版本的 PyTorch 进行 ONNX 导出。

## [](#model-checkpoints)[]()模型检查点

该模型的三个模型版本具有不同的主干大小。这些模型可以通过运行

```
from segment_anything import sam_model_registry
sam = sam_model_registry["<model_type>"](checkpoint="<path/to/checkpoint>")
```

单击下面的链接下载相应模型类型的检查点。

* **`默认值`或`vit_h`：[ViT-H SAM 模型。](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth)**
* `vit_l`：[ViT-L SAM 模型。](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth)
* `vit_b`：[ViT-B SAM模型。](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth)

## [](#dataset)数据

有关数据配置的概述，请参阅[此处](https://ai.facebook.com/datasets/segment-anything/)。数据集可[在此处](https://ai.facebook.com/datasets/segment-anything-downloads/)下载。下载数据集即表示您同意您已阅读并接受 SA-1B 数据集研究许可的条款。

我们将每个图像的遮罩保存为 json 文件。它可以以以下格式加载为 python 中的字典。

```
{
    "image"                 : image_info,
    "annotations"           : [annotation],
}

image_info {
    "image_id"              : int,              # Image id
    "width"                 : int,              # Image width
    "height"                : int,              # Image height
    "file_name"             : str,              # Image filename
}

annotation {
    "id"                    : int,              # Annotation id
    "segmentation"          : dict,             # Mask saved in COCO RLE format.
    "bbox"                  : [x, y, w, h],     # The box around the mask, in XYWH format
    "area"                  : int,              # The area in pixels of the mask
    "predicted_iou"         : float,            # The model's own prediction of the mask's quality
    "stability_score"       : float,            # A measure of the mask's quality
    "crop_box"              : [x, y, w, h],     # The crop of the image used to generate the mask, in XYWH format
    "point_coords"          : [[x, y]],         # The point coordinates input to the model to generate the mask
}
```

图像ID可以在sa\_images\_ids.txt中找到，也可以使用上面的[链接](https://ai.facebook.com/datasets/segment-anything-downloads/)下载。

要将 COCO RLE 格式的掩码解码为二进制：

```
from pycocotools import mask as mask_utils
mask = mask_utils.decode(annotation["segmentation"])
```

有关操作以 RLE 格式存储的掩码的更多说明，请参阅[此处](https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/mask.py)。

## License
The model is licensed under the [Apache 2.0 license](LICENSE).

## Contributing

See [contributing](CONTRIBUTING.md) and the [code of conduct](CODE_OF_CONDUCT.md).

## Contributors

The Segment Anything project was made possible with the help of many contributors (alphabetical):

Aaron Adcock, Vaibhav Aggarwal, Morteza Behrooz, Cheng-Yang Fu, Ashley Gabriel, Ahuva Goldstand, Allen Goodman, Sumanth Gurram, Jiabo Hu, Somya Jain, Devansh Kukreja, Robert Kuo, Joshua Lane, Yanghao Li, Lilian Luong, Jitendra Malik, Mallika Malhotra, William Ngan, Omkar Parkhi, Nikhil Raina, Dirk Rowe, Neil Sejoor, Vanessa Stark, Bala Varadarajan, Bram Wasti, Zachary Winstrom

## Citing Segment Anything

If you use SAM or SA-1B in your research, please use the following BibTeX entry. 

```
@article{kirillov2023segany,
  title={Segment Anything}, 
  author={Kirillov, Alexander and Mintun, Eric and Ravi, Nikhila and Mao, Hanzi and Rolland, Chloe and Gustafson, Laura and Xiao, Tete and Whitehead, Spencer and Berg, Alexander C. and Lo, Wan-Yen and Doll{\'a}r, Piotr and Girshick, Ross},
  journal={arXiv:2304.02643},
  year={2023}
}
```

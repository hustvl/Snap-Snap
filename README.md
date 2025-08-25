# Snap-Snap: Taking Two Images to Reconstruct 3D Human Gaussians in Milliseconds
### [Project Page](https://hustvl.github.io/Snap-Snap/) | [arxiv Paper](https://arxiv.org/abs/2508.14892)

[Snap-Snap: Taking Two Images to Reconstruct 3D Human Gaussians in Milliseconds](https://hustvl.github.io/Snap-Snap/)  

Jia Lu<sup>1*</sup>, [Taoran Yi](https://github.com/taoranyi)<sup>1*</sup>, [Jiemin Fang](https://jaminfong.cn/)<sup>2✉</sup>, [Chen Yang](https://scholar.google.com/citations?hl=zh-CN&user=StdXTR8AAAAJ)<sup>3</sup>, Chuiyun Wu<sup>1</sup>, [Wei Shen](https://shenwei1231.github.io/)<sup>3</sup>, [Wenyu Liu](http://eic.hust.edu.cn/professor/liuwenyu/)<sup>1</sup>, [Qi Tian](https://www.qitian1987.com/)<sup>2</sup> , [Xinggang Wang](https://xwcv.github.io/)<sup>1✉</sup>


<sup>1</sup>Huazhong University of Science and Technology &emsp;<sup>2</sup>Huawei Inc. &emsp; <sup>3</sup>Shanghai Jiaotong University &emsp; 

<sup>*</sup>Equal contribution (during internship at Huawei Inc.)  <sup>✉</sup>Corresponding authors


Reconstructing 3D human bodies from sparse views has been an appealing topic, which is crucial to broader the related applications. In this paper, we propose a quite challenging but valuable task to reconstruct the human body from only two images, i.e., the front and back view, which can largely lower the barrier for users to create their own 3D digital humans. The main challenges lie in the difficulty of building 3D consistency and recovering missing information from the highly sparse input. We redesign a geometry reconstruction model based on foundation reconstruction models to predict consistent point clouds even input images have scarce overlaps with extensive human data training. Furthermore, an enhancement algorithm is applied to supplement the missing color information, and then the complete human point clouds with colors can be obtained, which are directly transformed into 3D Gaussians for better rendering quality. Experiments show that our method can reconstruct the entire human in 190 ms on a single NVIDIA RTX 4090, with two images at a resolution of 1024$\times$1024, demonstrating state-of-the-art performance on the THuman2.0 and cross-domain datasets. Additionally, our method can complete human reconstruction even with images captured by low-cost mobile devices, reducing the requirements for data collection.

## Updates
- 8/21/2025: The rough code has been released, and there may still be some issues. Please feel free to raise issues. 


## Installation
To run Snap-Snap, you can use the following scripts, following [GPS-Gaussian](https://github.com/aipixel/GPS-Gaussian):
```
conda env create --file environment.yml
conda activate snapsnap
```
Then, compile the ```diff-gaussian-rasterization``` in [3DGS](https://github.com/graphdeco-inria/gaussian-splatting) repository:
```
git clone https://github.com/graphdeco-inria/gaussian-splatting --recursive
cd gaussian-splatting/
pip install -e submodules/diff-gaussian-rasterization
cd ..
```

## Run on synthetic human dataset

**Data Preparation**

We use the [Thuman2.0](https://github.com/ytrock/THuman2.0-Dataset), [Thuman2.1](https://github.com/ytrock/THuman2.0-Dataset), [2K2K](https://github.com/SangHunHan92/2K2K), [4D-Dress](https://github.com/eth-ait/4d-dress) for training or evaluation. Thanks to the dataset creators for their efforts.

To preprocess the human data, you can use the scripts in the `prepare_data` directory.  The detailed procedure can be found in [GPS-Gaussian](https://github.com/aipixel/GPS-Gaussian).


**Testing**

To construct human reconstruction from two images, you can use the command below. As for the evaluation, we following the calculation method of [GHG](https://github.com/humansensinglab/Generalizable-Human-Gaussians). The weights are provided in [google dirve](https://drive.google.com/file/d/1YMoOQEP_fa0yv8B2gGvjB0Mv-iSkIeaA/view?usp=sharing).
```
python test_view_interp.py

# evaluation
python compute_metrics.py
```

## 📑 Citation
If you find this repository/work helpful in your research, welcome to cite the paper and give a ⭐.
Some source code of ours is borrowed from [GPS-Gaussian](https://github.com/aipixel/GPS-Gaussian), [MASt3R](https://github.com/naver/mast3r), [GHG](https://github.com/humansensinglab/Generalizable-Human-Gaussians).We sincerely appreciate the excellent works of these authors.
```
@article{snapsnap,
        title={Snap-Snap: Taking Two Images to Reconstruct 3D Human Gaussians in Milliseconds}, 
        author={Jia Lu and Taoran Yi and Jiemin Fang and Chen Yang and Chuiyun Wu and Wei Shen and Wenyu Liu and Qi Tian and Xinggang Wang},
        journal={arxiv:2508.14892},
        year={2025}
        }
```

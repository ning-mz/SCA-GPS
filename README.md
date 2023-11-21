# SCA-GPS


## Introduction
Code for paper [A Symbolic Characters Aware Model for Solving Geometry Problems](https://dl.acm.org/doi/10.1145/3581783.3612570) - _ACM MM 2023_


## Run

### Environment
transformers==4.17.0

allennlp==0.9.0

### Refined GeoQA Dataset
Please note, we refined the GeoQA dataset to remove the Alpha Chanel in the geometry diagrams to satisfy the requirement of ViT input. The refined dataset named as GeoQA-Pro and used in this repo.

### Prepare Robert-CHN
Due to LFS space limited, please download robert-chn from https://huggingface.co/hfl/chinese-roberta-wwm-ext and move the pytorch_model.bin to the roberta folder in this repo.

### Training    
    allennlp train config/DPE.json --include-package DPE -s test/

### Evaluating
    allennlp evaluate test/  GeoQA-Data/Geo-Pro/pro_test.pk --include-package DPE-test --cuda-device 0


## Citation

If the paper or the code helps you, please cite the paper in the following format :

```
@inproceedings{ning2023SCAGPS, 
    author = {Ning, Maizhen and Wang, Qiu-Feng and Huang, Kaizhu and Huang, Xiaowei}, 
    title = {A Symbolic Characters Aware Model for Solving Geometry Problems}, 
    year = {2023}, 
    doi = {10.1145/3581783.3612570}, 
    booktitle = {Proceedings of the 31st ACM International Conference on Multimedia}, 
    series = {MM '23} 
}
```

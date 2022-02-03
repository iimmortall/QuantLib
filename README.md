<center> QuantLib </center>

## Introduction
QuantLib is an open source quantization toolbox based on PyTorch. 

## Supported methods
- [x] [LSQ (ICLR'2019)](configs/lsq) [LEARNED STEP SIZE QUANTIZATION](https://arxiv.org/abs/1902.08153)
- [x] [LSQ+ (CVPR'2020)](configs/lsq) [LSQ+: Improving low-bit quantization through learnable offsets and better initialization](https://arxiv.org/pdf/2004.09576.pdf)
- [x] [DAQ (ICCV'2021)](configs/daq) [Distance-aware Quantization](https://arxiv.org/abs/1902.08153)

## Getting Started
### Dependencies
* Python == 3.7
* PyTorch == 1.8.2

### Installation
* Clone github repository.
```bash
$ git clone git@github.com:iimmortall/QuantLib.git
```
* Install dependencies
```bash
$ pip install -r requirements.txt
```


### Datasets
* Cifar-10
    * This can be automatically downloaded by learning our code, you can config the save path in the '*.yaml' file.
* ImageNet
    * This is available at [here](http://www.image-net.org) 

### Training & Evaluation
Cifar-10 dataset (ResNet-20 architecture) 

* First, download full-precision model into your folder(you can config the model path in your *.yaml file). **Link: [[weights](https://drive.google.com/file/d/1II9jtowxaGYde8_rYLs-qnPwzVcB3QYZ/view?usp=sharing)]**

```bash
# DAQ: Cifar-10 & ResNet-20 W1A1 model
$ python run.py --config configs/daq/resnet20_daq_W1A1.yml
# DAQ Cifar-10 & ResNet-20 W1A32 model
$ python run.py --config configs/daq/resnet20_daq_W1A32.yml
# LSQ: Cifar-10 & ResNet-20 W8A8 model
$ python run.py --config configs/lsq/resnet20_lsq_W8A8.yml
# LSQ+: Cifar-10 & ResNet-20 W8A8 model
$ python run.py --config configs/lsq_plus/resnet20_lsq_plus_W8A8.yml
```

## Results 
#### **Note**
* Weight quantization: Signed, Symmetric, Per-tensor. [Why use symmetric quantization.](https://www.qualcomm.com/media/documents/files/presentation-enabling-power-efficient-ai-through-quantization.pdf)
* Activation quantization: Unsigned, Asymmetric.
* Don't quantize the first and last layer. 

| methods | Weight | Activation | Accuracy | Models
| ------ | --------- | ------ | ------ | ------ |
| float | - | - | 91.4 | [download]() |
| LSQ | 8 | 8 | 91.9 | [download]() |
| LSQ+ | 8 | 8 | 92.1 | [download]() |
| DAQ | 1 | 1 | 85.8 | [download](https://drive.google.com/file/d/1zq8zZO_YnrLkMPybzZLJEBuSg66eFV4g/view) |
| DAQ | 1 | 32 | 91.2 | [download](https://drive.google.com/file/d/1SKHmms5kRLF_nLHf0qPbEO0JUOr34O5a/view?usp=sharing) |

## Acknowledgement

QuantLib is an open source project. We appreciate all the contributors who implement their methods or add new features, as well as users who give valuable feedbacks.

## License

This project is released under the [MIT license](LICENSE).

---
### References
* https://github.com/ZouJiu1/LSQplus
* https://github.com/cvlab-yonsei/DAQ
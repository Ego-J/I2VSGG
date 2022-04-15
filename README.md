# A Pytorch Implementation of [Adaptive Image-to-video Scene Graph Generation via Knowledge Reasoning and Adversarial Learning](https://wuxinxiao.github.io/assets/papers/2022/I2VSGG.pdf) (AAAI 2022) 

## Introduction
Follow [faster-rcnn repository](https://github.com/jwyang/faster-rcnn.pytorch) to setup the environment. We used Pytorch 0.4.1 and Python 3.6 for this project.

Download related data of VRD->VidVRD [here](https://hxo562ksks.larksuite.com/drive/folder/fldusROra26WnXsRVOKqY1opZce).

### An example of VRD->VidVRD
#### Train and test the detection model
```
bash ./scripts/instance_styleD_resnet101.sh
```

#### Train and test the SGG model
```
bash ./scripts/SGG_emb_resnet.sh
```

### Citation
```
@article{chen2022image,
  title={Adaptive Image-to-video Scene Graph Generation via Knowledge Reasoning and Adversarial Learning},
  author={Jin Chen, Xiaofeng Ji, Xinxiao Wu,
  booktitle={The Thirty-Sixth {AAAI} Conference on Artificial Intelligence, {AAAI} 2022},
  year={2022},
}
```


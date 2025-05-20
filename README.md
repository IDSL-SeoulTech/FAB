# FAB: FPGA-Accelerated Fully-Pipelined Bottleneck Architecture with Batching for High-Performance MobileNetv2 Inference
###### This repository contains PyTorch, C reference code, some FPGA constraints for FAB project.

## Pretrained & C Reference Data
You can download the pretrained weights and the extracted data from the following URL.

https://drive.google.com/drive/folders/1cTyFCJMDTP-DIKNH75waVvRhpdTzzI-m?usp=sharing


---
## Pytorch Evaluation 
###### Modify the dataset root directly in the config
###### If the train dataset root is not set, the ImageNet task cannot be recognized, resulting in an FC layer size mismatch error.
```bash
python main_eval.py --common.config-file {config_url} --model.classification.pretrained ./base_weight/mobilenetv2-1.00.pt
```
#### Configuration File 
###### W Quant
```bash
./config/classification/imagenet/mobilenetv2_ptq.yaml
```
###### W/O Quant
```bash
./config/classification/imagenet/mobilenetv2.yaml
```

#### Quantization configuration Detail
```bash
quant.quant : Determining whether to perform quantization
quant.quant_method : meaningless argument
quant.weight_bit : Weight quantization bit
quant.activation_bit : Activation quantization bit
quant.calibration_a : Layer wise calibration
quant.calibration_w : Layer wise calibration
quant.calibration_c : Channel wise calibration
quant.calib_iter : Calibration iter 
```

#### Model Detail
##### Overall Model file
###### Build The MobileNetV2 Block
```
/cvnets/modules/mobilenetv2.py
```
#### Quantization Module
###### Build The Quantization Module
```
/cvnets/ptq
```

## C Reference
### Overall code explaination
```
main.c :
qact.c :
bottleneck.c :
quantizer.c :
conv.c :
utils.c :
fc.c:
```
## FPGA Constraints
###### Will be updated




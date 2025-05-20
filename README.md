# MobileNetv2 Quantization
This repository contains PyTorch, C reference code for FAB project.

## Evaluation 
###### Modify the dataset root directly in the config (e.g., --dataset.root_train and --dataset.root_val)
###### If the train dataset root is not set, the ImageNet task cannot be recognized, resulting in an FC layer size mismatch error.
```bash
python main_eval.py --common.config-file {config_url} --model.classification.pretrained ./base_weight/mobilenetv2-1.00.pt
```
## Pretrained & C Reference Data
```bash
    https://drive.google.com/drive/folders/1cTyFCJMDTP-DIKNH75waVvRhpdTzzI-m?usp=sharing
```


## Configuration File 
###### W Quant
```bash
./config/classification/imagenet/mobilenetv2_ptq.yaml
```
###### W/O Quant
```bash
./config/classification/imagenet/mobilenetv2.yaml
```

## Quantization configuration Detail
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

## Model Detail
### Overall Model file
###### Set calibration , Quantization Mode And Build the Overall Model
```
/cvnets/models/classification/mobilenetv2.py
```
### Overall Model forward file
```
/cvnets/models/classification/base_image_encoder.py
```
### MobileNet V2 Block
###### Build The MobileNetV2 Block
```
/cvnets/modules/mobilenetv2.py
```
### Quantization Module
###### Build The Quantization Module
```
/cvnets/ptq
```

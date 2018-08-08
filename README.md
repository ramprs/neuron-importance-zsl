

## NIWT: Neuron-Importance aware Weight Transfer

Code for the ECCV'18 paper

**[Choose-Your-Neuron: Incorporating Domain Knowledge into Deep Networks through Neuron-Importance]**  
Ramprasaath R. Selvaraju, Prithvijit Chattopadhyay, Mohammed Elhoseiny, Tilak Sharma, Dhruv Batra, Devi Parikh, Stefan Lee


![Overview](https://i.imgur.com/GVZVGs4.png)

### Usage

This codebase assumes that you have installed Tensorflow. If not, please follow installation instructions from [here](https://www.tensorflow.org/install/).  
Download data and pretrained checkpoints using `sh download.sh` and make sure the paths in the arg_config json files are correct.
You may also need to create an `imagenet_files.pkl` which contains a list of (atleast) 3000 randomly sampled imagenet image paths. 

#### Train a Generalized Zero Shot Learning model on AWA2 and CUB (class-level attributes)

```
python alpha2w.py --config_json arg_configs/vgg16_config_AWA.json
python alpha2w.py --config_json arg_configs/resnet_config_AWA.json
python alpha2w.py --config_json arg_configs/vgg16_config_CUB.json
python alpha2w.py --config_json arg_configs/resnet_config_CUB.json
```

#### Train a Generalized Zero Shot Learning model on CUB with captions (class-level)

```
python alpha2w.py --config_json arg_configs/vgg16_config_CUB_captions.json
python alpha2w.py --config_json arg_configs/resnet_config_CUB_captions.json
```


### Train a GZSL classifier from scratch

#### Pretrain base model on dataset
To do this, we first finetune the base model (vgg16 or resnet_v1) on a seen class images.

```
cd seen_pretraining/
sh cnn_finetune.sh 
```

#### Extract Neuron Importances (alphas)

Change the `ckpt_path` from  the config_json files to the trained checkpoint (obtained from above)
Extract Neuron-Importances (alphas) from the finetuned model.

```
sh alpha_extraction.sh
```

#### Domain knowledge to Neuron Importance:

Here we learn a transformation from domain knowledge (say attributes) to network neuron importances (alphas)
```
cd ..
python mod2alpha.py --config_json arg_configs/vgg16_config_AWA.json
python mod2alpha.py --config_json arg_configs/resnet_config_AWA.json
python mod2alpha.py --config_json arg_configs/vgg16_config_CUB.json
python mod2alpha.py --config_json arg_configs/resnet_config_CUB.json
```

#### Neuron Importance of unseen classes to classifier weights of unseen classes (training a GZSL model)
```
python alpha2w.py --config_json arg_configs/vgg16_config_AWA.json
python alpha2w.py --config_json arg_configs/resnet_config_AWA.json
python alpha2w.py --config_json arg_configs/vgg16_config_CUB.json
python alpha2w.py --config_json arg_configs/resnet_config_CUB.json
```

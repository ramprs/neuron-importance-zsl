python -c 'import alpha_extraction as ae; ae.get_alphas("arg_configs/alpha_extract/AWA2_vgg16_finetuned_train_cnn_trainval_alpha_extract_config.json")'
python -c 'import alpha_extraction as ae; ae.get_alphas("arg_configs/alpha_extract/AWA2_resnet_v1_101_finetuned_train_cnn_trainval_alpha_extract_config.json")'

python3 -c 'import alpha_extraction as ae; ae.get_alphas("arg_configs/alpha_extract/CUB_vgg16_finetuned_train_cnn_trainval_alpha_extract_config.json")'
python3 -c 'import alpha_extraction as ae; ae.get_alphas("arg_configs/alpha_extract/CUB_resnet_finetuned_train_cnn_trainval_alpha_extract_config.json")'

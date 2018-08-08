# val finetune with hyper-parameter search
python -c 'import cnn_finetune as cf; cf.run_training_no_search("arg_configs/seen_finetune/CUB_vgg_16_seen_proposed_finetune.json")' | tee train_logs/seen_train_CUB_vgg_16_seen_proposed_finetune.txt
python -c 'import cnn_finetune as cf; cf.run_training_no_search("arg_configs/seen_finetune/AWA2_vgg_16_seen_proposed_finetune.json")' | tee train_logs/seen_train_AWA2_vgg_16_seen_proposed_finetune.txt
python -c 'import cnn_finetune as cf; cf.validate_and_train("arg_configs/seen_finetune/CUB_resnet_v1_101_seen_proposed_finetune_hsearch.json")' | tee train_logs/seen_train_CUB_resnet_v1_101_seen_proposed_finetune_hsearch.txt
python -c 'import cnn_finetune as cf; cf.validate_and_train("arg_configs/seen_finetune/AWA2_resnet_v1_101_seen_proposed_finetune_hsearch.json")' | tee train_logs/seen_train_AWA2_resnet_v1_101_seen_proposed_finetune_hsearch.txt


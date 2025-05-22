export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

python gaze_shift.py --train_batch_size 30 --val_batch_size 30 --experiment ./experiments/gaze_shift --run gaze_shift_on_vrgaze --lr 0.0001 --weight_decay 0.05 --max_epochs 100 --clip-grad 2.0 --device 1 --accelerator gpu --dataset_type ts  --dataset_path /stage/algo-datasets/DB/GazeEstimation/ThundersoftVRSet/VRGaze --wandb --wandb_entity gil-shapira --log_every_n_steps 1 --num_workers 8 --limit_val_batches 1.0 --limit_train_batches 1.0 --input_width 400 --input_height 400 --backbone_model_name mobilenetv2_0x0 --backbone_feature_index 4  --strategy auto --precision 32 --channels 1 --backbone_feature_size 128 --gaze_format yaw_pitch --check_val_every_n_epoch 1 --poly_calib_degree_train 1 --poly_calib_degree_val 4 --validation_fit_size 50 --binocular_mode --train_eval_mode --width_multiplier 2.0 --unsupervised --val_check_interval 2000 --batch_per_person




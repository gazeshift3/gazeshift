



def get_args(parser):

    #An experiment has one to one relationship with a git commit.
    #A run is an experiment with a specific set of parameters.
    #../experiments/exp_1/run_1/run_config.sh
    #Holds the run parameters of run_1 of exp_1

    #Trainer params:
    parser.add_argument('--run', default='', help='run name')
    #Each experiment is related to a specific model. Separate run differ in hyper-params
    parser.add_argument('--experiment', default='', help='experiment path')
    parser.add_argument('--accelerator', default='auto', help='auto, cpu or gpu')
    parser.add_argument('--devices', default=1, type=int, help='cpu or gpu')
    parser.add_argument('--gaze_dim', default=5, type=int, help='cpu or gpu')
    parser.add_argument('--strategy', default='ddp_find_unused_parameters_false', help='training strategy')
    parser.add_argument('--sync_batchnorm', action='store_true', help='sync bacthnorm')
    parser.add_argument('--ckpt_path', default='', help='path to checkpoint pth file')
    parser.add_argument('--wandb_project', default='', help='wandb project')

    parser.add_argument('--validation_ids', nargs='+', help='List of items')
    parser.add_argument('--vae_path', default='', help='')
    parser.add_argument('--wandb_proj', default='', help='wandb project name')
    parser.add_argument('--precision', default=32, help='16, 32, 64, bf16')
    parser.add_argument('--fast_dev_run', action='store_true', default=False, help='unsupervised pretraining')
    parser.add_argument('--val_check_interval', default=1.0, type=float, help='when to launch validation fraction*full_epoch')
    parser.add_argument('--cvm_vae_negative_margin', default=0.2, type=float, help='when to launch validation fraction*full_epoch')
    parser.add_argument('--log_every_n_steps', type=int, default=50, help='')
    parser.add_argument('--limit_train_batches', default=1.0, type=float, help='How much of training dataset to check')
    parser.add_argument('--triplet_loss_margin', default=0.5, type=float, help='How much of training dataset to check')
    parser.add_argument('--residual_loss_weight', default=0.7, type=float, help='How much of training dataset to check')
    parser.add_argument('--kld_weight', default=0.0, type=float, help='How much of training dataset to check')
    parser.add_argument('--same_eye_weight', default=0.5, type=float, help='How much of training dataset to check')
    parser.add_argument('--limit_val_batches', default=1.0, type=float, help='How much of training dataset to check')
    parser.add_argument('--num_sanity_val_steps', type=int, default=0, help='Sanity check runs n validation batches before starting the training routine')
    parser.add_argument('--check_val_every_n_epoch', type=int, default=1,
                        help='check_val_every_n_epoch')

    parser.add_argument('--vae_decoder_initial_channel_dim', type=int, default=256,
                        help='check_val_every_n_epoch')


    parser.add_argument('--margin', default=0.5, type=float,
                        help='Trieplet loss margin')
    parser.add_argument('--alpha', default=0.5, type=float,
                        help='Trieplet loss margin')


    parser.add_argument('--poly_calib_degree_val', type=int, default=1,
                        help='Plynomail degre features for calibration')

    parser.add_argument('--poly_calib_degree_train', type=int, default=1,
                        help='Plynomail degre features for calibration')

    parser.add_argument('--validation_fit_size', type=int, default=10,
                        help='Default size for fit caclulation in validation')


    parser.add_argument('--backbone_feature_size', type=int, default=64,
                        help='Plynomail degree features for calibration')

    parser.add_argument('--training_set_size', type=int, default=500,
                        help='per person training set size')

    parser.add_argument('--gradient_clip_algorithm', default='norm', help='')
    parser.add_argument('--gradient_clip_val', default='10', help='')
    parser.add_argument('--accumulate_grad_batches', type=int, default=1, help='Accumulates grads every k batches')
    parser.add_argument('--max_epochs', type=int, default=100, help='')
    parser.add_argument('--num_workers', type=int, default=0, help='num of threads for dataloaders')

    parser.add_argument('--input_width', type=int, default=640, help='')
    parser.add_argument('--input_height', type=int, default=480, help='')

    #Dataset
    parser.add_argument('--person_id_calib', default=None, type=int)
    parser.add_argument('--person_ids_calib', nargs="+", type=int)
    parser.add_argument('--calib_mat_size', default=4, type=int)

    parser.add_argument('--oracle_type', default='VAE',
                        choices=['VAE', 'static_labels', 'fixed'])
    parser.add_argument('--dataset_type', default='openeds_2020', choices=['nvgaze_real_ar', 'nvgaze_real_vr', 'openeds_2020','ts','columbia', 'mpiigaze'])
    parser.add_argument('--eye_type', default='L',
                        choices=['L', 'R'])
    parser.add_argument('--gaze_format', default='3d_norm_dir',
                        choices=['3d_norm_dir', 'yaw_pitch'])

    parser.add_argument('--calib_sampling_mathod', default='FPS',
                        choices=['FPS', 'random'])
    parser.add_argument('--dataset_path', type=str, nargs="+", default=['/stage/algo-datasets/DB/GazeEstimation/NVGAZE/nvgaze_real_dataset_ar'])

    parser.add_argument('--seed', default=42, type=int)

    parser.add_argument('--train_batch_size', type=int, default=32, metavar='N',
                        help='input batch size for training (default: 32)')
    parser.add_argument('--val_batch_size', type=int, default=32, metavar='N',
                        help='input batch size for training (default: 32)')

    parser.add_argument('--clip-grad', type=float, default=10.0, metavar='NORM',
                        help='Clip gradient norm (default: 10.0)')

    # parser.add_argument('--depth_multiplier', type=float, default=1.0, help='depth multiplier (default: 1.0)')
    parser.add_argument('--width_multiplier', type=float, default=1.0, help='width multiplier (default: 1.0)')

    parser.add_argument('--clip-mode', type=str, default='norm',
                        help='Gradient clipping mode. One of ("norm", "value", "agc")')

    parser.add_argument('--backbone_model_name', type=str, default='efficientnet_lite0',
                        help='see timm.list_models() for available models')

    parser.add_argument('--backbone_feature_index', type=int, default=4,
                        help='The index of the feature map to extract')

    parser.add_argument('--train_eval_mode', action='store_true', default=False, help='validation only')
    parser.add_argument('--validate', action='store_true', default=False,help='validation only')
    parser.add_argument('--unsupervised', action='store_true', default=False,help='unsupervised pretraining')
    parser.add_argument('--embedding_loss', action='store_true', default=False, help='embedding_loss')
    parser.add_argument('--only_stationary_train_point', action='store_true', default=False, help='validation only')
    parser.add_argument('--only_stationary_validation_point', action='store_true', default=False, help='validation only')
    parser.add_argument('--batch_per_person', action='store_true', default=False, help='validation only')
    parser.add_argument('--train_inline_calib', action='store_true', default=False, help='validation only')
    parser.add_argument('--predict', action='store_true', default=False,help='prediction only')

    parser.add_argument('--binocular_mode', action='store_true', default=False, help='validation only')
    # Model Exponential Moving Average
    parser.add_argument('--model-ema', action='store_true', default=False,
                        help='Enable tracking moving average of model weights')
    parser.add_argument('--model-ema-decay', type=float, default=0.9998,
                        help='decay factor for model weights moving average (default: 0.9998)')

    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Learning rate')
    # parser.add_argument('--channel_mult', type=float, default=0.5,
    #                     help='Learning rate')

    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='Learning rate')

    parser.add_argument('--drop_rate', type=float, default=0.2,
                        help='drop rate')

    parser.add_argument('--drop_path_rate', type=float, default=0.2,
                        help='drop path rate')

    parser.add_argument('--lr_milestones', default='25', type=str, help='epochs for reducing LR')
    parser.add_argument('--input_dir', default='./', type=str, help='input directory for prediction')
    parser.add_argument('--lr_gamma', default=0.1, type=float, help='multiply when reducing LR')

    parser.add_argument('--wandb', action='store_true', help='wandb logging')
    parser.add_argument('--dump_output', action='store_true', help='wandb logging')
    parser.add_argument('--wandb_entity', help='wandb entity')
    parser.add_argument('--merge_test_train', action='store_true', help='merge train and test')

    parser.add_argument('--channels', default=1, type=int, help='number of channels in the input image')
    parser.add_argument('--feature_width', default=100, type=int, help='number of channels in the feature embedding')
    args = parser.parse_args()

    return args

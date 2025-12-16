import sys
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from utils.functions import read_config
from utils.collate import collate_Yolo

COLLATE_FUNCTIONS = {'2DObjDet': collate_Yolo}


def load_datasetloader(args, dtype, world_size, rank, mode='train'):
    '''
    Load dataset and create PyTorch DataLoader with optional DDP support.

    Args:
        args: Argument parser containing dataset/model configuration.
        dtype: PyTorch tensor dtype (e.g., torch.FloatTensor).
        world_size: Number of processes for distributed training.
        rank: Process rank for distributed training.
        mode: 'train', 'val', 'valid', or 'test'.

    Returns:
        For train/val mode:
            (train_dataset, val_dataset), (train_loader, val_loader), (train_sampler, val_sampler)
        For test mode:
            dataset, loader, None
    '''

    # Make sure your model name and app mode are defined in './config/config.json'
    # If the name and mode defined in argumentparser.py don't match those in the json file, it stops working.
    cfg = read_config()
    SUPPORTED_MODELS = cfg['supported_models']
    if args.model_name not in SUPPORTED_MODELS:
        sys.exit(f'[Error] nuScenes has no loaders for {args.model_name}!')
    if args.app_mode not in cfg['supported_app_modes']:
        sys.exit(f'[Error] {args.app_mode} mode is not supported!')


    # TODO : modify this part based on your implementation
    # Validate dataset and model
    if args.dataset_type == 'nuscenes':
        if args.model_name == 'Yolo':
            from dataset.NuscenesDataset.loader_2d_obj_det import DatasetLoader
        else:
            sys.exit(f'[Error] {args.model_name} is not supported for {args.dataset_type}!')
    else:
        sys.exit(f'[Error] {args.dataset_type} is not supported!')
    

    # Test mode
    collate_fn = COLLATE_FUNCTIONS[args.app_mode]
    if mode not in ['train', 'val', 'valid']:
        dataset = DatasetLoader(args=args, dtype=dtype, world_size=1, rank=0, mode='test')
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_cores, drop_last=False, collate_fn=collate_fn)
        return dataset, loader, None

    # Train/Val mode
    train_dataset = DatasetLoader(args=args, dtype=dtype, world_size=world_size if args.ddp else 1,
                                  rank=rank if args.ddp else 0, mode='train')
    val_dataset = DatasetLoader(args=args, dtype=dtype, world_size=world_size if args.ddp else 1,
                                rank=rank if args.ddp else 0, mode='val')

    if args.ddp:
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False,
                                  num_workers=args.num_cores, pin_memory=True,
                                  sampler=train_sampler, collate_fn=collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                num_workers=args.num_cores, pin_memory=True,
                                sampler=val_sampler, collate_fn=collate_fn)
    else:
        train_sampler, val_sampler = None, None
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.num_cores, drop_last=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                num_workers=args.num_cores, drop_last=True, collate_fn=collate_fn)

    return (train_dataset, val_dataset), (train_loader, val_loader), (train_sampler, val_sampler)


def load_solvers(args, num_train_scenes, logger, dtype, world_size=None, rank=None, isTrain=True):
    '''
    Load the appropriate solver based on the model name.

    Args:
        args: Argument parser containing model configuration.
        num_train_scenes: Number of training scenes.
        logger: Logger object for logging.
        dtype: PyTorch tensor dtype (e.g., torch.FloatTensor).
        world_size: Number of processes for distributed training.
    '''
    
    if args.model_name == 'Yolo':
        from optimization.Yolo_solver import Solver
        return Solver(args, num_train_scenes, world_size, rank, logger, dtype, isTrain)
    sys.exit(f'[Error] No solver available for {args.model_name}!')

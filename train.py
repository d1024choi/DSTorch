'''
DSTorch : Pytorch-based DNN training platform supporting DDP

Author: Dooseop Choi
Date: 2025-12-03
e-mail: d1024.choi@etri.re.kr
'''

import os
import sys
import warnings

# Suppress all warnings - must be done before other imports
warnings.filterwarnings('ignore')
warnings.simplefilter("ignore")
os.environ['PYTHONWARNINGS'] = 'ignore'
os.environ['TORCH_WARN'] = '0'  # Suppress PyTorch warnings

import time
import logging
import traceback
import torch
import torch.distributed as dist

from helper import load_datasetloader, load_solvers


def main(args):

    # logging setting
    log_format = '%(asctime)s %(levelname)s:%(message)s'
    date_format = '%m/%d/%Y %I:%M:%S %p'

    # Clear any existing handlers to avoid duplicates
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # File handler - writes to log file
    # Use a custom handler that flushes after each write to prevent data loss
    log_file_path = args.save_dir + '/training.log'
    
    class FlushingFileHandler(logging.FileHandler):
        """File handler that flushes after each emit to prevent data loss on crash."""
        def emit(self, record):
            super().emit(record)
            self.flush()
    
    file_handler = FlushingFileHandler(log_file_path, mode='w')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter(log_format, datefmt=date_format))

    # Console handler - prints to terminal
    console_handler = logging.StreamHandler(stream=sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(logging.Formatter(log_format, datefmt=date_format))

    # Configure root logger with both handlers
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    logger = logging.getLogger(__name__)
    
    # Ensure file handler flushes immediately to prevent data loss on crash
    file_handler.flush()

    # DDP setting
    if (bool(args.ddp)):
        backend = 'nccl'
        dist_url = 'env://'
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])

        dist.init_process_group(backend=backend, init_method=dist_url, rank=rank, world_size=world_size)
        torch.cuda.set_device(local_rank)
        torch.distributed.barrier()
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(int(args.gpu_num))
        world_size, local_rank = 1, 0

    try:

        # prepare data -> (train_dataset, val_dataset), (train_loader, val_loader), (train_sampler, val_sampler)
        (train_dataset, val_dataset), (train_loader, val_loader), (train_sampler, val_sampler) = load_datasetloader(args=args,
        dtype=torch.FloatTensor, world_size=world_size, rank=local_rank, mode='train')

        # define network
        solver = load_solvers(args, train_dataset.num_samples, logger, torch.FloatTensor,
                              world_size=world_size, rank=local_rank, isTrain=True)


        # training and validation
        for e in range(args.start_epoch, args.num_epochs):

            # ------------------------------------------
            # Training
            # ------------------------------------------
            solver.mode_selection(isTrain=True)
            if (bool(args.ddp)):
                train_sampler.set_epoch(e)
                torch.distributed.barrier() # Synchronize before training

            start_epoch = time.time()
            for b, data in enumerate(train_loader):
                start_batch = time.time()
                solver.train(data)
                solver.learning_rate_step(e)
                end_batch = time.time()
                solver.print_training_progress(e, b, (end_batch-start_batch))
            end_epoch = time.time()
            
            # print training progress and reset loss tracker
            solver.normalize_loss_tracker()
            solver.print_status(e, start_epoch, end_epoch)
            solver.init_loss_tracker()


            # ------------------------------------------
            # Evaluation
            # ------------------------------------------
            if (e % int(args.save_every) == 0):
                if (bool(args.ddp)):
                    torch.distributed.barrier()  # Synchronize before evaluation
                solver.eval(val_dataset, val_loader, val_sampler, e)


    except Exception:
        # Log error with full traceback
        error_msg = traceback.format_exc()
        logger.error("="*80)
        logger.error("ERROR OCCURRED DURING TRAINING")
        logger.error("="*80)
        logger.error(error_msg)
        logger.error("="*80)
        # Force flush to ensure error is written to file
        file_handler.flush()
        # Re-raise to see error in console
        raise

if __name__ == '__main__':

    # load args
    import argumentparser as ap
    args = ap.args

    # Setup save directory
    args.save_dir = os.path.join('./saved_models/', f'{args.dataset_type}_{args.model_name}_model{args.exp_id}')
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir, exist_ok=True)
        args.load_pretrained = 0  # no pre-trained nets in new directory

    # run main()
    main(args)



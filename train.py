import argparse
import os
import torch
from utils.builder import ConfigBuilder
import utils.misc as utils
import yaml
from utils.logger import get_logger


#----------------------------------------------------------------------------

def subprocess_fn(args):
    utils.setup_seed(args.seed * args.world_size + args.rank)

    logger = get_logger("train", args.run_dir, utils.get_rank(), filename='iter.log', resume=args.resume)

    # args.logger = logger
    args.cfg_params["logger"] = logger

    # build config
    logger.info('Building config ...')
    builder = ConfigBuilder(**args.cfg_params)

    logger.info('Building dataloaders ...')
    #args.dos_normalize = False
    train_dataloader = builder.get_dataloader(split = 'train', smear = args.smear, dos_normalize=args.dos_normalize)
    logger.info('Train dataloaders build complete')
    test_dataloader = builder.get_dataloader(split = 'test', smear = args.smear, dos_normalize=args.dos_normalize)
    logger.info('Test dataloaders build complete')
    valid_dataloader = builder.get_dataloader(split = 'valid', smear = args.smear, dos_normalize=args.dos_normalize)
    logger.info('valid dataloaders build complete')
    print(type(test_dataloader), type(valid_dataloader))
    steps_per_epoch = len(train_dataloader)

    model_params = args.cfg_params['model']['params']
    lr_scheduler_params = model_params['lr_scheduler']
    for key in lr_scheduler_params:
        if 'by_step' in lr_scheduler_params[key]:
            if lr_scheduler_params[key]['by_step']:
                for key1 in lr_scheduler_params[key]:
                    if "epochs" in key1:
                        lr_scheduler_params[key][key1] *= steps_per_epoch
    




    # build model
    logger.info('Building models ...')
    model = builder.get_model()
    model_checkpoint = os.path.join(args.run_dir, 'checkpoint_latest.pth')
    if args.resume:
        if os.path.exists(model_checkpoint):
            model.load_checkpoint(model_checkpoint)
        else:
            logger.info("checkpoint not exist")

    model_without_ddp = utils.DistributedParallel_Model(model, args.local_rank)

    if args.world_size > 1:
        for key in model_without_ddp.model:
            utils.check_ddp_consistency(model_without_ddp.model[key])

    for key in model_without_ddp.model:
        params = [p for p in model_without_ddp.model[key].parameters() if p.requires_grad]
        cnt_params = sum([p.numel() for p in params])
        # print("params {key}:".format(key=key), cnt_params)
        logger.info("params {key}: {cnt_params}".format(key=key, cnt_params=cnt_params))



    # valid_dataloader = builder.get_dataloader(split = 'valid')
    # logger.info('valid dataloaders build complete')
    logger.info('begin training ...')

    model_without_ddp.stat()
    
    model_without_ddp.trainer(train_dataloader, test_dataloader, valid_dataloader,  builder.get_max_epoch(), checkpoint_savedir=args.run_dir, resume=args.resume)
    
    #model_without_ddp.test(test_data_loader=valid_dataloader, epoch=0)
    
def main(args):
    if args.world_size > 1:
        utils.init_distributed_mode(args)
    else:
        args.rank = 0
        args.distributed = False
        args.local_rank = 0
        torch.cuda.set_device(args.local_rank)
    desc = f'world_size{args.world_size:d}'

    if args.desc is not None:
        desc += f'-{args.desc}'

    alg_dir = args.cfg.split("/")[-1].split(".")[0]
    args.outdir = args.outdir + "/" + alg_dir
    run_dir = os.path.join(args.outdir, f'{desc}')
    print(run_dir)
    os.makedirs(run_dir, exist_ok=True)

    with open(args.cfg, 'r') as cfg_file:
        cfg_params = yaml.load(cfg_file, Loader = yaml.FullLoader)

    cfg_params['dataloader']['num_workers'] = args.per_cpus
    dataset_vnames = cfg_params['dataset']['train'].get("vnames", None)
    dataset_smear = cfg_params['dataset']['smear'] = args.smear
    if dataset_vnames is not None:
        constants_len = len(dataset_vnames.get('constants'))
    else:
        constants_len = 0
    cfg_params['model']['params']['constants_len'] = constants_len

    if args.rank == 0:
        with open(os.path.join(run_dir, 'training_options.yaml'), 'wt') as f:
            yaml.dump(vars(args), f, indent=2, sort_keys=False)
            yaml.dump(cfg_params, f, indent=2, sort_keys=False)

    args.cfg_params = cfg_params
    args.run_dir = run_dir

    print('Launching processes...')
    subprocess_fn(args)

    
if __name__ == "__main__":
    def str2bool(v):
        if v.lower() in ('true', 't', 'yes', 'y', '1'):
            return True
        elif v.lower() in ('false', 'f', 'no', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume',         type = str2bool,    default = False,                                        help = 'resume')
    parser.add_argument('--seed',           type = int,     default = 0,                                            help = 'seed')
    parser.add_argument('--cuda',           type = int,     default = 0,                                            help = 'cuda id')
    parser.add_argument('--world_size',     type = int,     default = 1,                                            help = 'Number of progress')
    parser.add_argument('--per_cpus',       type = int,     default = 1,                                            help = 'Number of perCPUs to use')
    # parser.add_argument('--world_size',     type = int,     default = -1,                                           help = 'number of distributed processes')
    parser.add_argument('--init_method',    type = str,     default='tcp://127.0.0.1:23456',                        help = 'multi process init method')
    parser.add_argument('--outdir',         type = str,     default='./output',  help = 'Where to save the results')
    parser.add_argument('--cfg', '-c',      type = str,     default = os.path.join('configs', 'default.yaml'),      help = 'path to the configuration file')
    parser.add_argument('--desc',           type=str,       default='STR',                                          help = 'String to include in result dir name')
    parser.add_argument('--smear',          type = float,   default = None,                                         help = 'Gaussian smearing')
    parser.add_argument('--dos_normalize',  type = str2bool,    default = False,                                        help = 'standarlize')
    args = parser.parse_args()

    main(args)

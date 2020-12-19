from train import parse_args
import os


if __name__ == '__main__':
    args = parse_args()
    os.system('python -m torch.distributed.launch '
              '--nproc_per_node={} '
              '--master_port={} '
              './train.py '
              '{}'
              '--launcher {}'.format(args.gpus, args.port, args.config, args.launcher))
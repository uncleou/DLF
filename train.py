"""Training entry point for the original and selective-null LFA variants."""
import argparse

from run import DLF_run

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', choices=['mosi', 'mosei'], default='mosi')
parser.add_argument('--lfa-mode', choices=['original', 'selective_null'], default='original')
parser.add_argument(
    '--shared-loss-mode',
    choices=['original', 'none', 'atp', 'sent', 'atp_sent', 'original_atp_sent'],
    default='original'
)
args = parser.parse_args()

DLF_run(
    model_name='DLF', dataset_name=args.dataset,
    config={
        'lfa_mode': args.lfa_mode,
        'shared_loss_mode': args.shared_loss_mode,
    },
    is_tune=False, seeds=[1111], model_save_dir='./pt', res_save_dir='./result',
    log_dir='./log', mode='train', is_training=True
)

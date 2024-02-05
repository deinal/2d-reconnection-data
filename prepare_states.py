import os
import argparse
import numpy as np
from constants import runs
import re


def extract_number(file_name):
    match = re.search(r'(\d+)', file_name)
    if match:
        return int(match.group(0))

def create_datasets(data_dir, out_dir, step_size, num_steps):
    for run in runs:
        run_id = run['id']
        t_min = run['t_min']
        t_max = run['t_max']

        # Ensure output directories exist
        os.makedirs(os.path.join(out_dir, 'test'), exist_ok=True)
        os.makedirs(os.path.join(out_dir, 'val'), exist_ok=True)
        os.makedirs(os.path.join(out_dir, 'train'), exist_ok=True)

        files = sorted(
            [f for f in os.listdir(data_dir) if f.startswith(run_id) and f.endswith('.npy')],
            key=extract_number)
        
        all_steps = [extract_number(f) for f in files]
            
        # Test Set
        test_start_step = t_max - 32*step_size+step_size # aim at 30 step forecast with 2 init states
        test_steps = list(range(test_start_step, t_max+step_size, step_size))
        test_files = [files[step-t_min] for step in test_steps]
        print(test_files, len(test_files))
        test_data = np.stack([np.load(os.path.join(data_dir, f)) for f in test_files], axis=0)
        np.save(os.path.join(out_dir, 'test', f'{run_id}_{test_start_step}.npy'), test_data)

        # Validation Set
        val_steps = []
        for test_step in test_steps:
            # test step +/- 15 steps belong to validation
            surrounding_indices = [test_step + offset for offset in range(-15, 16) if offset != 0]
            val_steps.extend(surrounding_indices)
        for step in val_steps[::3]:
            start = step - t_min
            val_files = files[start : start + step_size*num_steps : step_size]
            if len(val_files) == num_steps:
                val_data = np.stack([np.load(os.path.join(data_dir, f)) for f in val_files], axis=0)
                np.save(os.path.join(out_dir, 'val', f'{run_id}_{str(step).zfill(4)}.npy'), val_data)

        # Training Set
        train_steps = set(all_steps) - set(test_steps) - set(val_steps)
        for step in sorted(list(train_steps))[::3]:
            start = step - t_min
            train_files = files[start : start + step_size*num_steps : step_size]
            if len(train_files) == num_steps:
                train_data = np.stack([np.load(os.path.join(data_dir, f)) for f in train_files], axis=0)
                np.save(os.path.join(out_dir, 'train', f'{run_id}_{str(step).zfill(4)}.npy'), train_data)

if __name__ == '__main__': 
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--out_dir', type=str)
    parser.add_argument('-d', '--data_dir', type=str, default='frames')
    parser.add_argument('-s', '--step_size', type=int, default=60)
    parser.add_argument('-n', '--num_steps', type=int, default=5)
    args = parser.parse_args()

    create_datasets(args.data_dir, args.out_dir, args.step_size, args.num_steps)

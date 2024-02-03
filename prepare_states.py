import os
import argparse
import numpy as np
from constants import runs


def load_and_concatenate(step_size, num_states, data_dir):
    for run in runs:
        run_id = run['id']
        files = sorted([f for f in os.listdir(data_dir) if f.startswith(run_id) and f.endswith('.npy')])
        total_files = len(files)
    
    # Calculate total possible start indices for concatenation
    total_starts = total_files - step_size * (num_states - 1)
    print(total_starts, 'series in total')
    
    # Loop over all possible start indices
    for start in range(total_starts):
        series_files = files[start:start + step_size * num_states:step_size]
        print(series_files)

        # Ensure k files to concatenate
        if len(series_files) == num_states:
            series_data = [np.load(os.path.join(data_dir, f)) for f in series_files]
            concatenated_series = np.stack(series_data, axis=0)
            
            save_path = os.path.join(args.out_dir, f'{args.run_id}_{start+1}.npy')
            np.save(save_path, concatenated_series)
            print(f'Saved series {start} to {save_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--out_dir', type=str)
    parser.add_argument('-d', '--data_dir', type=str, default='frames/test')
    parser.add_argument('-s', '--step_size', type=int, default=60)
    parser.add_argument('-n', '--num_states', type=int, default=15)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    load_and_concatenate(args.step_size, args.num_states, args.data_dir)

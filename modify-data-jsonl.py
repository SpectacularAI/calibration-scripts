"""
Modify Spectacular AI data JSONL files

Input JSONL in read from stdin an output written to stdout
"""
import json
import sys

def modify_jsonl_sorted(in_f, out_f, func):
    lines = []
    for line in in_f:
        d = func(json.loads(line))
        if d is not None:
            lines.append(d)
    for l in sorted(lines, key=lambda d: d.get('time', -1e100)):
        out_f.write(json.dumps(l)+'\n')

def modify_jsonl_stream(in_f, out_f, func):
    for line in in_f:
        d = func(json.loads(line))
        if d is not None:
            out_f.write(json.dumps(d)+'\n')
    return out_f

def shift_imu_time(data, delta_t):
    if 'sensor' in data:
        data['time'] += delta_t
    return data

def shift_frame_times(data, delta_t):
    if 'frames' in data:
        data['time'] += delta_t
        for cam in data['frames']:
            if 'time' in cam: cam['time'] += delta_t
    return data

def drop_zero_imu_samples(data):
    if 'sensor' in data and all([v == 0.0 for v in data['sensor']['values']]): return None
    return data

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(__doc__)
    subs = p.add_subparsers(dest='command')

    sub_shift_imu = subs.add_parser('shift_imu_time', help='shift IMU timestamps by given delta_t')
    sub_shift_imu.add_argument('delta_t', help='timestamp shift in seconds', type=float)

    sub_shift_frames = subs.add_parser('shift_frame_times', help='shift frame timestamps by given delta_t')
    sub_shift_frames.add_argument('delta_t', help='timestamp shift in seconds', type=float)

    sub_drop_zero_imus = subs.add_parser('drop_zero_imu_samples', help='remove all-zero IMU samples')

    p.add_argument('-s', '--sort', action='store_true', help='sort the resulting file based on timestamp')
    
    args = p.parse_args()
    f_in = sys.stdin
    f_out = sys.stdout

    if args.command == 'shift_imu_time':
        func = lambda x: shift_imu_time(x, args.delta_t)
    elif args.command == 'shift_frame_times':
        func = lambda x: shift_frame_times(x, args.delta_t)
    elif args.command == 'drop_zero_imu_samples':
        func = drop_zero_imu_samples
    else:
        assert(False)

    if args.sort:
        modify_jsonl_sorted(f_in, f_out, func)
    else:
        modify_jsonl_stream(f_in, f_out, func)

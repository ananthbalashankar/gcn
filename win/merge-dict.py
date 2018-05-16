import utils
import os
from argparse import ArgumentParser

arguments = ArgumentParser()
arguments.add_argument('--base', type=str, default='')
arguments.add_argument('--pattern', type=str, default='')
args = arguments.parse_args()

base_dir = args.base
merged_res = {}
for i in range(100):
    try:
        res = utils.read_pickle(os.path.join(base_dir, '{0}_{1}'.format(args.pattern, i)))
    except:
        continue
    for k, v in res.iteritems():
        merged_res[k] = v

utils.write_pickle(merged_res, os.path.join(base_dir, '{0}_merged'.format(args.pattern)))

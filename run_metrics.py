
import os
import time
import re
import bisect
from collections import OrderedDict
import numpy as np
import tensorflow as tf
import scipy.ndimage
import scipy.misc
import sys
import argparse

import misc
import tfutil
import train
import dataset
import util_scripts

if __name__ == "__main__":
    argv = sys.argv
    prog = argv[0]
    parser = argparse.ArgumentParser(
        prog        = prog,
        description = 'Script for evaluating the metrics',
        epilog      = 'Type "%s <command> -h" for more information.' % prog)

    parser.add_argument("--run-id", type=int, required=True)
    parser.add_argument("--metric", type=str, choices=["swd-16k", "fid-10k", "fid-50k", "is-50k", "msssim-20k", "ALL"], default="ALL")

    args = parser.parse_args(argv[1:])
    print("Arguments used:", args)
    run_id = args.run_id
    metric = args.metric

    print("Metric: ", metric)   
    import config
    
    import contextlib
    @contextlib.contextmanager
    def setup_before_metric():
        misc.init_output_logging()
        np.random.seed(config.random_seed)
        print('Initializing TensorFlow...')
        os.environ.update(config.env)
        tfutil.init_tf(config.tf_config)
        try:
            yield
        except Exception as e:
            print(e)
            raise e
        finally:
            if tf.get_default_session():
                tf.get_default_session().close()
            tf.reset_default_graph()

   
    if metric == "fid-10k" or metric == "ALL":
        with setup_before_metric():
            log='metric-fid-10k.txt'
            config.num_gpus = 1
            config.desc = log.split('.')[0] + '-' + str(run_id)
            util_scripts.evaluate_metrics(run_id=run_id, log=log, metrics=['fid'], num_images=10000, real_passes=1)
    
    if metric == "fid-50k" or metric == "ALL":
        with setup_before_metric():
            log='metric-fid-50k.txt'
            config.num_gpus = 1
            config.desc = log.split('.')[0] + '-' + str(run_id)
            util_scripts.evaluate_metrics(run_id=run_id, log=log, metrics=['fid'], num_images=50000, real_passes=1)
    
    if metric ==  "is-50k" or metric == "ALL":
        with setup_before_metric():
            log='metric-is-50k.txt'
            config.num_gpus = 1
            config.desc = log.split('.')[0] + '-' + str(run_id)
            util_scripts.evaluate_metrics(run_id=run_id, log=log, metrics=['is'], num_images=50000, real_passes=1)
    
    if metric == "msssim-20k" or metric == "ALL":
        with setup_before_metric():
            log='metric-msssim-20k.txt'
            config.num_gpus = 1
            config.desc = log.split('.')[0] + '-' + str(run_id)
            util_scripts.evaluate_metrics(run_id=run_id, log='metric-msssim-20k.txt', metrics=['msssim'], num_images=20000, real_passes=1)
    
    if metric == "swd-16k" or metric == "ALL":
        with setup_before_metric():
            log='metric-swd-16k.txt'
            config.num_gpus = 1
            config.desc = log.split('.')[0] + '-' + str(run_id)
            util_scripts.evaluate_metrics(run_id=run_id, log=log, metrics=['swd'], num_images=16384, real_passes=2)
    
    print('Exiting...')

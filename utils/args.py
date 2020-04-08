"""
Copyright (c) Facebook, Inc. and its affiliates.
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import argparse
import pathlib


class Args(argparse.ArgumentParser):
    """
    Defines global default arguments.
    """

    def __init__(self, **overrides):
        """
        Args:
            **overrides (dict, optional): Keyword arguments used to override default argument values
        """

        super().__init__(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

        self.add_argument('-seed', default=2019, type=int, help='Seed for random number generators')
        self.add_argument('-data-path', type=str, default='../data/CC359/')
        self.add_argument('-mask-style', default='cartesian_1d', type=str)
        self.add_argument('-same', action='store_true', help='all mask will be same in a sequence')
        self.add_argument('-cf', default=0.08, type=float,
                            help='center fraction')
        self.add_argument('-acc', default=4, type=float, help='accerlation')
        self.add_argument('-mf', type=int, default=32, help='max frames')

        self.add_argument('-dset', default='calgary')
        self.add_argument('-gap', default=1, type=int, help='sequence sample interval')

        # settings for fast mri 
        self.add_argument('-sample-rate', type=float, default=0.1)
        self.add_argument('-resolution', default=320, type=int)
        self.add_argument('-acquisition', nargs='+', default=['CORPD_FBK', 'CORPDFS_FBK'])

        self.add_argument('-lr', type=float, default=1e-3)
        self.add_argument('-weight-decay', type=float, default=0.)
        self.add_argument('-num-epochs', type=int, default=200)
        self.add_argument('-lr-step', type=int, default=160)
        self.add_argument('-lr-gamma', type=float, default=1e-1)
        
        
        
        self.add_argument('-device', type=str, default='cuda')
        
        self.add_argument('-exp-dir', type=str, required=True)
        self.add_argument('-log-interval', type=int, default=10)
        self.add_argument('-eval', action='store_true')
        # Override defaults with passed overrides
        self.set_defaults(**overrides)
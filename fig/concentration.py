import itertools as it
import random
import warnings
from glob import glob
from pathlib import Path
from socket import gethostname

import click

import numpy as np
import pandas as pd
import ziggurat
from psutil import virtual_memory
from tqdm import tqdm


def lambda_min(A):
    return np.linalg.eigvalsh(A)[0]


def chunks(n, batch_size):
    return [batch_size] * (n // batch_size) + [n % batch_size]


def get_samplefun(N, d, m, samples,
                  available_memory=int(0.2 * virtual_memory().available),
                  rgen=np.random):
    batch_size = available_memory // (64 * samples * d**2)
    print(f'Using batch_size={batch_size} for N={N} d={d} m={m} samples={samples}')
    assert batch_size > 0

    def prod_gaussians(shape, N, rgen):
        return np.prod(rgen.randn(*shape, N), axis=-1)

    def sample(seed=1234):
        rgen.seed(seed)
        Bs = np.zeros((samples, d, d))
        Gs = np.zeros((samples, d, d))

        desc = f'N={N} d={d} '
        the_iter = tqdm(chunks(m, batch_size), desc=desc, bar_format='{desc} {percentage:3.0f}%')
        for chunk, chunk_size in enumerate(the_iter):
            X = prod_gaussians((chunk_size, samples), N - 1, rgen)
            Y = prod_gaussians((chunk_size, samples), N - 1, rgen)
            A = rgen.randn(chunk_size, samples, d)

            z = X[:, :, None] * A
            Bs += 1 / m * np.sum(z[:, :, :, None] * z[:, :, None, :], axis=0)
            Gs += 1 / m * np.sum(((X * Y)[:, :, None, None] * A[:, :, :, None]) * A[:, :, None, :], axis=0)

        print(Gs.shape)
        result = np.empty((2, samples), dtype=np.float_)
        result[0, :] = np.fromiter((lambda_min(B) for B in Bs), dtype=np.float_)
        # compute 2 -> 2 norm
        result[1, :] = np.linalg.norm(Gs, ord=2, axis=(1, 2))
        return result

    return sample


NR_MEASUREMENTS = {
    'lin_lin': lambda C, N, d: C * N * d,
    'sq_sq': lambda C, N, d: C * N**2 * d**2,
    'sq_lin': lambda C, N, d: C * N**2 * d,
    'lin_sq': lambda C, N, d: C * N * d**2
}


def sample_as_df(site, dim, const, mode, samples, seed_gen):
    m = NR_MEASUREMENTS[mode](const, site, dim)
    samplefun = get_samplefun(site, dim, m, samples, rgen=ziggurat)
    x = samplefun(seed_gen.randint(2**31))
    return pd.DataFrame(data={'N': [site] * samples,
                              'd': [dim] * samples,
                              'C': [const] * samples,
                              'mode': [mode] * samples,
                              'lmin_B': np.log(x[0]),
                              'lmax_G': np.log(x[1])})


@click.group()
def main():
    pass


@main.command()
@click.option('--sites', required=True)
@click.option('--dims', required=True)
@click.option('--const', default=10, type=int)
@click.option('--samples', default=100000, type=int)
@click.option('--mode', default='lin_lin', type=click.Choice(NR_MEASUREMENTS.keys()))
@click.option('--output-dir', required=True,
              type=click.Path(exists=True, file_okay=False, dir_okay=True, writable=True))
@click.option('--seed', default=random.randint(0, 2**31))
def compute(sites, dims, const, samples, mode, output_dir, seed):
    sites_pool = [int(s) for s in sites.split(':')]
    dim_pool = [int(s) for s in dims.split(':')]
    pool = list(it.product(sites_pool, dim_pool))
    seed_gen = np.random.RandomState(seed=seed)
    print(f'Using global seed={seed}')

    df = pd.concat((sample_as_df(site, dim, const, mode, samples, seed_gen)
                    for site, dim, in tqdm(pool, desc='TOTAL')),
                   ignore_index=True)

    filename = f'samples_const={const}_mode={mode}_{gethostname()}.csv'
    df.to_csv(filename, index=False)
    print()
    print('Done')


@main.command()
@click.argument('globpatt')
@click.option('--output-file', required=True, default='samples.csv',
              type=click.Path(file_okay=True, dir_okay=False, writable=True))
def merge(globpatt, output_file):
    files = list(glob(globpatt))
    df = pd.concat((pd.read_csv(fpath, index_col=False) for fpath in tqdm(list(files))),
                   ignore_index=True)
    df.to_csv(output_file, index=False)


if __name__ == '__main__':
    main()

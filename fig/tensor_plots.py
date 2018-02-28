import functools as ft
import itertools as it
import time
from collections import namedtuple
from pathlib import Path

import click

import matplotlib.pyplot as pl
import numpy as np
import seaborn as sns
from tqdm import tqdm
from scipy import stats


@click.group()
def main():
    pass


def lambda_min(A):
    return np.linalg.eigvalsh(A)[0]


def get_samplefun(N, d, m):
    def sample(samples, rgen=np.random):
        # since overlaps are iid Gaussians
        overlap_a_v = np.prod(rgen.randn(samples, m, N - 1), axis=2)
        overlap_a_v_perp = np.prod(rgen.randn(samples, m, N - 1), axis=2)
        a_N = rgen.randn(samples, m, d)
        B_Ns = 1 / m * np.sum(((overlap_a_v * overlap_a_v)[:, :, None, None] * a_N[:, :, :, None]) * a_N[:, :, None, :],
                              axis=1)
        G_Ns = 1 / m * np.sum(((overlap_a_v * overlap_a_v_perp)[:, :, None, None] * a_N[:, :, :, None]) * a_N[:, :, None, :],
                              axis=1)
        result = np.empty((2, samples), dtype=np.float_)
        result[0, :] = np.fromiter((lambda_min(B_N) for B_N in B_Ns), dtype=np.float_)
        # compute 2 -> 2 norm
        result[1, :] = np.linalg.norm(G_Ns, ord=2, axis=(1, 2))
        return result

    return sample


NR_MEASUREMENTS = {
    'lin_lin': lambda C, N, d: C * N * d,
    'sq_sq': lambda C, N, d: C * N**2 * d**2
}


def compute_routine(sites, dim, const, samples, batch_size, mode, output_dir,
                    use_tqdm=True):
    batch_samples = [batch_size] * (samples // batch_size)
    batch_samples = tqdm(batch_samples) if use_tqdm else batch_samples
    rest = samples % batch_size
    if rest > 0:
        batch_samples += [rest]

    m = NR_MEASUREMENTS[mode](const, sites, dim)
    samplefun = get_samplefun(sites, dim, m)
    samples = (samplefun(batch, rgen=np.random.RandomState(seed=np.random.randint(int(2**31))))
               for batch in batch_samples)
    result = np.concatenate(list(samples), axis=1)
    np.save(Path(output_dir) / 'samples_N={}_d={}_C={}_mode={}.npy'.format(sites, dim, const, mode),
            result, allow_pickle=False)


@main.command()
@click.option('--sites', required=True, type=int)
@click.option('--dim', required=True, type=int)
@click.option('--const', default=10, type=int)
@click.option('--samples', default=100000, type=int)
@click.option('--batch-size', default=256, type=int)
@click.option('--mode', default='lin_lin', type=click.Choice(NR_MEASUREMENTS.keys()))
@click.option('--output-dir', default='../data/',
              type=click.Path(exists=True, file_okay=False, dir_okay=True, writable=True))
def compute(sites, dim, const, samples, batch_size, mode, output_dir):
    assert batch_size <= samples
    compute_routine(sites, dim, const, samples, batch_size, mode, output_dir)
    print('Done.')


class MashedFunction(object):
    def __init__(self, f):
        self.f = f

    def __call__(self, args):
        return self.f(*args)


@main.command(name='compute-all')
@click.option('--samples', default=100000, type=int)
@click.option('--batch-size', default=256, type=int)
@click.option('--output-dir', default='../data/',
              type=click.Path(exists=True, file_okay=False, dir_okay=True, writable=True))
@click.option('--mode', default='lin_lin', type=click.Choice(NR_MEASUREMENTS.keys()))
def compute_all(samples, batch_size, output_dir, mode):
    f = ft.partial(compute_routine, output_dir=output_dir, samples=samples,
                   batch_size=batch_size, mode=mode, use_tqdm=False)
    sites_pool = [10, 20, 40, 80]
    dim_pool = [5, 10, 20, 40]
    const_pool = [10, 100]

    iterator = it.product(sites_pool, dim_pool, const_pool)
    total = len(sites_pool) * len(dim_pool) * len(const_pool)
    for sites, dim, const in tqdm(iterator, total=total):
        f(sites, dim, const)


@main.command(name='tensor_lognormal.pdf')
@click.option('--samples', default=100000, type=int)
def lognormal_plot(samples):
    N = [5, 20, 50]
    rgen = np.random.RandomState(1234)

    fig, ax = pl.subplots(figsize=(6, 3))
    z = np.linspace(-100, 10, 1000)
    bins = 100
    for n in N:
        x = np.prod(rgen.randn(samples, n)**2, axis=-1)
        dist = stats.norm(loc=-(np.euler_gamma + np.log(2)) * n,
                          scale=np.pi / np.sqrt(2) * np.sqrt(n))
        l, = pl.plot(z, dist.pdf(z), label=r'$n=' + str(n) + '$')
        ax.hist(np.log(x), bins=bins, color=l.get_color(), density=True)

    ax.set_xlim(-100, 10)
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$\mathbb{P}\left((\log(X^2) = x \right)$')
    ax.legend(loc='upper left')
    fig.tight_layout()
    fig.savefig('tensor_lognormal.pdf')



if __name__ == '__main__':
    main()

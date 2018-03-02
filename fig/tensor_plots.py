import functools as ft
import itertools as it
import time
from collections import namedtuple
from pathlib import Path

import click

import matplotlib.pyplot as pl
import mpmath
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from tqdm import tqdm

MATLAB_ENGINE = None
π = np.pi


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
    'sq_sq': lambda C, N, d: C * N**2 * d**2,
    'sq_lin': lambda C, N, d: C * N**2 * d
}


def concentration_compute_routine(sites, dim, const, samples, batch_size, mode, output_dir,
                    use_tqdm=True):
    batch_samples = [batch_size] * (samples // batch_size)
    batch_samples = batch_samples
    rest = samples % batch_size
    if rest > 0:
        batch_samples += [rest]
    if use_tqdm:
        batch_samples = tqdm(batch_samples)

    m = NR_MEASUREMENTS[mode](const, sites, dim)
    samplefun = get_samplefun(sites, dim, m)
    samples = (samplefun(batch, rgen=np.random.RandomState(seed=np.random.randint(int(2**31))))
               for batch in batch_samples)
    result = np.concatenate(list(samples), axis=1)
    np.save(Path(output_dir) / 'samples_N={}_d={}_C={}_mode={}.npy'.format(sites, dim, const, mode),
            result, allow_pickle=False)


@main.command(name='concentration-compute')
@click.option('--sites', required=True, type=int)
@click.option('--dim', required=True, type=int)
@click.option('--const', default=10, type=int)
@click.option('--samples', default=100000, type=int)
@click.option('--batch-size', default=256, type=int)
@click.option('--mode', default='lin_lin', type=click.Choice(NR_MEASUREMENTS.keys()))
@click.option('--output-dir', default='../data/',
              type=click.Path(exists=True, file_okay=False, dir_okay=True, writable=True))
def concentration_compute(sites, dim, const, samples, batch_size, mode, output_dir):
    assert batch_size <= samples
    concentration_compute_routine(sites, dim, const, samples, batch_size, mode, output_dir)
    print('Done.')


@main.command(name='concentration-compute-batch')
@click.option('--samples', default=10000, type=int)
@click.option('--batch-size', default=256, type=int)
@click.option('--output-dir', default='../data/',
              type=click.Path(exists=True, file_okay=False, dir_okay=True, writable=True))
@click.option('--mode', default='lin_lin', type=click.Choice(NR_MEASUREMENTS.keys()))
@click.option('--const', default=10)
@click.option('--sites', default='2:4:8:16:32:64')
@click.option('--dims', default='2:4:8:16:32')
def concentration_compute_batch(samples, batch_size, output_dir, mode, const,
                                sites, dims):
    f = ft.partial(concentration_compute_routine, output_dir=output_dir,
                   samples=samples, batch_size=batch_size, mode=mode,
                   const=const, use_tqdm=True)
    sites_pool = [int(s) for s in sites.split(':')]
    dim_pool = [int(s) for s in dims.split(':')]
    pool = list(it.product(sites_pool, dim_pool))
    for sites, dim, in tqdm(pool):
        f(sites, dim)
    print('Done')


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


def true_cdf_function(α, n):
    def f(z):
        y = mpmath.meijerg(a_s=[[1] + [1/2] * n, []], b_s=[[], [0]], z=z)
        return 1 - 1/2**α * π**(-n / 2) * y
    return np.frompyfunc(f, 1, 1)


def approx_cdf_function(ν, ξ, N, order):
    # convert to float due to MATLAB's weird behavior
    H0j = MATLAB_ENGINE.H0j(float(N))
    H1j = MATLAB_ENGINE.H1j(float(N))

    def zeroth_order_approx(z):
        zeroth_order = np.sum(H0j * np.log(z)**np.arange(N))
        return ν + 2**(-ξ) * π**(-N / 2) * z**(-1 / 2) * zeroth_order

    def first_order_approx(z):
        zeroth_order = z**(-1 / 2) * np.sum(H0j * np.log(z)**np.arange(N))
        first_order = z**(-3 / 2) * np.sum(H1j * np.log(z)**np.arange(N))
        return ν + 2**(-ξ) * π**(-N / 2) * (zeroth_order + first_order)

    f = {0: zeroth_order_approx, 1: first_order_approx}.get(order)
    return np.frompyfunc(f, 1, 1)


def powerlog_dataframe(t, N, key):
    true_cdfs = {
        'X': true_cdf_function(1, N),
        'Y': true_cdf_function(0, N),
        'Z': true_cdf_function(0, N)
    }

    zeroth_orders = {
        'X': approx_cdf_function(1/2, 1, N, 0),
        'Y': approx_cdf_function(0, 0, N, 0),
        'Z': approx_cdf_function(0, 0, N, 0),
    }

    first_orders = {
        'X': approx_cdf_function(1/2, 1, N, 1),
        'Y': approx_cdf_function(0, 0, N, 1),
        'Z': approx_cdf_function(0, 0, N, 1),
    }

    t_to_zs = {
        'X': lambda n, t: 2**n / t**2,
        'Y': lambda n, t: 2**n / t,
        'Z': lambda n, t: 2**n / t**2
    }

    z = t_to_zs[key](N, t)
    df = pd.DataFrame(data=t, columns=['t'])
    df['y_inf'] = true_cdfs[key](z)
    df['y_0'] = zeroth_orders[key](z)
    df['y_1'] = first_orders[key](z)
    df['N'] = N
    df['rv'] = key
    return df


@main.command(name='series-compute')
@click.option('--sites', default='2:5:8')
@click.option('--output-file', default='../data/series.csv',
              type=click.Path(file_okay=True, dir_okay=False, writable=True))
@click.option('--nr-points', default=100)
def series_compute(sites, output_file, nr_points):
    print('Starting MATLAB engine. This might take a while...')
    import matlab.engine
    global MATLAB_ENGINE
    MATLAB_ENGINE = matlab.engine.start_matlab(option='-nodesktop -nojvm')
    MATLAB_ENGINE.addpath('matlab_lib/')

    t = np.linspace(1e-10, 1, nr_points)
    Ns = [int(n) for n in sites.split(':')]
    df = pd.concat([powerlog_dataframe(t, N, key)
                    for N, key in tqdm(list(it.product(Ns, ['X', 'Y', 'Z'])))],
                   ignore_index=True, verify_integrity=True)
    df.to_csv(output_file)
    print('Done.')


def melt_cols(df, col_names, **kwargs):
    return df.melt(id_vars=df.columns.drop(col_names), value_vars=col_names,
                   **kwargs)


@main.command(name='series-plot')
@click.option('--datafile', default='../data/series.csv',
              type=click.Path(exists=True, file_okay=True, dir_okay=False,
                              readable=True))
@click.option('--size', default=2)
@click.option('--aspect', default=1.6)
def series_plot(datafile, size, aspect):
    df = pd.read_csv(datafile, index_col=0)
    df['err_0'] = np.abs(df['y_inf'] - df['y_0'])
    df['err_1'] = np.abs(df['y_inf'] - df['y_1'])
    df.rv.replace('X', r'$X$', inplace=True)
    df.rv.replace('Y', r'$Y$', inplace=True)
    df.rv.replace('Z', r'$Z$', inplace=True)
    f_names = {'y_0': r'$k=0$', 'y_1': r'$k=1$', 'y_inf': r'$k=∞$'}
    err_names = {'err_0': r'$k=0$', 'err_1': r'$k=1$'}

    x_name = r'$t$'
    y_name = r'$f_k(x)$'
    df_1 = melt_cols(df.rename(columns=f_names), list(f_names.values()),
                     var_name='truncation order', value_name=y_name)
    df_1.rename(columns={'t': x_name}, inplace=True)
    grid = sns.FacetGrid(df_1, col='N', row='rv', hue='truncation order',
                         sharex='col', sharey='row', legend_out=True,
                         margin_titles=True, size=size, aspect=aspect,
                         hue_order=list(f_names.values()))
    grid.map(pl.plot, x_name, y_name, ls='--') \
        .add_legend()
    pl.savefig('tensor_series_f.pdf')

    x_name = r'$t$'
    y_name = r'$| f_∞(x) - f_k(x) |$'
    df_1 = melt_cols(df.rename(columns=err_names), list(err_names.values()),
                     var_name='truncation order', value_name=y_name)
    df_1.rename(columns={'t': x_name}, inplace=True)
    grid = sns.FacetGrid(df_1, col='N', row='rv', hue='truncation order',
                         sharex='col', sharey='row', legend_out=True,
                         margin_titles=True, size=size, aspect=aspect)
    grid.map(pl.semilogy, x_name, y_name) \
        .add_legend()
    pl.savefig('tensor_series_err.pdf')


if __name__ == '__main__':
    main()

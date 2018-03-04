import functools as ft
import itertools as it
import pickle
import re
import time
from collections import namedtuple
from pathlib import Path

import click

import matplotlib.pyplot as pl
import mpmath
import mpnum as mp
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


CONCENTRATION_DESCS = {
    'lin_lin': r'm = {} \times N \times d',
    'sq_lin': r'm = {} \times N^2 \times d',
    'lin_sq': r'm = {} \times N \times d^2',
}

@main.command(name='concentration-preprocess')
@click.option('--datafile', default='../data/concentration_samples.csv',
              type=click.Path(exists=True, file_okay=True, dir_okay=False,
                              readable=True))
@click.option('--outfile', default='../data/concentration_samples.pkl')
def concentration_preprocess(datafile, outfile):
    df = pd.read_csv(datafile)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    df['quotient'] = df['lmin_B'] - df['lmax_G']

    df['desc'] = df.apply(
        lambda row: r'$' + CONCENTRATION_DESCS[row['mode']].format(row['C']) + '$',
        axis=1
    )
    df.to_pickle(outfile)


def add_quantiles(df, grid, mode='lin_lin', q=0.05):
    quantiles = df.groupby(['N', 'd', 'C', 'mode']).quotient.quantile(q)
    the_iter = it.product(enumerate(grid.row_names),
                          enumerate(grid.col_names),
                          enumerate(grid.hue_names))
    for (i, N), (j, d), (k, C) in the_iter:
        try:
            # dirty hack :)
            C = int(C.split()[2])
            x = quantiles[(N, d, C, mode)]
        except KeyError:
            print(f'No quantile found for {N}, {d}, {mode}')
            continue
        ax = grid.axes[i, j]
        color = ax.get_lines()[k].get_color()
        ax.axvline(x, color=color, ls=':')


@main.command(name='tensor_concentration_dists.pdf')
@click.option('--datafile', default='../data/concentration_samples.pkl',
              type=click.Path(exists=True, file_okay=True, dir_okay=False,
                              readable=True))
@click.option('--outfile', default='tensor_concentration_dists.pdf')
@click.option('--size', default=2.2)
@click.option('--aspect', default=0.9)
def concentration_distplot(datafile, outfile, size, aspect):
    df = pd.read_pickle(datafile)
    df_sel = df.loc[(df['N'] <= 64) & (df['d'] <= 32) & (df['mode'] == 'lin_lin')]
    grid = sns.FacetGrid(df_sel, col='d', row='N', hue='desc', sharex=True,
                         sharey=False, margin_titles=True, aspect=aspect,
                         size=size)
    grid.map(sns.distplot, 'quotient', bins=100)
    add_quantiles(df, grid, mode='lin_lin')

    grid.add_legend(title='Legend')
    grid.set_xlabels(r'$\log\, \frac{z_B}{z_G}$')
    pl.savefig(outfile)


def compute_quantiles(df, quantile):
    res1 = df.groupby(['N', 'd', 'desc']).quotient.quantile(quantile)
    q_B = df.groupby(['N', 'd', 'desc']).lmin_B.quantile(quantile / 2)
    q_G = df.groupby(['N', 'd', 'desc']).lmax_G.quantile(1 - quantile / 2)
    res2 = q_B - q_G
    result = pd.concat([res1, res2], axis=1)
    result.columns = ['coherent', 'incoherent']
    return result


@main.command(name='tensor_concentration_quantiles.pdf')
@click.option('--datafile', default='../data/concentration_samples.pkl',
              type=click.Path(exists=True, file_okay=True, dir_okay=False,
                              readable=True))
@click.option('--outfile', default='tensor_concentration_quantiles.pdf')
@click.option('--size', default=2.2)
@click.option('--aspect', default=2.8)
@click.option('--quantile', default=0.05)
def concentration_quantileplot(datafile, outfile, size, aspect, quantile):
    df = pd.read_pickle(datafile)
    df_sel = df.loc[(df['N'] <= 128) & (df['d'] <= 32)]
    quantiles = compute_quantiles(df_sel, quantile).reset_index()
    row_order = [
        '$' + CONCENTRATION_DESCS['lin_lin'].format(10) + '$',
        '$' + CONCENTRATION_DESCS['lin_lin'].format(100) + '$',
        '$' + CONCENTRATION_DESCS['lin_sq'].format(10) + '$',
        '$' + CONCENTRATION_DESCS['sq_lin'].format(10) + '$',
    ]

    grid = sns.FacetGrid(quantiles, row='desc', hue='d', sharex='col',
                         sharey=False, aspect=aspect, size=size,
                         row_order=row_order)
    grid.map(pl.semilogx, 'N', 'coherent')
    grid.map(pl.semilogx, 'N', 'incoherent', ls='--')

    grid.add_legend(title='Legend')
    grid.set_xlabels(r'$N$')
    grid.set_ylabels(r'$\log \ x_{0.05}$')
    grid.set_titles('{row_name}')

    for ax in grid.axes[:, 0]:
        ax.set_xticks([2, 4, 8, 16, 32, 64, 128])
        ax.set_xticks([], minor=True)
        ax.set_xticklabels([2, 4, 8, 16, 32, 64, 128])
    pl.savefig(outfile)


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
    df_1 = melt_cols(df.rename(columns=f_names), list(f_names.values()),
                     var_name='truncation order', value_name='f')
    grid = sns.FacetGrid(df_1, col='N', row='rv', hue='truncation order',
                         sharex=True, sharey='row', legend_out=True,
                         margin_titles=True, size=size, aspect=aspect,
                         hue_order=list(f_names.values()))
    grid.map(pl.plot, 't', 'f', ls='--').add_legend()
    grid.set_xlabels(r'$t$')
    grid.set_ylabels(r'$f_k(t)$')
    pl.savefig('tensor_series_f.pdf')

    err_names = {'err_0': r'$k=0$', 'err_1': r'$k=1$'}
    df_1 = melt_cols(df.rename(columns=err_names), list(err_names.values()),
                     var_name='truncation order', value_name='err')
    grid = sns.FacetGrid(df_1, col='N', row='rv', hue='truncation order',
                         sharex=True, sharey='row', legend_out=True,
                         margin_titles=True, size=size, aspect=aspect)
    grid.map(pl.semilogy, 't', 'err').add_legend()
    grid.set_xlabels(r'$t$')
    grid.set_ylabels(r'$\left\vert f_\infty(x) - f_k(x) \right\vert$')
    pl.savefig('tensor_series_err.pdf')


@main.command(name='recovery-plot')
@click.option('--datafile', default='../data/recoveries_rank1_5percent.pkl',
              type=click.Path(exists=True, file_okay=True, dir_okay=False,
                              readable=True))
def recovery_plot(datafile):
    with open(datafile, 'rb') as buf:
        X, recoveries = pickle.load(buf)

    fig, ax = pl.subplots(figsize=(3, 4))
    x = np.arange(len(recoveries[0])) + 1

    for X_sharp in recoveries:
        ax.plot(x, [mp.normdist(X, X_s) for X_s in X_sharp])

    outfile = 'tensor_' + str(Path(datafile).stem) + '.pdf'
    ax.set_xlabel(r'Epoch $h$')
    ax.set_ylabel(r'$\left\Vert X - Y \right\Vert_2$')
    pl.tight_layout()
    pl.savefig(outfile)


if __name__ == '__main__':
    main()

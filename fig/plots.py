#!/usr/bin/env python3
# encoding: utf-8

import os
import pickle
import re
import warnings
from glob import glob
from pathlib import Path
from warnings import warn

import click

import h5py
import matplotlib.pyplot as pl
import numpy as np
import pandas as pd
import pypllon as plon
import seaborn as sns
from matplotlib.gridspec import GridSpec
from pypllon.parsers import load_simdata
from tqdm import tqdm

pl.style.use('ggplot')

try:
    from tools.helpers import Progress
except ImportError:
    Progress = lambda iterator: iterator


NR_MEASUREMENTS = {2: 10, 3: 15, 5: 25}


@click.group()
def main():
    pass

###############################################################################
#                              Simulation Plots                               #
###############################################################################


def devplot(x, y, label, ax, semilog=False):
    mus = np.mean(y, axis=1)
    if semilog:
        l, = ax.semilogy(x, mus, label=label)
    else:
        l, = ax.plot(x, mus, label=label)

    y1 = np.percentile(y, 2.5, axis=1)
    y1 = np.percentile(y, 2.5, axis=1)
    y2 = np.percentile(y, 97.5, axis=1)
    ax.fill_between(x, y1, y2, alpha=0.2, color=l.get_color())


@main.command()
@click.argument('h5file')
@click.argument('key')
@click.option('--outfile', help='File to save to', default='data/simdata.h5')
def pandarize(h5file, key, outfile):
    with h5py.File(h5file, 'r') as infile:
        df = load_simdata(infile)
        metadata = dict(infile.attrs)

    print('Number of failed reconstructions:', len(df[df.isnull().any(axis=1)]))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=pd.io.pytables.PerformanceWarning)
        with pd.HDFStore(outfile, mode='a') as store:
            store.put(key, df, format='f')
            store.get_storer(key).attrs.metadata = metadata
    return 0


@np.vectorize
def recons_error(target, recov):
    a, b = plon.fix_phases('rows_by_max', target, recov)
    return np.linalg.norm(a - b)


@main.command()
@click.argument('key')
@click.option('--infile', help='File to load pandas dataframe from', default='data/simdata.h5')
@click.option('--errconst', help='Constant to premultiply error threshold', default=4)
@click.option('--noline', help='Plot conjectured phase transition line', is_flag=True)
def simplot(key, infile, errconst, noline):
    fig = pl.figure(0, figsize=(5, 4))
    errconst = float(errconst)
    print('Loading {} from {}'.format(key, infile))
    with pd.HDFStore(infile, 'r') as store:
        df = store[key]
        metadata = dict(store.get_storer(key).attrs.metadata)

    # Preprocessing
    full_len = len(df)
    df = df[df['recons'].notnull()]
    print('Dropping {} failed simulations'.format(full_len - len(df)))

    # Creating error & success variables
    df['recons_err'] = recons_error(df['target'], df['recons'])
    df['recons_err'].fillna(1.0, inplace=True)
    sigma = metadata['SIGMA']

    if sigma <= 0.0:
        print('Assuming noiseless measurements')
        df['recons_success'] = df['recons_err'] < errconst * df['dim']
        title = r'$\mathbb{P}\left(\Vert M - M^\sharp \Vert_2 < ' \
            + str(errconst) + r'\times n \right)$'
    else:
        print('Assuming noisy measurements with sigma={}'.format(sigma))
        df['recons_success'] = df['recons_err'] < errconst * sigma * df['dim']
        title = r'$\mathbb{P}\left(\Vert M - M^\sharp \Vert_2 < ' \
            + str(errconst) + r'\times \sigma n \right)$'

    # create success matrix
    p_success = df.groupby(['dim', 'measurements']) \
        .recons_success.mean().astype(float).reset_index().pivot('measurements', 'dim')
    missing_rows = set(range(min(p_success.index), max(p_success.index) + 1)) \
        .difference(p_success.index)
    p_success = p_success.append(pd.DataFrame(index=missing_rows)).sort()
    print('Appending rows without entries', missing_rows)
    # fill in the gaps using recover probability from lower nr. of measurements
    p_success.fillna(method='ffill', inplace=True)
    # for too low nr. of measurements just set to 0
    p_success.fillna(value=0, inplace=True)

    fig = pl.figure(0, figsize=(10, 8))
    ax = fig.add_subplot(1, 1, 1)

    ax.set_title(title)
    ax.set_xlabel(r'$n$ (dimension)')
    ax.set_ylabel(r'$m$ (number of measurements)')

    dims = p_success.columns.levels[1].values
    ax.set_xticks(range(0, max(dims))[::2])
    ax.set_xticks(range(0, max(dims))[::2])
    ax.set_xticklabels(np.sort(dims)[::2])
    measurements = p_success.index.values
    yticks = np.arange(10, max(measurements) + 10, 10)
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticks)
    ax.grid(False)

    if not noline:
        # x = dims - 2 due to offset of imshow
        pl.plot(dims - 2, 4 * dims - 4)

    img = ax.imshow(p_success.values[:101], aspect='auto', cmap=pl.cm.gray,
                     origin='lower')
    fig.colorbar(img)
    fig.tight_layout()

    fig.savefig('sim_{}.pdf'.format(key))


@main.command(name='error-scaling')
@click.option('--name', default='dft')
@click.option('--dim', default=5)
def error_scaling(name, dim):
    fig = pl.figure(0, figsize=(5, 4))
    ax = fig.gca()
    fname_query = re.compile(r'dft_dim=(\d*)_ts=(\d*).npz')
    mpl_query = re.compile(r'\$t=(\d*)\$')
    for fname in glob(f'../data/{name}_dim={dim}_*.npz'):
        match = fname_query.search(os.path.basename(fname))
        ts = int(match.group(2))

        data = np.load(fname)
        ms, errors = data['arr_0'], data['arr_1'] / dim
        devplot(ms, errors, label=r'$t=' + str(ts) + '$', ax=ax, semilog=True)

    handles, labels = ax.get_legend_handles_labels()
    # sort both labels and handles by labels
    sortf = lambda t: int(mpl_query.search(t[0]).group(1))
    labels, handles = zip(*sorted(zip(labels, handles), key=sortf))
    ax.legend(handles, labels)
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    ax.set_xlim(10 if dim == 5 else 20, ms[-1])
    ax.set_ylim(5e-3 if dim == 5 else 1e-2, ylim[1])

    ax.set_xlabel(r'Number of preparation vectors $m$')
    ax.set_ylabel(r'$\frac{1}{n} \, \left\Vert M - M^\sharp \right\Vert_2$')

    fig.tight_layout()
    fig.savefig('phaselift_sim_errorscaling_{}_{}.pdf'.format(name, dim))


###############################################################################
#                             Experimental Plots                              #
###############################################################################
def get_tmat_singles(df):
    DETECTORS = df.attrs['DETECTORS']
    # Average total photon count (for normalization purposes)
    tmat_single = np.array([df['SINGLE_COUNTS'][str(i)] for i in range(len(DETECTORS))], dtype=float)
    deteff = df['SINGLE_COUNTS'].attrs['DETEFF']
    tmat_single = tmat_single * deteff
    # axis = 0 since we flip the tmat later
    tmat_single /= np.max(np.sum(tmat_single, axis=0))
    tmat_single = np.sqrt(tmat_single.T)
    return tmat_single


def get_reference(df):
    try:
        dip_reconstructed = df['DIP_RECONSTRUCTED'].value
        normalization_sq = np.max(np.sum(np.abs(dip_reconstructed)**2, axis=1))
        return dip_reconstructed / np.sqrt(normalization_sq), True
    except KeyError:
        warn('No dips reconstruction found for {} in {}'
             .format(df, df.file.filename))
        return get_tmat_singles(df), False


def get_intensities(df, sel=slice(None)):
    # NOT NORMALIZED!
    deteff = df['RAWCOUNTS'].attrs['DETEFF']

    intensities = {}
    for key, counts_group in df['RAWCOUNTS'].items():
        counts = counts_group.value * deteff
        time_avg = 1.0 * np.mean(counts[sel], axis=0)  # take time-average
        intensities[key] = time_avg
    # we normalize them globally for now, otherwise the solver might have trouble
    normalization = max(sum(x) for x in intensities.values())
    return {key: x / normalization for key, x in intensities.items()}


def recover(df, optim_func=plon.lr_recover_l2, m_sel=slice(None),
            t_sel=slice(None)):
    pvec_keys = np.array(list(df['PREPVECS'].keys()))[m_sel]
    # note that intensities are not normalized!
    intesity_dict = get_intensities(df, sel=t_sel)
    valid_keys = set(pvec_keys).intersection(set(intesity_dict.keys()))

    pvecs = np.asarray([df['PREPVECS'][pkey].value for pkey in valid_keys])
    intensities = np.asarray([intesity_dict[pkey] for pkey in valid_keys])
    recov, errs = plon.recover(pvecs, intensities, optim_func=optim_func, reterr=True)
    # but also force our normalization on the reconstruction
    recov /= np.sqrt(np.max(np.sum(np.abs(recov)**2, axis=1)))
    return recov, errs


EXDESC = {'Fou': 'Fourier', 'Id': 'Identity', 'Swap': 'Swap',
          '_01': 'Random A', '_02': 'Random B', '_03': 'Random C'}
SIZEDESC = {'M2': '2x2', 'M3': '3x3', 'M5': '5x5'}
EXDATAFILES = ['M2Fou.h5', 'M2Id.h5', 'M2_01.h5', 'M2_02.h5', 'M2_03.h5',
               'M3Fou.h5', 'M3Id.h5', 'M3_01.h5', 'M3_02.h5', 'M3_03.h5',
               'M5Fou.h5', 'M5Id.h5', 'M5Swap.h5',
               'M5_01.h5', 'M5_02.h5', 'M5_03.h5']


def id_to_label(the_id):
    label = SIZEDESC[the_id[:2]] + ' '
    for key, value in EXDESC.items():
        if the_id[2:].startswith(key):
            return label + value
    raise ValueError('No label found for', the_id)


def empty_dataframe(**kwargs):
    df = pd.DataFrame(columns=kwargs.keys())
    for col, dtype in kwargs.items():
        df[col] = df[col].astype(dtype)
    return df


def append_reconstructions(recons, df, scheme, idx, samples):
    dim = len(df['TARGET'])
    max_measurements = len(df['PREPVECS'])
    m = min(max_measurements, NR_MEASUREMENTS[dim])

    for sample in range(samples):
        m_sel = np.random.choice(max_measurements, size=m, replace=False)
        recovery, _ = recover(df, m_sel=m_sel)
        recons.loc[len(recons)] = [idx, recovery, m_sel, scheme, sample]


def df_with_errors(refs, recons):
    def targetref_error(row):
        target = refs.loc[row.idx].target
        dim = len(target)
        colphaseopt = row.Scheme == 'HOM-dip'
        recons, _ = plon.best_tmat_phases(target, row.recons, cols=colphaseopt,
                                         rows=True)
        return np.linalg.norm(target - recons) / dim
    recons['targetref_error'] = recons.apply(targetref_error, axis=1)

    def dipsref_error(row):
        if row.Scheme == 'HOM-dip':
            return np.nan

        dipsref = refs.loc[row.idx].reference
        dim = len(dipsref)

        if refs.loc[row.idx].with_phases:
            recov, _ = plon.best_tmat_phases(dipsref, row.recons, cols=True, rows=True)
            return np.linalg.norm(dipsref - recov) / dim
        else:
            return np.linalg.norm(np.abs(dipsref) - np.abs(row.recons)) / dim
    recons['dipsref_error'] = recons.apply(dipsref_error, axis=1)
    return recons


@main.command()
@click.option('--datadir', help='Directory containing datafiles',
              default='data/')
@click.option('--samples', help='Number of bootstrapping samples',
              default=100)
@click.option('--outfile', help='File to save to', default='data/exdata.h5')
def expandarize(datadir, samples, outfile):
    references = empty_dataframe(target=object, reference=object, with_phases=bool)
    recons = empty_dataframe(idx=object, recons=object, m_sel=object,
                             Scheme=object, sample=int)

    for datafile in tqdm(EXDATAFILES):
        with h5py.File(datadir + datafile) as h5file:
            # since GAUSS/RECR have the same target/dips reconstruction
            ref, with_phases = get_reference(h5file['GAUSS'])
            target = h5file['GAUSS']['TARGET'].value
            ref, _ = plon.best_tmat_phases(target, ref, cols=True, rows=True)

            idx = id_to_label(os.path.splitext(os.path.basename(datafile)[0])[0])
            references.loc[idx] = [target, ref, with_phases]
            append_reconstructions(recons, h5file['GAUSS'], 'Uniform', idx, samples)
            append_reconstructions(recons, h5file['RECR'], 'RECR', idx, samples)
            # append dipsref
            recons.loc[len(recons)] = [idx, ref, None, 'HOM-dip', 0]

    recons = df_with_errors(references, recons)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=pd.io.pytables.PerformanceWarning)
        with pd.HDFStore(outfile, mode='a') as store:
            store.put('references', references, format='f')
            store.put('reconstructions', recons, format='f')

    print('DONE')
    return 0


def plot_errors(x, y, hue, data, mode='violin', **kwargs):
    if mode == 'violin':
        plot = sns.violinplot(x=x, y=y, hue=hue, data=data, linewidth=.1,
                            scale='count', **kwargs)
    elif mode == 'scatter':
        plot = sns.stripplot(x=x, y=y, hue=hue, data=data, dodge=True, jitter=.2,
                            size=3, **kwargs)
    else:
        raise ValueError('{} is not a valid mode', mode)

    medians = data.groupby([x, hue]).median().reset_index()
    sns.stripplot(x=x, y=y, hue=hue, data=medians, dodge=True, marker='D',
                  linewidth=0.5, size=6, **kwargs)
    return plot


def set_grid(ax):
    ax.yaxis.grid(True, which='major')
    ax.xaxis.grid(False, which='major')
    ax.xaxis.grid(True, which='minor', linewidth=2.5)
    nr_ticks = len(ax.get_xticklabels())
    ax.set_xticks(np.arange(nr_ticks), minor=False)
    ax.set_xticks(np.arange(nr_ticks + 1) - 0.5, minor=True)
    ax.xaxis.set_tick_params(which='minor', width=0)


@main.command(name='phaselift_ex_overview.pdf')
@click.option('--infile', help='File used for expandarize', default='../data/exdata.h5')
@click.option('--mode', help='Which plot mode to use', default='violin')
def explot_overview(infile, mode):
    with pd.HDFStore(infile, 'r') as store:
        recons = store['reconstructions']

    fig = pl.figure(0, figsize=(9, 4))
    gs = GridSpec(1, 2, width_ratios=[10, 6])
    axes = tuple(fig.add_subplot(g) for g in gs)

    ax = axes[0]
    df = recons[(recons.Scheme != 'HOM-dip') & ~(recons.idx.str.startswith('5x5'))]
    order = df.idx.unique()
    plot_errors(x='idx', y='dipsref_error', hue='Scheme', mode=mode, data=df,
                ax=ax, order=order)
    ax.set_ylabel(r'$\frac{1}{n} \, \left\Vert M_\mathrm{dips} - M^\sharp \right\Vert_2$')

    ax = axes[1]
    df = recons[(recons.Scheme != 'HOM-dip') & (recons.idx.str.startswith('5x5'))]
    order = df.idx.unique()
    plot = plot_errors(x='idx', y='dipsref_error', hue='Scheme', mode=mode,
                       data=df, ax=ax, order=order)
    ax.set_ylabel(r'$\frac{1}{n} \, \left\Vert \vert M_\mathrm{dips} \vert - \vert M^\sharp \vert \right\Vert_2$')

    ylim = max(ax.get_ylim()[1] for ax in axes)
    for ax in axes:
        ax.set_ylim(0, ylim)
        ax.set_xlabel('')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=70, ha='right')
        set_grid(ax)
        ax.legend_.remove()
    axes[1].set_yticklabels([])

    pl.tight_layout()
    fig.subplots_adjust(top=0.95, bottom=0.30)
    handles, labels = plot.get_legend_handles_labels()
    axes[0].legend(handles[:2], ['Uniform', 'RECR'], loc='upper left')
    pl.savefig('phaselift_ex_overview.pdf')


@main.command(name='phaselift_ex_targetref.pdf')
@click.option('--infile', help='File used for expandarize', default='../data/exdata.h5')
@click.option('--mode', help='Which plot mode to use', default='violin')
def explot_target(infile, mode):
    with pd.HDFStore(infile, 'r') as store:
        recons = store['reconstructions']

    fig = pl.figure(0, figsize=(7.0, 4.))
    ax = pl.gca()
    df = recons[~(recons.idx.str.startswith('5x5'))]
    order = df.idx.unique()
    plot = plot_errors(x='idx', y='targetref_error', hue='Scheme', mode=mode, ax=ax, data=df,
                       hue_order=['Gaussian', 'RECR', 'HOM-dip'], order=order)
    ax.set_ylabel(r'$\frac{1}{n} \, \left\Vert M_\mathrm{target} - M^\sharp \right\Vert_2$')

    set_grid(ax)
    ax.set_xlabel('')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=70, ha='right')
    ax.legend_.remove()
    pl.tight_layout()
    fig.subplots_adjust(top=0.95, bottom=0.30)
    handles, labels = plot.get_legend_handles_labels()
    ax.legend(handles[:3], ['Uniform', 'RECR', 'HOM-dip'], loc='upper left')
    ax.set_ylim(0, ax.get_ylim()[1])
    pl.savefig('phaselift_ex_targetref.pdf')


def get_intensities(df, sel=slice(None)):
    # NOT NORMALIZED!
    deteff = df['RAWCOUNTS'].attrs['DETEFF']

    intensities = {}
    for key, counts_group in df['RAWCOUNTS'].items():
        counts = counts_group.value * deteff
        time_avg = 1.0 * np.mean(counts[sel], axis=0)  # take time-average
        intensities[key] = time_avg
    # we normalize them globally for now, otherwise the solver might have trouble
    normalization = max(sum(x) for x in intensities.values())
    return {key: x / normalization for key, x in intensities.items()}


def recover_old(df, optim_func=plon.lr_recover_l2, m_sel=slice(None),
                t_sel=slice(None)):
    pvec_keys = np.array(list(df['PREPVECS'].keys()))[m_sel]
    # note that intensities are not normalized!
    intesity_dict = get_intensities(df, sel=t_sel)
    valid_keys = set(pvec_keys).intersection(set(intesity_dict.keys()))

    pvecs = np.asarray([df['PREPVECS'][pkey].value for pkey in valid_keys])
    intensities = np.asarray([intesity_dict[pkey] for pkey in valid_keys])
    recov, errs = plon.recover(pvecs, intensities, optim_func=optim_func, reterr=True)
    # but also force our normalization on the reconstruction
    recov /= np.sqrt(np.max(np.sum(np.abs(recov)**2, axis=1)))
    return recov, errs


def recons_errors(data, t_sel=slice(None), m_sel=slice(None),
                  ref_func=get_reference, power_corrected=False):
    recov, vec_errs = recover_old(data, m_sel=m_sel, t_sel=t_sel)
    ref, with_phases = ref_func(data)

    if with_phases:
        recov, _ = plon.best_tmat_phases(ref, recov)
        errors = np.ravel(np.abs(recov - ref))
    else:
        errors = np.ravel(np.abs(np.abs(recov) - np.abs(ref)))

    return errors, vec_errs


def m_dependence_plot(data, measurements=None, t_sel=slice(None),
                      ref_func=get_reference, samples=20):
    m_max = len(data['PREPVECS'])
    x = measurements if measurements is not None \
        else list(range(1, len(data['PREPVECS']), 3))
    errors = np.array([[np.linalg.norm(recons_errors(data, t_sel=t_sel, ref_func=ref_func,
                                                     m_sel=np.random.choice(m_max, size=m, replace=False))[0])
                       for _ in tqdm(range(samples))] for m in tqdm(x)])
    return x, errors


def t_dependence_plot(data, m_sel=slice(None), timesteps=None,
                      ref_func=get_reference, samples=20):
    max_timesteps, _ = list(data['RAWCOUNTS'].values())[0].shape
    x = timesteps if timesteps is not None \
        else list(range(1, max_timesteps, 2))
    errors = np.array([[np.linalg.norm(recons_errors(data, m_sel=m_sel, ref_func=ref_func,
                                                     t_sel=np.random.choice(max_timesteps, size=t, replace=False))[0])
                       for _ in tqdm(range(samples))] for t in tqdm(x)])
    return x, errors




@main.command(name='phaselift_ex_details.pdf')
@click.argument('datafile', default='../data/M5_03.h5')
@click.option('--dipsref/--targetref', help='Use dips as reference, otherwise use target',
              default=False)
@click.option('--samples', help='Nr of samples to use', default=20)
@click.option('--compute/--plot-only', help='Nr of samples to use', default=False)
@click.option('--cache-path', help='Path to cache directory', default='../data/')
def exdetails(datafile, dipsref, samples, compute, cache_path):
    fig = pl.figure(0, figsize=(9, 3.5))

    ref_func = get_reference if dipsref else lambda d: (d['TARGET'], True)
    with h5py.File(datafile) as h5file:
        ax1 = fig.add_subplot(1, 2, 1)
        plot_data = {}
        if compute:
            for key in ['GAUSS', 'RECR']:
                plot_data[key] = m_dependence_plot(h5file[key], samples=samples,
                                                   ref_func=ref_func)

            with open(Path(cache_path) / 'mplot.pkl', 'wb') as buf:
                pickle.dump(plot_data, buf)
        else:
            with open(Path(cache_path) / 'mplot.pkl', 'rb') as buf:
                plot_data = pickle.load(buf)

        devplot(*plot_data['GAUSS'], label='Uniform', ax=ax1)
        devplot(*plot_data['RECR'], label='RECR', ax=ax1)
        ax1.set_xlabel(r'# of measurements $m$')
        ax1.set_ylabel(r'$\frac{1}{n} \, \left\Vert \vert M_\mathrm{dips} \vert - \vert M^\sharp \vert \right\Vert_2$')

        ax2 = fig.add_subplot(1, 2, 2)
        plot_data = {}
        if compute:
            m_sel = np.random.choice(30, size=20, replace=False)
            for key in ['GAUSS', 'RECR']:
                plot_data[key] = t_dependence_plot(h5file[key], samples=samples,
                                                   ref_func=ref_func)


            with open(Path(cache_path) / 'tplot.pkl', 'wb') as buf:
                pickle.dump(plot_data, buf)
        else:
            with open(Path(cache_path) / 'tplot.pkl', 'rb') as buf:
                plot_data = pickle.load(buf)

        n = 5
        x, y = plot_data['GAUSS']
        y_0 = np.mean(y[-1])
        devplot(x, (y - y_0) / n, label='Uniform', ax=ax2)
        x, y = plot_data['RECR']
        y_0 = np.mean(y[-1])
        devplot(x, (y - y_0) / n, label='RECR', ax=ax2)
        ax2.set_xlabel(r'# of time bins $t$')
        ax2.set_ylabel(r'$\frac{1}{n} \, \left\Vert \vert M_\mathrm{dips} \vert - \vert M^\sharp \vert \right\Vert_2 - E_\infty$')
        ax2.legend()

    pl.tight_layout()
    fig.subplots_adjust(top=0.95, bottom=0.15)
    pl.savefig('phaselift_ex_details.pdf')


if __name__ == '__main__':
    np.random.seed(1234567890)
    main()

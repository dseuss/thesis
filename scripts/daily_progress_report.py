#!/usr/bin/env python3
# encoding: utf-8

import collections
import functools as ft
import logging
import os
import tempfile
from datetime import date
from subprocess import PIPE, Popen

import click
import colorama
import git

colorama.init()

CONFIG = {
    'logging_level': logging.ERROR,
    'word_goal': 100,
}


ASCII_THUMPS_UP = """
            _
           /(|
          (  :
         __\  \  _____
       (____)  `|
      (____)|   |
       (____).__|
        (___)__.|_____"""


def log_functioncall(function, logger=logging):
    """@todo: Docstring for functionlogger.

    :param function: @todo
    :param logger: @todo
    :returns: @todo

    """
    @ft.wraps(function)
    def logged_function(*args, **kwargs):
        logging.debug(f'Called {function.__name__} with the following arguments: {args}, {kwargs}')
        return function(*args, **kwargs)
    return logged_function


def get_latest_commit(repo, predicate=lambda commit: True):
    """Returns the last commit that statistfied the predicate

    :param repo: @todo
    :returns: @todo

    """
    for c in repo.iter_commits():
        if predicate(c):
            return c

    raise ValueError('No commit found that satisfies predicate')


def get_basecommit(repo):
    try:
        today = date.today()
        predicate = lambda c, today=today: date.fromtimestamp(c.committed_date) < today
        basecommit = get_latest_commit(repo, predicate)
        logging.debug(f'Found commit {basecommit} as basecommit.')
        return basecommit
    except ValueError:
        raise ValueError('No commit found that was not from today')


def parse_line(line):
    try:
        count_str = line.split(':')[1].split(' ')[2:]
        assert len(count_str) % 3 == 0
        return dict(zip(count_str[2::3], count_str[::3]))
    except IndexError:
        logging.debug(f'line "{line}" seems to be empty')
        return {}



def parse_wdiff_output(output):
    counts = collections.defaultdict(int)

    for count_dict in (parse_line(l) for l in output.strip().split('\n')):
        for key, val in count_dict.items():
            counts[key] = counts[key] + int(val)

    return counts

def get_wordcount_diff(repo, basecommit, currentcommit):
    script_dir = os.path.dirname(os.path.realpath(__file__))
    command = [f'GIT_EXTERNAL_DIFF={script_dir}/tex-worddiff',
               'git', '--no-pager', f'--git-dir="{repo.git_dir}"',
               f'--work-tree="{repo.working_tree_dir}"',
               'diff', str(basecommit), str(currentcommit), ' --', '"*.tex"']
    command_str = ' '.join(command)
    logging.info(f'Calling command: {command_str}')

    proc = Popen(command_str, stdout=PIPE, stderr=PIPE, shell=True)
    stdout, stderr = proc.communicate()
    logging.debug(stderr)
    return parse_wdiff_output(stdout.decode('utf-8'))


def print_message(word_diff, words):
    print("")
    print(colorama.Fore.GREEN + 'INSERTED: ' + str(word_diff['inserted']))
    print(colorama.Fore.RED +   'DELETED:  ' + str(word_diff['deleted']))

    if word_diff['inserted'] >= words:
        print(colorama.Fore.YELLOW + ASCII_THUMPS_UP)
    else:
        print(colorama.Fore.YELLOW + f'Keep working until you have {words} words')
    print(colorama.Style.RESET_ALL)


@click.command()
@click.option('--words', default=None, help='Goal set for the number of daily written words')
@click.option('--directory', default='.', help='Base directory of the git repo')
@click.option('--basecommit', default=None, help='Commit to compare current status to')
@log_functioncall
def report(words, directory, basecommit):
    """Looks at all tex documents in `directory` and compares the number of
    words written in them now to the last commit of yesterday.
    """
    directory = os.path.abspath(os.path.expanduser(directory))
    words = CONFIG.get('word_goal', 0) if words is None else int(words)

    try:
        repo = git.Repo(path=directory)
    except (git.InvalidGitRepositoryError, git.NoSuchPathError) as e:
        print(f'No git repository found in {directory}')
        return -1

    try:
        basecommit = get_basecommit(repo) if basecommit is None \
            else repo.commit(basecommit)
        currentcommit = repo.commit('HEAD')
        word_diff = get_wordcount_diff(repo, basecommit, currentcommit)
        print_message(word_diff, words)
    except ValueError as e:
        print(e)
        return -1

    return 0

if __name__ == '__main__':
    logging.basicConfig(level=CONFIG.get('logging_level', logging.DEBUG))
    report()

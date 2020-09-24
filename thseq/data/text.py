import re

__all__ = ['debpe']

BPE_PATTERN = re.compile('(@@ )|(@@ ?$)')


def debpe(line):
    tokenized = isinstance(line, (tuple, list))
    if tokenized:
        line = ' '.join(line)
    line = BPE_PATTERN.sub('', line)
    if tokenized:
        line = line.split()
    return line

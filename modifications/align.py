import hashlib
from subprocess import Popen, PIPE
from modifications.util import file_cached_function
import sys
import os

def parse_pharaoh_file(f):
    result = []
    for line in f:
        pairs = line.split(' ')
        pairs = [x.split('-') for x in pairs]
        pairs = [x for x in pairs if len(x) == 2]
        pairs = [(int(s), int(t)) for s, t in pairs]
        result.append(pairs)
    return result

def _align(iterator_pairs, truncation, replace=False):
    # Create cache directory if it does not exist.
    alignment_string = []

    # Assemble the alignment corpus
    for source_iterator, target_iterator in iterator_pairs:
        for source, target in zip(source_iterator, target_iterator):
            alignment_string.append('%s ||| %s' % (' '.join(source), ' '.join(target)))

    full_string = '\n'.join(alignment_string)

    fin = 'tmp.align.txt' # TODO possibly allow this location to be specified

    # Run alignment again iff this hasn't been cached
    # Write the file for usage
    with open(fin, 'w') as inp:
        inp.write(full_string)

    # Run alignments
    fast_align = Popen(['fast_align/build/fast_align', '-i',
        fin, '-d', '-o', '-v'], stdout=PIPE,
        stderr=sys.stdout)
    head = Popen(['head', '-n', '%d' % truncation],
        stdin=fast_align.stdout,
        stdout=PIPE,
        stderr=PIPE)
    fast_align.stdout.close()
    output, _ = head.communicate()
    output = output.decode()

    # Remove the (large) alignment file now that we have
    # the alignments.
    os.remove(fin)

    # Parse the resulting alignment file
    return parse_pharaoh_file(output.split('\n'))

align = file_cached_function(_align, 0)

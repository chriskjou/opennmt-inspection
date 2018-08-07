import torch
import json
import hashlib
import os
from tqdm import tqdm

cache_dir = os.environ['EXPERIMENT_CACHE_DIR']

def file_cached_function(f, function_id):
    def cached_f(*args, **kwargs):
        signature = hashlib.md5(
            json.dumps(
                (function_id, args, kwargs),
                sort_keys=True
            ).encode()
        ).hexdigest()

        tqdm.write('Calling cached function %d with signature: %s' % (function_id, signature))

        if os.path.exists(os.path.join(cache_dir, signature)):
            tqdm.write('Using cache')
            result = torch.load(os.path.join(cache_dir, signature))
        else:
            tqdm.write('Cache not available')
            result = f(*args, **kwargs)
            torch.save(result, os.path.join(cache_dir, signature))
        return result
    return cached_f

def make_if_needed(fname):
    if not os.path.exists(fname):
        os.makedirs(fname)

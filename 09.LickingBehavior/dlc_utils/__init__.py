"""
dlc_utils
=========

DeepLabCut file-handling utilities, copied VERBATIM from
`freely-moving-2P-preg/fm2p/utils/` (author: DMM). The .py files in this
package are byte-for-byte identical to their source (verified by sha256);
only this __init__.py is new.

Source commit: freely-moving-2P-preg @ 4d4fcb4  ("260727")
Copied files : files.py, helper.py, filter.py, time.py, paths.py, correlation.py

Why the sys.modules shim below
------------------------------
`files.py` and `helper.py` contain a bare `import fm2p` and reference
`fm2p.time2str` / `fm2p.up_dir` / `fm2p.read_yaml` at call time. Rather
than editing those files (so they stay verbatim and easy to diff against
upstream), we register THIS package under the name ``fm2p`` in
``sys.modules`` before importing them. Every name they reach for through
``fm2p.*`` is re-exported here, so the lookups resolve back to this
package.

Key entry point
---------------
``open_dlc_h5(path)`` -> (DataFrame, column_names), columns flattened to
``<bodypart>_x`` / ``<bodypart>_y`` / ``<bodypart>_likelihood``.
Then ``split_xyl`` + ``apply_liklihood_thresh`` to get clean x / y traces.

Dependency note
---------------
``open_dlc_h5`` calls ``pandas.read_hdf`` on the DLC ``.h5``, which needs
PyTables (``conda install -c conda-forge pytables``). It is NOT currently
in the JSY_SpMod env. If PyTables is unavailable, read DLC's ``.csv``
export instead (a thin CSV loader can be added here).
"""

import sys as _sys

# Register this package as `fm2p` so the verbatim files' `import fm2p`
# resolves here. setdefault: never clobber a real fm2p if one is installed.
_sys.modules.setdefault("fm2p", _sys.modules[__name__])

from . import time, filter, paths, correlation, files, helper  # noqa: E402,F401

from .files import (  # noqa: E402,F401
    open_dlc_h5,
    read_h5,
    write_h5,
    read_yaml,
    write_yaml,
)
from .helper import (  # noqa: E402,F401
    split_xyl,
    apply_liklihood_thresh,
    to_dict_of_arrays,
    nan_filt,
    fix_dict_dtype,
    str_to_bool,
)
from .filter import (  # noqa: E402,F401
    convfilt,
    nanmedfilt,
    sub2ind,
)
from .time import (  # noqa: E402,F401
    read_timestamp_series,
    read_timestamp_file,
    interp_timestamps,
    time2str,
    str2time,
    time2float,
    interpT,
    find_closest_timestamp,
    fmt_now,
)
from .paths import (  # noqa: E402,F401
    choose_most_recent,
    up_dir,
    find,
    filter_file_search,
    check_subdir,
    list_subdirs,
)
from .correlation import (  # noqa: E402,F401
    nanxcorr,
    corr2_coeff,
)

# V1_SpatialModulation/
# ├── main.py
# ├── helper/
#     ├── __init__.py
#     ├── subroutine.py
#     ├── suite2p2data_JSYEdit.py
#     ├── Multi_Naturalmovie_code_cohensd_suite2p.py
#     ├── read_xml.py
#     ├── time2float.py    
#     ├── twop.py

# Import functions
from .subroutine import subroutine_find_corr, subroutine_test_r
from .suite2p2data_JSYEdit import suite2p2data_JSYEdit
from .Multi_Naturalmovie_code_cohensd_suite2p import multi_naturalmovie_code_cohensd_suite2p
from .read_xml import read_xml
from .time2float import time2float
from .twop import TwoP

# Specify what is available when you import the package
__all__ = ["subroutine_find_corr", "subroutine_test_r", 
           "suite2p2data_JSYEdit", 
           "multi_naturalmovie_code_cohensd_suite2p",
           "read_xml",
           "time2float",
           "TwoP"]
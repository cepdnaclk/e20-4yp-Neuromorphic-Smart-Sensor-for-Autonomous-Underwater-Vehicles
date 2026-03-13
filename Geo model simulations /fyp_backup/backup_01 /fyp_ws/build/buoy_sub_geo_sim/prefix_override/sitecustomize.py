import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/heshan-sidantha/fyp_ws/install/buoy_sub_geo_sim'

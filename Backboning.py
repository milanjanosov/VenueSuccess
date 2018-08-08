''' ========================================= '''
''' ==--          test backboning        --== '''


import sys
sys.path.append("./backboning")
import backboning




table, nnodes, nnedges = backboning.read("/path/to/input", "column_of_interest")



#nc_table = backboning.noise_corrected(table)
#nc_backbone = backboning.thresholding(nc_table, threshold_value)
#backboning.write(nc_backbone, "network_name", "nc", "/path/to/output")

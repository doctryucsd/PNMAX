from tools import *
from backend.base import BaseCodegen
import numpy as np

class name(BaseCodegen):
    def __init__(self, require_power_of_2):
        super(name, self).__init__(require_power_of_2)
        # TODO: predictor should be defined in subclass
        self.predictor = np.array([
            0, # 'pu'
            0, # 'pu_col'
            0, # 'pu_row_change'
            0, # 'device_reg2buf'
            0, # 'device_buf2reg'
            0, # 'device_buf2bk'
            0, # 'device_buf2bk_col'
            0, # 'device_bk2buf'
            0, # 'device_bk2buf_col'
            0, # 'device_bk2gb'
            0, # 'device_bk2gb_col'
            0, # 'device_gb2bk'
            0, # 'device_gb2bk_col'
            0, # 'host_read'
            0, # 'host_read_col'
            0, # 'host_write'
            0, # 'host_write_col'
            0, # 'host_write_device_buffer'
            0, # 'host_write_device_buffer_col'
            0, # 'host_write_pu_inbuf'
            0, # 'host_write_pu_inbuf_col'
            0, # 'host_read_mac_reg'
            0, # 'host_write_mac_reg'
        ])

    # TODO: code micro for mm operator, aim global buffer
    def mm_micro(self, mm_schedule, base_group_id,
                    channel_list, rank_list, device_list, pu_num, simd_l,
                    input_bank, input_row_offset, weight_bank, weight_row_offset, output_bank, output_row_offset,
                    m_block, k_block, l_block, b_block,
                    m_row, k_row, l_row, b_row, 
                    m_block_corner, k_block_corner, l_block_corner, b_block_corner,
                    om_block, ol_block, ob_block,
                    om_row, ol_row, ob_row,
                    om_block_corner, ol_block_corner, ob_block_corner,
                    pu_m, pu_k, pu_l, pu_b,
                    pu_list, performance_threshold):
        tmp_inst_groups = []
        group_id = base_group_id
        cmd_left = performance_threshold
        
        return tmp_inst_groups, performance_threshold - cmd_left
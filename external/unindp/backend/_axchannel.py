from tools import *
from backend.base import BaseCodegen
import numpy as np

class axchannel(BaseCodegen):
    def __init__(self, require_power_of_2):
        super(axchannel, self).__init__(require_power_of_2)
        device_num = 1
        rank_num = 1
        self.predictor = np.array([
            0, # 'pu'
            SimConfig.pu_lat, # 'pu_col'
            SimConfig.read_row_change_apox, # 'pu_row_change'
            0, # 'device_reg2buf'
            0, # 'device_buf2reg'
            0, # 'device_buf2bk' # will auto-precharge
            0, # 'device_buf2bk_col'
            0, # 'device_bk2buf' # will auto-precharge
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
            SimConfig.BL/2, # 'host_write_pu_inbuf_col'
            SimConfig.BL/2, # 'host_read_mac_reg'
            SimConfig.BL/2, # 'host_write_mac_reg'
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
        cmd_left = performance_threshold
        group_id = base_group_id

        if not self.gen_code:
            row_change_num = m_row * k_row * l_row
            pu_col = ((m_row-1) * m_block + m_block_corner) * ((k_row-1) * k_block + k_block_corner) * ((l_row-1) * l_block + l_block_corner)
            read_mac = ((m_row-1) * m_block + m_block_corner) * k_row * ((l_row-1) * l_block + l_block_corner)
            wirte_inbuf = ((m_row-1) * m_block + m_block_corner) * ((k_row-1) * k_block + k_block_corner)
            return {}, row_change_num * SimConfig.read_row_change_apox + pu_col * SimConfig.pu_lat + wirte_inbuf * SimConfig.BL/2 + 2 * read_mac * SimConfig.BL/2

        # pu_row_change_lat = self.device_num * SimConfig.read_row_change_apox / SimConfig.col_change_apox
        for channel_id in channel_list:
            for rank_id in rank_list:
                device_mask = [i in device_list for i in range(SimConfig.de)]
                tmp_inst_list = []
                pu_mask = [(i in pu_list) for i in range(pu_num)]
                # output reg / buffer
                # row loop m-k-l, col loop m-l-k (fixed, best for output change)
                if mm_schedule == 'mkl':
                    for m_row_id in range(m_row):
                        m_block_real = m_block if m_row_id < m_row - 1 else m_block_corner
                        for k_row_id in range(k_row):
                            # consider corner case
                            k_block_real = k_block if k_row_id < k_row - 1 else k_block_corner
                            # compute length
                            col_len = k_block_real
                            # reload inputs
                            # NOTE: lat approx
                            # if first_rank:
                            #     cmd_left -= input_buf_write_lat * k_block_real * m_block_real
                            for input_group in range(pu_m*pu_k):
                                limited_pu_list = pu_list[input_group*pu_l:(input_group+1)*pu_l]
                                # bank_list = [ i * bank_per_pu + input_bank for i in limited_pu_list]
                                # bank_mask = [( i in bank_list ) for i in range(self.bank_num)]
                                limited_pu_mask = [(i in limited_pu_list) for i in range(pu_num)]
                                # device并行的写入pu的输入buffer
                                tmp_inst_list.append(
                                    self.create_host_write_pu_inbuf(
                                        channel_id, rank_id, device_mask, limited_pu_mask,
                                        0, k_block_real * m_block_real * SimConfig.ra # consider the channel bus
                                    )
                                )
                            for l_row_id in range(l_row):
                                l_block_real = l_block if l_row_id < l_row - 1 else l_block_corner
                                weight_row_id = l_row_id + k_row_id * l_row
                                for m_block_id in range(m_block_real):
                                    for l_block_id in range(l_block_real):
                                        input_col_offset = m_block_id * k_block_real
                                        weight_col_offset = l_block_id * k_block_real
                                        # determine precharge
                                        input_rowchange = (m_block_id == m_block_real - 1)
                                        weight_rowchange = (l_block_id == l_block_real - 1)
                                        need_rowchange = input_rowchange and weight_rowchange
                                        # if need_rowchange and first_rank:
                                        # load in result from buffer
                                        tmp_inst_list.append(self.create_host_read_mac_reg(
                                            channel_id, rank_id, device_mask, pu_mask
                                        ))
                                        # compute using pu in buf
                                        for device_id in device_list:
                                            tmp_inst_list.append(self.create_device_pu(
                                                channel_id, rank_id, device_id, pu_num, pu_mask, 
                                                (weight_bank, weight_row_offset + weight_row_id, weight_col_offset), 
                                                (weight_bank, 0, input_col_offset),
                                                col_len, need_rowchange
                                            ))
                                        tmp_inst_list.append(self.create_host_write_mac_reg(
                                            channel_id, rank_id, device_mask, pu_mask
                                        ))
                
                                            # partial = predict_*outer_loop
                                            

                elif mm_schedule == 'kml':
                    pass

                # if not self.gen_code:
                #     predict_ = np.dot(self.inst_count, self.predictor)
                #     outer_loop = k_row * \
                #         ((m_row - 1)*m_block + m_block_corner) *\
                #         ((l_row - 1)*l_block + l_block_corner) *\
                #         len(rank_list)
                #     # as rank A can change row when rank B occupies the bus, no row change occurs when reading out
                #     read_out_latency =  2 * len(pu_list) * len(rank_list) * \
                #                         ((om_row - 1) * om_block + om_block_corner) * \
                #                         ((ol_row - 1) * ol_block + ol_block_corner) * \
                #                         SimConfig.col_change_apox
                #     return {}, predict_ # * outer_loop + partial # + read_out_latency

                tmp_inst_groups.append((group_id, [], tmp_inst_list))
                group_id += 1
                cmd_left -= len(tmp_inst_list)
            break
        return tmp_inst_groups, performance_threshold - cmd_left       

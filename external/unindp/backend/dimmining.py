from math import ceil
from tools import *
import itertools as its
from backend.base import BaseCodegen
import numpy as np

class dimmining(BaseCodegen):
    def __init__(self, require_power_of_2):
        super(dimmining, self).__init__(require_power_of_2)
        rank_num = SimConfig.ra
        self.predictor = np.array([
            0, # 'pu'
            SimConfig.pu_lat/rank_num, # 'pu_col'
            SimConfig.read_row_change_apox/rank_num, # 'pu_row_change'
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
            # (SimConfig.write_to_read_row_change_apox-SimConfig.col_change_apox)/rank_num, # 'host_write'
            SimConfig.col_change_apox, # 'host_write_col'
            0, # 'host_write_device_buffer'
            SimConfig.BL/2, # 'host_write_device_buffer_col'
            0, # 'host_write_pu_inbuf'
            0, # 'host_write_pu_inbuf_col'
            SimConfig.BL/2, # 'host_read_mac_reg'
            SimConfig.BL/2, # 'host_write_mac_reg'
        ])
        
    # code micro for mm operator, aim global buffer
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
        device_mask = [True for _ in range(SimConfig.de)]

        if not self.gen_code:
            row_change_num = m_row * k_row * l_row
            pu = max(((m_row-1) * m_block + m_block_corner) * k_row * ((l_row-1) * l_block + l_block_corner), 0)
            pu_col = ((m_row-1) * m_block + m_block_corner) * ((k_row-1) * k_block + k_block_corner) * ((l_row-1) * l_block + l_block_corner)
            read_mac = max(((m_row-1) * m_block + m_block_corner) * k_row * ((l_row-1) * l_block + l_block_corner), 0)
            write_mac = max(((m_row-1) * m_block + m_block_corner) * k_row * ((l_row-1) * l_block + l_block_corner), 0)
            host_write_col = ((m_row-1) * m_block + m_block_corner) * ((k_row-1) * k_block + k_block_corner)
            return {}, (row_change_num+pu)/2 * SimConfig.read_row_change_apox +\
                pu_col * SimConfig.pu_lat +\
                      len(rank_list) * host_write_col * SimConfig.col_change_apox +\
                          read_mac * SimConfig.BL/2 + write_mac * SimConfig.BL/2
        
        for channel_id in channel_list:
            for rank_id in rank_list:
                # device_mask = [(i in device_list) for i in range(SimConfig.de)]
                pu_mask = [(i in pu_list) for i in range(pu_num)]
                tmp_inst_list = []
                # 1. 从Host端读入数据
                # 每个bank，写input_row_num_after_div行，并且
                for m_row_id in range(m_row):
                    # print(f"p_id = {os.getpid()}, 进度 = {m_row_id}/{m_row}")
                    m_block_real = m_block if m_row_id < m_row - 1 else m_block_corner
                    for k_row_id in range(k_row):
                        k_block_real = k_block if k_row_id < k_row - 1 else k_block_corner
                        # 每个input group写的时候是广播写入的
                        # only the pu that shares input can work together
                        row_id = k_row_id + m_row_id * k_row
                        bank_mask = [True] + [False for _ in range(1, self.bank_num)]
                        # 交替bank group写入，这样最大的影响只在于总线，但实际上误差的并不多
                        # 写入的带宽只用到了一半、、、
                        tmp_inst_list.append(self.create_host_write(
                            channel_id, rank_id, device_mask, bank_mask,
                            input_row_offset + row_id, 0, m_block_real * k_block_real, True
                        ))
                    #     if not self.gen_code:
                    #         predict_ = np.dot(self.inst_count, self.predictor)
                    #         outer_loop = m_row * k_row * len(rank_list)
                    #         partial = predict_*outer_loop
                    #         break
                    # if not self.gen_code:
                    #     self.reset_inst_count()
                    #     break
                    
                # 2. 计算
                outpoint_log = np.zeros((om_block*om_row+om_block_corner-om_block,
                                            (ol_block*ol_row+ol_block_corner-ol_block)*simd_l), dtype=np.bool_)
                # row loop m-k-l, col loop m-l-k (fixed, best for output change)
                if mm_schedule == 'mkl':
                    # 换行逻辑 start
                    row_iter = its.product(range(m_row), range(k_row), range(l_row))
                    row = next(row_iter) # m_row_id, k_row_id, l_row_id
                    input_row_id = row[1] + k_row * row[0] # k_row_id + m_row_id * k_row 
                    weight_row_id = row[2] + l_row * row[1] # l_row_id + k_row_id * l_row 
                    # row loop m-k-l, col loop m-l-k (fixed, best for input reuse)
                    for m_row_id in range(m_row):
                        # print(f"p_id = {os.getpid()}, 进度 = {m_row_id}/{m_row}")
                        m_block_real = m_block if m_row_id < m_row - 1 else m_block_corner
                        for k_row_id in range(k_row):
                            k_block_real = k_block if k_row_id < k_row - 1 else k_block_corner
                            for l_row_id in range(l_row):
                                # consider corner case
                                l_block_real = l_block if l_row_id < l_row - 1 else l_block_corner
                                # get the row id in input & weight
                                # input_row_id = k_row_id + m_row_id * k_row
                                # weight_row_id = l_row_id + k_row_id * l_row
                                # compute length
                                col_len = k_block_real
                                # loop over the block with the row fixed
                                for m_block_id in range(m_block_real):
                                    for l_block_id in range(l_block_real):
                                        input_col_offset = m_block_id * k_block_real
                                        weight_col_offset = l_block_id * k_block_real
                                        # determine precharge
                                        row_change = False
                                        nxt_row = None
                                        if m_block_id == m_block_real - 1 and l_block_id == l_block_real - 1:
                                            nxt_row = next(row_iter, None)
                                            if nxt_row is None: # computation ends
                                                row_change = True
                                            else: # still change
                                                nxt_input_row_id = nxt_row[1] + nxt_row[0] * k_row
                                                nxt_weight_row_id = nxt_row[2] + nxt_row[1] * l_row
                                                row_change = (nxt_input_row_id != input_row_id) or (nxt_weight_row_id != weight_row_id)
                                        # input_rowchange = (m_block_id == m_block_real - 1)
                                        # weight_rowchange = (l_block_id == l_block_real - 1)
                                        # get output id
                                        om_id = m_block_id + m_row_id * m_block
                                        ol_id = l_block_id + l_row_id * l_block
                                        # get l position in output
                                        ol_col_flat_id = ol_id // simd_l
                                        ol_col_id = ol_col_flat_id % ol_block
                                        ol_row_id = ol_col_flat_id // ol_block
                                        # get l position in output
                                        om_col_flat_id = om_id
                                        om_col_id = om_col_flat_id % om_block
                                        om_row_id = om_col_flat_id // om_block
                                        # get the col & row of output point
                                        o_col_id = om_col_id * ol_block + ol_col_id
                                        o_row_id = om_row_id * ol_row + ol_row_id
                                        # 用 host 读出并写入所有 rank pu 的输出Reg
                                        if outpoint_log[om_id, ol_id]:
                                            tmp_inst_list.append(
                                                # 2. Host -> 输出Reg
                                                self.create_host_write_rank_pu_reg(
                                                    channel_id, rank_id, pu_mask
                                                )
                                            )
                                        outpoint_log[om_id, ol_id] = True
                                        tmp_inst_list.append(self.create_rank_pu(
                                            channel_id, rank_id, pu_num, pu_mask,
                                            (weight_bank, 0, weight_row_offset + weight_row_id, weight_col_offset),
                                            (input_bank, 0, input_row_offset + input_row_id, input_col_offset),
                                            k_block_real, row_change # 不控制单独一边的开关
                                        ))
                                        # 用 host 读出并写入所有 rank pu 的输出Reg
                                        tmp_inst_list.append(
                                            # 1. 输出Reg -> Host
                                            self.create_host_read_rank_pu_reg(
                                                channel_id, rank_id, pu_mask
                                            )
                                        )
                                        if nxt_row is not None:
                                            input_row_id = nxt_input_row_id
                                            weight_row_id = nxt_weight_row_id
                                        # # check the command threshold
                                        # if not self.gen_code:
                                        #     predict_ = np.dot(self.inst_count, self.predictor)
                                        #     outer_loop = len(rank_list) * k_row * ((m_row - 1)*m_block + m_block_corner) * ((l_row - 1)*l_block + l_block_corner)
                                        #     return {}, predict_ * outer_loop + partial
                elif mm_schedule == 'kml':
                    pass

                # if not self.gen_code:
                #     predict_ = np.dot(self.inst_count, self.predictor)
                #     outer_loop = len(rank_list)
                #     return {}, predict_ * outer_loop

                tmp_inst_groups.append((group_id, [], tmp_inst_list))
                group_id += 1
                cmd_left -= len(tmp_inst_list)

            break
        
        return tmp_inst_groups, performance_threshold - cmd_left
    
    def elewise_micro(self, mm_schedule, base_group_id,
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
        bank_mask = [True] + [False for _ in range(1, self.bank_num)]
        pu_mask = [(i in pu_list) for i in range(pu_num)]
        # only k counts
        for channel_id in channel_list:
            for rank_id in rank_list:
                tmp_inst_list = []
                for k_row_id in range(k_row):
                    # NOTE: assume that the output of a row can be acommadated in the output buffer
                    col_len = k_block if k_row_id < k_row - 1 else k_block_corner
                    tmp_inst_list.append(self.create_rank_pu(
                        channel_id, rank_id, pu_num, pu_mask,
                        (input_bank, 0, input_row_offset + k_row_id, 0),
                        (weight_bank, 0, weight_row_offset + k_row_id, 0),
                        col_len, True
                    ))
                    # write back results to bank 0, use host write instead, NOTE: output_bank here is the output device
                    device_list = [pu_id * self.device_num / pu_num + output_bank for pu_id in range(pu_num)]
                    device_mask = [(i in device_list) for i in range(self.device_num)]
                    tmp_inst_list.append(self.create_host_write(
                        channel_id, rank_id, device_mask, bank_mask,
                        input_row_offset + k_row_id, 0, col_len, True
                    ))
                # predictor
                if not self.gen_code:
                    predict_ = np.dot(self.inst_count, self.predictor)
                    outer_loop = len(rank_list)
                    return {}, predict_ * outer_loop
                cmd_left -= len(tmp_inst_list)
                tmp_inst_groups.append((group_id, [], tmp_inst_list))
                group_id += 1
            break
        return tmp_inst_groups, performance_threshold - cmd_left
                        
    def softmax_micro(self, mm_schedule, base_group_id,
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
        bank_mask = [True] + [False for _ in range(1, self.bank_num)]
        pu_mask = [(i in pu_list) for i in range(pu_num)]
        # k = reduce dimension, l = parrallel dimension
        for channel_id in channel_list:
            for rank_id in rank_list:
                tmp_inst_list = []
                rw_delay = 0
                for l_row_id in range(l_row):
                    for k_row_id in range(k_row):
                        # NOTE: assume that the output of a row can be acommadated in the output buffer
                        l_block_real = l_block if l_row_id < l_row - 1 else l_block_corner
                        col_len = k_block if k_row_id < k_row - 1 else k_block_corner
                        # Element-wise exp
                        tmp_inst_list.append(self.create_rank_pu( # PU: compute exp, elewise
                            channel_id, rank_id, pu_num, pu_mask,
                            (input_bank, 0, input_row_offset + l_row_id * k_row + k_row_id, 0),
                            (weight_bank, 0, weight_row_offset + l_row_id * k_row + k_row_id, 0),
                            l_block_real * col_len, False
                        )) 
                        tmp_inst_list.append(self.create_rank_pu( # PU: compute sum, elewise
                            channel_id, rank_id, pu_num, pu_mask,
                            (input_bank, 0, input_row_offset + l_row_id * k_row + k_row_id, 0),
                            (weight_bank, 0, weight_row_offset + l_row_id * k_row + k_row_id, 0),
                            l_block_real * col_len, False
                        ))
                        rw_delay += SimConfig.read_to_write_apox
                        # write back results to bank 0, use host write instead, NOTE: output_bank here is the output device
                        # device_list = [pu_id * self.device_num / pu_num + input_bank for pu_id in range(pu_num)]
                        device_mask = [True for _ in range(self.device_num)]
                        tmp_inst_list.append(self.create_host_write(
                            channel_id, rank_id, device_mask, bank_mask,
                            input_row_offset + l_row_id * k_row + k_row_id, 0, 
                            l_block_real * col_len, True
                        ))

                # host collect the sum
                sum_len = ceil((l_block * (l_row-1) + l_block_corner)/self.simd)
                device_mask = [True for _ in range(self.device_num)]
                tmp_inst_list.append(
                    self.create_host_write_device_buffer(
                        channel_id, rank_id, device_mask, 0, sum_len
                    )
                )
                # host broadcast the sum
                tmp_inst_list.append(
                    self.create_host_write_device_buffer(
                        channel_id, rank_id, device_mask, 0, 1
                    )
                )
                # elewise div
                for l_row_id in range(l_row):
                    for k_row_id in range(k_row):
                        l_block_real = l_block if l_row_id < l_row - 1 else l_block_corner
                        col_len = k_block if k_row_id < k_row - 1 else k_block_corner
                        tmp_inst_list.append(self.create_rank_pu( # PU: compute div, elewise
                            channel_id, rank_id, pu_num, pu_mask,
                            (input_bank, 0, input_row_offset + l_row_id * k_row + k_row_id, 0),
                            (weight_bank, 0, weight_row_offset + l_row_id * k_row + k_row_id, 0),
                            l_block_real * col_len, False
                        ))
                        # extra_latency = read to write
                        rw_delay += SimConfig.read_to_write_apox
                        # write back results to bank 0, use host write instead, NOTE: output_bank here is the output device
                        # device_list = [pu_id * self.device_num / pu_num + output_bank for pu_id in range(pu_num)]
                        device_mask = [True for _ in range(self.device_num)]
                        tmp_inst_list.append(self.create_host_write(
                            channel_id, rank_id, device_mask, bank_mask,
                            input_row_offset + l_row_id * k_row + k_row_id, 0, 
                            l_block_real * col_len, True
                        ))
                # predictor
                if not self.gen_code:
                    predict_ = np.dot(self.inst_count, self.predictor)
                    outer_loop = len(rank_list)
                    return {}, predict_ * outer_loop + rw_delay
                cmd_left -= len(tmp_inst_list)
                tmp_inst_groups.append((group_id, [], tmp_inst_list))
                group_id += 1
            break
        return tmp_inst_groups, performance_threshold - cmd_left

    def layernorm_micro(self, mm_schedule, base_group_id,
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
        bank_mask = [True] + [False for _ in range(1, self.bank_num)]
        pu_mask = [(i in pu_list) for i in range(pu_num)]
        # k = reduce dimension, l = parrallel dimension
        # NOTE: x need to be both in the input device and the weight device
        for channel_id in channel_list:
            for rank_id in rank_list:
                tmp_inst_list = []
                for l_row_id in range(l_row):
                    for k_row_id in range(k_row):
                        # NOTE: assume that the output of a row can be acommadated in the output buffer
                        l_block_real = l_block if l_row_id < l_row - 1 else l_block_corner
                        col_len = k_block if k_row_id < k_row - 1 else k_block_corner
                        # Element-wise mul: x * x, result store in pu outbuf
                        tmp_inst_list.append(self.create_rank_pu( # PU: compute exp, elewise
                            channel_id, rank_id, pu_num, pu_mask,
                            (input_bank, 0, input_row_offset + l_row_id * k_row + k_row_id, 0),
                            (weight_bank, 0, weight_row_offset + l_row_id * k_row + k_row_id, 0),
                            l_block_real * col_len, False
                        ))
                        # reduce x * x
                        tmp_inst_list.append(self.create_rank_pu( # PU: compute sum, elewise
                            channel_id, rank_id, pu_num, pu_mask,
                            (input_bank, 0, input_row_offset + l_row_id * k_row + k_row_id, 0),
                            (weight_bank, 0, weight_row_offset + l_row_id * k_row + k_row_id, 0),
                            l_block_real * col_len, False
                        ))
                        # reduce x
                        tmp_inst_list.append(self.create_rank_pu( # PU: compute sum, elewise
                            channel_id, rank_id, pu_num, pu_mask,
                            (input_bank, 0, input_row_offset + l_row_id * k_row + k_row_id, 0),
                            (weight_bank, 0, weight_row_offset + l_row_id * k_row + k_row_id, 0),
                            l_block_real * col_len, True
                        ))
                # host collect the sum
                sum_len = ceil(2*(l_block * (l_row-1) + l_block_corner)/self.simd)
                device_mask = [True for _ in range(self.device_num)]
                tmp_inst_list.append(
                    self.create_host_write_device_buffer(
                        channel_id, rank_id, device_mask, 0, sum_len
                    )
                )
                # host broadcast the sum
                tmp_inst_list.append(
                    self.create_host_write_device_buffer(
                        channel_id, rank_id, device_mask, 0, 1
                    )
                )
                # elewise div
                for l_row_id in range(l_row):
                    for k_row_id in range(k_row):
                        l_block_real = l_block if l_row_id < l_row - 1 else l_block_corner
                        col_len = k_block if k_row_id < k_row - 1 else k_block_corner
                        tmp_inst_list.append(self.create_rank_pu( # PU: compute minus, elewise
                            channel_id, rank_id, pu_num, pu_mask,
                            (output_bank, 0, output_row_offset + l_row_id * k_row + k_row_id, 0),
                            (output_bank, 0, 0, 0),
                            l_block_real * col_len, False
                        ))
                        tmp_inst_list.append(self.create_rank_pu( # PU: compute div, elewise
                            channel_id, rank_id, pu_num, pu_mask,
                            (output_bank, 0, output_row_offset + l_row_id * k_row + k_row_id, 0),
                            (output_bank, 0, 0, 0),
                            l_block_real * col_len, False
                        ))
                        # write back results to bank 0, use host write instead, NOTE: output_bank here is the output device
                        # device_list = [pu_id * self.device_num / pu_num + output_bank for pu_id in range(pu_num)]
                        device_mask = [True for _ in range(self.device_num)]
                        tmp_inst_list.append(self.create_host_write(
                            channel_id, rank_id, device_mask, bank_mask,
                            output_row_offset + l_row_id * k_row + k_row_id, 0, 
                            l_block_real * col_len, True
                        ))
                # predictor
                if not self.gen_code:
                    predict_ = np.dot(self.inst_count, self.predictor)
                    outer_loop = len(rank_list)
                    return {}, predict_ * outer_loop
                cmd_left -= len(tmp_inst_list)
                tmp_inst_groups.append((group_id, [], tmp_inst_list))
                break
            break
        return tmp_inst_groups, performance_threshold - cmd_left
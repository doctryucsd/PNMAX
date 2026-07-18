import time
from frontend import *
from midend import *
from backend import *
from sim import sim
from tools import *
import argparse
from math import inf
import tqdm
import csv
import os

# NOTE: if you want to disable tqdm in the framework, you can use the following code
# def tqdm_replacement(iterable_object,*args,**kwargs):
#     return iterable_object
# tqdm_copy = tqdm.tqdm # store it if you want to use it later
# tqdm.tqdm = tqdm_replacement

def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--architecture', '-A', type=str, default='aim')
    argparser.add_argument('--name', '-N', type=str, default='gemm')
    argparser.add_argument('--outdir', '-O', type=str, default='test')
    argparser.add_argument('--workload', '-W', type=str, default='mm', help="""workload type: 
                           mm [M, K (Reduce_dimension), N, B],
                           elewise [C, H, W, B],
                           softmax [C(Softmax Dimension), H, W, B],
                           layernorm [C(norm), H(norm), W(norm), B],
                           batchnorm [C, H(norm), W(norm), B(norm)]""")
    argparser.add_argument('--workloadsize', '-S', nargs='+', type=int, default=[5000,5000,1,1])
    argparser.add_argument('--po2', '-P', action='store_true')
    argparser.add_argument('--nosim', '-NS', action='store_true')
    argparser.add_argument('--refpred', '-RP', action='store_true')
    argparser.add_argument('--cmdthre', '-T', type=float, default=3.0)
    argparser.add_argument('--topk', '-K', type=int, default=30)
    argparser.add_argument('--quicksearch', '-Q', action='store_true')
    argparser.add_argument('--allow_under_ultize', '-UU', action='store_true')
    args = argparser.parse_args()

    # set up log file
    output_dir_name = os.path.join('test_pred', args.outdir) # "debug_test" # NOTE: you can change the output dir name
    os.makedirs(f"{output_dir_name}/csv", exist_ok=True)
    os.makedirs(f"{output_dir_name}/log", exist_ok=True)
    search_strategies = 'quick' if args.quicksearch else 'slow'
    search_strategies = search_strategies + f"_top{args.topk}"
    outcsv_name = f"./{output_dir_name}/csv/_{args.name}.csv"
    log_name = f"./{output_dir_name}/log/_{args.name}.log"
    log_file = open(log_name, 'w')

    # set up workload size
    #   mm workload size: [M, K, N, B]
    #   elewise / softmax / layernorm workload size: [C, H, W, B]
    assert len(args.workloadsize) == 4 , f"Invalid workload size: {args.workloadsize}"
    if args.workload == 'mm': # [M, K (Reduce_dimension), N, B]
        mm_size = ( args.workloadsize[0]*args.workloadsize[3],
                    args.workloadsize[1],
                    args.workloadsize[2],
                    1)
        # mm_size = tuple(args.workloadsize)
    elif args.workload == 'elewise': # [C, H, W, B]
        mm_size = ( 1,
                    args.workloadsize[0]*args.workloadsize[1]*args.workloadsize[2]*args.workloadsize[3], # actually no reduce dimension
                    1,
                    1) # only k
    elif args.workload == 'softmax':    # [C(Softmax Dimension), H, W, B         ]
        mm_size = ( 1, 
                    args.workloadsize[0], 
                    args.workloadsize[1]*args.workloadsize[2]*args.workloadsize[3], 
                    1) # k,n
    elif args.workload == 'layernorm':  # [C(norm),  H(norm), W(norm), B         ]
        mm_size = ( 1, 
                    args.workloadsize[0]*args.workloadsize[1]*args.workloadsize[2], 
                    args.workloadsize[3], 
                    1)
    elif args.workload == 'batchnorm':  # [C,        H(norm), W(norm), B(norm)   ]
        mm_size = ( 1,
                    args.workloadsize[3]*args.workloadsize[1]*args.workloadsize[2],
                    args.workloadsize[0],
                    1)
    else:
        raise ValueError(f"Unknown workload: {args.workload}")
    
    # set up hw config
    if args.architecture == 'aim':
        SimConfig.read_from_yaml('./config/gddr6-aim.yaml')
        SimConfig.de_pu = [16]
        Codegen = aim16
    elif args.architecture == 'aim8':
        SimConfig.read_from_yaml('./config/gddr6-aim.yaml')
        SimConfig.de_pu = [8]
        Codegen = aim8
    elif args.architecture == 'hbm-pim':
        SimConfig.read_from_yaml('./config/hbm-pim.yaml')
        Codegen = hbmpim
    elif args.architecture == 'upmem':
        SimConfig.read_from_yaml('./config/upmem.yaml')
        Codegen = upmem
    elif args.architecture == 'dimmining':
        SimConfig.read_from_yaml('./config/dimmining.yaml')
        Codegen = dimmining
    # elif args.architecture == 'axchannel':
    #     SimConfig.read_from_yaml('./config/axchannel.yaml')
    #     Codegen = axchannel
    else:
        raise ValueError(f"Unknown hw architecture: {args.architecture}")
    # set NDP level
    if args.architecture == 'dimmining':
        SimConfig.pu_level = LEVEL.RA
    else:
        SimConfig.pu_level = LEVEL.DE
    # check sim config
    print("sim config: ", SimConfig.__dict__, file=log_file)

    # start tick
    start_tick = time.time()

    """
    NOTE: 1. Get design space
    """
    # A. get hw partition space
    design_space = []
    partition_tool = Partition(require_power_of_2 = args.po2)
    partition_space = partition_tool.get_partition_space_mm(mm_size)
    # filter hw partition space
    filtered_partition_space = partition_tool.choose_from_partition_space_mm(partition_space)
    if not args.allow_under_ultize:
        partition_space = filtered_partition_space
    print(len(partition_space), file=log_file)
    # partition space + mapping space = design space
    for index in tqdm.tqdm(range(len(partition_space))):
        compute_level, pu_num, partition = partition_space[index]
        # B. get mem partition space
        simd_k, mkl_Input_to_row, simd_l, ml_Out_to_row \
            = partition_tool.mem_partition_mm(mm_size, partition)
        # report design space
        for input_choice in reversed(mkl_Input_to_row):
            if args.architecture == 'aim':
                if ml_Out_to_row:
                    output_choice = ml_Out_to_row[0]
                    design_space.append((compute_level, pu_num, partition, simd_k, input_choice, simd_l, output_choice))
                continue
            for output_choice in reversed(ml_Out_to_row):
                # print(f"simd_k: {simd_k}, mkl_Input_to_row: {mkl_Input_to_row}, simd_l: {simd_l}, ml_Out_to_row: {ml_Out_to_row}")
                design_space.append((compute_level, pu_num, partition, simd_k, input_choice, simd_l, output_choice))
    print(f"constructed design space, size = {len(design_space)}")
    get_info = Codegen(require_power_of_2 = args.po2).inst_info

    # create ./dump/csv/ if not exist
    os.makedirs(os.path.dirname(outcsv_name), exist_ok=True)
    csvfile = open(outcsv_name, 'w', newline='')
    writer = csv.writer(csvfile)
    # if SimConfig.pu_level == LEVEL.DE:
    #     writer.writerow(["compute_level", "pu_num", "partition", "channel_used", "rank_used", "device_used", "pu_used", "simd_k", "mkl_Input_to_row", "simd_l", "ml_Out_to_row", "predict_result", "sim_result"] + get_info)
    # elif SimConfig.pu_level == LEVEL.RA:
    #     writer.writerow(["compute_level", "pu_num", "partition", "channel_used", "rank_used", "pu_used", "simd_k", "mkl_Input_to_row", "simd_l", "ml_Out_to_row", "predict_result", "sim_result"] + get_info)
    # else:
    #     raise ValueError(f"Unknown pu_level: {SimConfig.pu_level}")
    
    """
    NOTE: 2. get the baseline result
    """
    print("processing baseline")
    baseline = None
    if args.architecture in ['aim', 'aim8']:
        # baseline = design_space[0]
        for compute_level, pu_num, partition, simd_k, mkl_Input_to_row, simd_l, ml_Out_to_row in design_space:
            # if mul([part[0] for part in partition]) * partition[3][0] * partition[3][1] * mkl_Input_to_row[0] * mkl_Input_to_row[2] == 1:
            if partition[3][0] * partition[3][1] == 1 and (mkl_Input_to_row[0][0] * mkl_Input_to_row[0][2] * mkl_Input_to_row[0][3] == 1 or mkl_Input_to_row[0][1]*simd_k==mm_size[1]):
                baseline = compute_level, pu_num, partition, simd_k, mkl_Input_to_row, simd_l, ml_Out_to_row
                print(f"baseline strategy: {baseline}", file=log_file)
                break
    elif args.architecture == 'hbm-pim':
        for compute_level, pu_num, partition, simd_k, mkl_Input_to_row, simd_l, ml_Out_to_row in design_space:
            # if mul([part[0] for part in partition]) * partition[3][0] * partition[3][1] == 1 and mkl_Input_to_row[1] == 8:
            if partition[3][0] * partition[3][1] == 1 and mkl_Input_to_row[0][1] == 8:
                baseline = compute_level, pu_num, partition, simd_k, mkl_Input_to_row, simd_l, ml_Out_to_row
                print(f"baseline strategy: {baseline}", file=log_file)
                break
    elif args.architecture == 'upmem':
        for compute_level, pu_num, partition, simd_k, mkl_Input_to_row, simd_l, ml_Out_to_row in design_space:
            # if mul([part[0] for part in partition]) * partition[3][0] * partition[3][1] * mkl_Input_to_row[0] * mkl_Input_to_row[2] == 1:
            if partition[3][0] * partition[3][1] == 1 and (mkl_Input_to_row[0][0] * mkl_Input_to_row[0][2] * mkl_Input_to_row[0][3] == 1 or mkl_Input_to_row[0][1]*simd_k==mm_size[1]):
                baseline = compute_level, pu_num, partition, simd_k, mkl_Input_to_row, simd_l, ml_Out_to_row
                print(f"baseline strategy: {baseline}", file=log_file)
                break
        if baseline == None:
            for compute_level, pu_num, partition, simd_k, mkl_Input_to_row, simd_l, ml_Out_to_row in design_space:
            # if mul([part[0] for part in partition]) * partition[3][0] * partition[3][1] * mkl_Input_to_row[0] * mkl_Input_to_row[2] == 1:
                if (mkl_Input_to_row[0][0] * mkl_Input_to_row[0][2] * mkl_Input_to_row[0][3] == 1 or mkl_Input_to_row[0][1]*simd_k==mm_size[1]):
                    baseline = compute_level, pu_num, partition, simd_k, mkl_Input_to_row, simd_l, ml_Out_to_row
                    print(f"baseline strategy: {baseline}", file=log_file)
                    break   
    elif args.architecture == 'dimmining':
        for compute_level, pu_num, partition, simd_k, mkl_Input_to_row, simd_l, ml_Out_to_row in design_space:
            # if mul([part[0] for part in partition]) * partition[2][0] * partition[2][1] == 1 and (mkl_Input_to_row[0] * mkl_Input_to_row[2] == 1 or mkl_Input_to_row[1]*simd_k==mm_size[1]):
            if partition[2][0] * partition[2][1] == 1 and (mkl_Input_to_row[0][0] * mkl_Input_to_row[0][2] * mkl_Input_to_row[0][3] == 1 or mkl_Input_to_row[0][1]*simd_k==mm_size[1]):
                baseline = compute_level, pu_num, partition, simd_k, mkl_Input_to_row, simd_l, ml_Out_to_row
                print(f"baseline strategy: {baseline}", file=log_file)
                break

    if baseline == None: # corner case
        baseline = design_space[0]
        print(f"baseline strategy: {baseline}", file=log_file)
    compute_level, pu_num, partition, simd_k, mkl_Input_to_row, simd_l, ml_Out_to_row = baseline
    # A. hw mapping
    mapping_tool = Mapping(require_power_of_2 = args.po2)
    hw_id_list = mapping_tool.assign_hw(partition)
    # B. dram mapping
    input_bank, input_row_offset, \
    weight_bank, weight_row_offset, \
    output_bank, output_row_offset \
        = mapping_tool.assign_dram(pu_num, mkl_Input_to_row, ml_Out_to_row, partition) 
    # C. scheduling: TODO
    # D. Codegen
    codegen_tool = Codegen(require_power_of_2 = args.po2)
    codegen_tool.set_gen()
    gen_code, inst_count, predict_result = \
    codegen_tool.codegen(args.workload, compute_level, pu_num, partition,
                simd_k, mkl_Input_to_row, simd_l, ml_Out_to_row,
                hw_id_list, (input_bank, input_row_offset,
                            weight_bank, weight_row_offset,
                            output_bank, output_row_offset),
                            cmd_threshold=0)
    baseline_inst_num, baseline_pu_dram_num, baseline_host_dram_num, baseline_row_change_num = codegen_tool.get_matrix()
    # E. simulation
    baseline_sim_result = sim(gen_code, silent=True, filename=None)
    # 保存结果，控制excel文件的大小
    if SimConfig.pu_level == LEVEL.DE:
        hw_utilization_info = [mul(partition[i]) for i in range(4)]
    elif SimConfig.pu_level == LEVEL.RA:
        hw_utilization_info = [mul(partition[i]) for i in range(3)]
    else:
        raise ValueError(f"Unknown pu_level: {SimConfig.pu_level}")
    # writer.writerow([compute_level, pu_num, partition] + hw_utilization_info + [simd_k, mkl_Input_to_row, simd_l, ml_Out_to_row, predict_result, baseline_sim_result] + inst_count)
    csvfile.flush()
    print(f"baseline result: {baseline_sim_result}", file=log_file)

    """
    NOTE: 3. search for the optimal: predictor-aided / brute force
    """
    if args.quicksearch: # use predictor-aided search
        print(f"quick search, size = {len(design_space)}, topk = {args.topk}, cmdthre = {args.cmdthre}")
        predict_result_list = []
        min_codelen = 0
        # 1st round: get the topk
        print("1st round: get the topk")
        for index in tqdm.tqdm(range(len(design_space))):
            thre = min_codelen * args.cmdthre
            # tqdm.tqdm.write(f"thre: {thre}")
            design_point = design_space[index]
            compute_level, pu_num, partition, simd_k, mkl_Input_to_row, simd_l, ml_Out_to_row = design_point
            # A. hw mapping
            mapping_tool = Mapping(require_power_of_2 = args.po2)
            hw_id_list = mapping_tool.assign_hw(partition)
            # B. dram mapping
            input_bank, input_row_offset, \
            weight_bank, weight_row_offset, \
            output_bank, output_row_offset \
                = mapping_tool.assign_dram(pu_num, mkl_Input_to_row, ml_Out_to_row, partition)
            # C. scheduling: TODO
            # D. Codegen
            codegen_tool = Codegen(require_power_of_2 = args.po2)
            gen_code, inst_count, predict_result = \
            codegen_tool.codegen(args.workload, compute_level, pu_num, partition,
                        simd_k, mkl_Input_to_row, simd_l, ml_Out_to_row,
                        hw_id_list, (input_bank, input_row_offset,
                                    weight_bank, weight_row_offset,
                                    output_bank, output_row_offset),
                                    cmd_threshold=thre)
            if gen_code is not None:
                predict_result_list.append((predict_result, design_point))
                if len(predict_result_list) > args.topk:
                    predict_result_list = sorted(predict_result_list, key=lambda x: x[0])
                    predict_result_list = predict_result_list[:args.topk]

        design_space = [x[1] for x in predict_result_list]
        for predict_result in predict_result_list:
            print(f"predict_result: {predict_result}", file=log_file)
        
        if args.nosim:
            best_predict_result = predict_result_list[0][0]
            for i in range(len(predict_result_list)):
                if predict_result_list[i][0] > best_predict_result:
                    break
            design_space = design_space[:i]
        
        # 2nd round: get the best
        print("2nd round: get the best")
        best_design = []
        best_result = inf
        for index in tqdm.tqdm(range(len(design_space))):
            design_point = design_space[index]
            compute_level, pu_num, partition, simd_k, mkl_Input_to_row, simd_l, ml_Out_to_row = design_point
            # hw mapping
            mapping_tool = Mapping(require_power_of_2 = args.po2)
            hw_id_list = mapping_tool.assign_hw(partition)
            # dram mapping
            input_bank, input_row_offset, \
            weight_bank, weight_row_offset, \
            output_bank, output_row_offset \
                = mapping_tool.assign_dram(pu_num, mkl_Input_to_row, ml_Out_to_row, partition)
            # C. scheduling
            # D. Codegen
            codegen_tool = Codegen(require_power_of_2 = args.po2)
            codegen_tool.set_gen()
            gen_code, inst_count, predict_result = \
            codegen_tool.codegen(args.workload, compute_level, pu_num, partition,
                        simd_k, mkl_Input_to_row, simd_l, ml_Out_to_row,
                        hw_id_list, (input_bank, input_row_offset,
                                    weight_bank, weight_row_offset,
                                    output_bank, output_row_offset),
                                    cmd_threshold=0)
            if gen_code is not None:
                # E. simulation
                sim_result = sim(gen_code, silent=True, filename=None)
                inst_num, pu_dram_num, host_dram_num, row_change_num = codegen_tool.get_matrix()
                # csv output
                if SimConfig.pu_level == LEVEL.DE:
                    hw_utilization_info = [mul(partition[i]) for i in range(4)]
                elif SimConfig.pu_level == LEVEL.RA:
                    hw_utilization_info = [mul(partition[i]) for i in range(3)]
                else:
                    raise ValueError(f"Unknown pu_level: {SimConfig.pu_level}")
                if sim_result < best_result or (args.nosim and sim_result > best_result):
                    best_result = sim_result
                    best_design = [inst_num, pu_dram_num, host_dram_num, row_change_num]
                # elif sim_result == best_result:
                #     best_design.append(design_point)
        writer.writerow(
            ["name", "workload", "best_result", "baseline_result", "speedup"]
        )      
        writer.writerow(
            [args.name, args.workload, best_result, baseline_sim_result, round(baseline_sim_result/best_result, 2)]
        )
        # writer.writerow([compute_level, pu_num, partition] + hw_utilization_info + [simd_k, mkl_Input_to_row, simd_l, ml_Out_to_row, predict_result_list[index][0], sim_result] + inst_count)
        csvfile.flush()
    
    else: # TODO: no need to provide the detail result, but should provide the header
        print(f"search using brute force, size = {len(design_space)}, cmdthre = {args.cmdthre}")
        writer.writerow(
            ["compilation_strategy", "predict_result", "sim_result"]
        )
        min_codelen = {}
        best_design = []
        best_result = inf
        for index in tqdm.tqdm(range(len(design_space))):
        # for design_point in design_space:
            # decode
            design_point = design_space[index]
            compute_level, pu_num, partition, simd_k, mkl_Input_to_row, simd_l, ml_Out_to_row = design_point
            # print(f"design_point: {design_point}")
            # A. hw mapping
            mapping_tool = Mapping(require_power_of_2 = args.po2)
            hw_id_list = mapping_tool.assign_hw(partition)
            # B. dram mapping
            input_bank, input_row_offset, \
            weight_bank, weight_row_offset, \
            output_bank, output_row_offset \
                = mapping_tool.assign_dram(pu_num, mkl_Input_to_row, ml_Out_to_row, partition)
            # C. scheduling: TODO
            # D. Codegen
            if pu_num not in min_codelen.keys():
                thre = 0
            else:
                thre = args.cmdthre * min_codelen[pu_num]
            codegen_tool = Codegen(require_power_of_2 = args.po2)
            _, _, predict_result = \
            codegen_tool.codegen(args.workload, compute_level, pu_num, partition,
                        simd_k, mkl_Input_to_row, simd_l, ml_Out_to_row,
                        hw_id_list, (input_bank, input_row_offset,
                                    weight_bank, weight_row_offset,
                                    output_bank, output_row_offset),
                                    cmd_threshold=thre)
            codegen_tool = Codegen(require_power_of_2 = args.po2)
            codegen_tool.set_gen()
            gen_code, inst_count, _ = \
            codegen_tool.codegen(args.workload, compute_level, pu_num, partition,
                        simd_k, mkl_Input_to_row, simd_l, ml_Out_to_row,
                        hw_id_list, (input_bank, input_row_offset,
                                    weight_bank, weight_row_offset,
                                    output_bank, output_row_offset),
                                    cmd_threshold=thre)
            if gen_code is not None:
                if pu_num not in min_codelen.keys() or predict_result < min_codelen[pu_num]:
                    min_codelen[pu_num] = predict_result
                # E. simulation
                inst_num, pu_dram_num, host_dram_num, row_change_num = codegen_tool.get_matrix()
                # NOTE: 
                if args.refpred:
                    sim_result = 0
                else:
                    sim_result = sim(gen_code, silent=True, filename=None)
                # sim_result = predict_result
                if SimConfig.pu_level == LEVEL.DE:
                    hw_utilization_info = [mul(partition[i]) for i in range(4)]
                elif SimConfig.pu_level == LEVEL.RA:
                    hw_utilization_info = [mul(partition[i]) for i in range(3)]
                else:
                    raise ValueError(f"Unknown pu_level: {SimConfig.pu_level}")
                if sim_result < best_result:
                    best_result = sim_result
                    best_design = [inst_num, pu_dram_num, host_dram_num, row_change_num]

                writer.writerow([str(
                    [compute_level, pu_num, partition] + hw_utilization_info + [simd_k, mkl_Input_to_row, simd_l, ml_Out_to_row]
                ), predict_result, sim_result])
                # writer.writerow([compute_level, pu_num, partition] + hw_utilization_info + [simd_k, mkl_Input_to_row, simd_l, ml_Out_to_row, predict_result, sim_result] + inst_count)
                csvfile.flush()

    end_tick = time.time()

    # output the best result and close csv file
    print(f"best_result: {best_result}", file=log_file)
    print(f"best_design: {best_design}", file=log_file)
    print(f"compile_time: {end_tick-start_tick}", file=log_file)
    if not args.refpred:
        print(f"speedup: {baseline_sim_result / best_result}", file=log_file)
    csvfile.close()

if __name__ == '__main__':
    main()

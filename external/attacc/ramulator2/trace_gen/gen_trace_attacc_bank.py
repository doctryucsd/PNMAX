import argparse
import math
import copy
import numpy as np

model = "gpt-3-175B"

# Testing command
# python3 gen_trace_attacc_bank.py -dh 128 -l 2048 -maxl 4096 -db 2 -o attacc_bank.trace

dhead = 128
max_L = 2048
data_size = 16 # FP 16

n_attacc = 8
max_n_hbm = 8
n_hbm = 5
n_channel = 16
n_pch = 2
n_rank = 2
n_bank = 4
n_bg = 4
n_row = pow(2, 14)
n_col = pow(2, 5)
prefetch_size = 32 # byte
n_mac = 16


def append_command(container, cmd):
  """Store a command for later emission."""
  container.append(cmd)


# Granularity size
HBM_GS = {}
HBM_GS['col']     = prefetch_size
HBM_GS['row']     = n_col * HBM_GS['col']
HBM_GS['ba']      = n_row * HBM_GS['row'] 
HBM_GS['bg']      = n_bank * HBM_GS['ba'] 
HBM_GS['rank']     = n_bg * HBM_GS['bg'] 
HBM_GS['pch']     = n_rank * HBM_GS['rank'] 
HBM_GS['ch']      = n_pch * HBM_GS['pch']
HBM_GS['hbm']     = n_channel * HBM_GS['ch']
HBM_GS['attacc']  = max_n_hbm * HBM_GS['hbm']


## --------------------------------------  HBM memory space -----------------------------------------##
## ------|  legacy CH  |  pCH  |  rank  | BG | BA |  row index  |  column index  |  access granularity  |------ ##
## bits  |     4       |   1   |   1   | 2  | 2  |     14      |        5       |          5           |       ##

## ----------------------------  Commands -------------------------------##
## MACAB: 8tCK (tCCDLx 2)
##  WRGB: 4tCK (write to SRAM not DRAM)
##  MVSB: 4tCK
##  MVGB: 4tCK
##  SFM: 16tCK (for L = 256)

cmd_score_wrgb   = []
cmd_score_mac    = []
cmd_score_mvsb   = []
cmd_sfm          = []
cmd_context_mvgb  = []
cmd_context_mac  = []
cmd_context_mvsb = []

valid_channels = []

#! Not modeling input loading and reduction cost
def cmd_list_reset():
  cmd_score_wrgb   = []
  cmd_score_mac    = []
  cmd_score_mvsb   = []
  cmd_sfm          = []
  cmd_context_mvgb = []
  cmd_context_mac  = []
  cmd_context_mvsb = []

  valid_channel = []

def Attention(L, key_addr, val_addr, itr, valid_channel = n_channel):
  cmd_score_wrgb.append([])
  cmd_score_mac.append([])
  cmd_score_mvsb.append([])

  valid_channels.append(valid_channel);

  def score_cpvec(addr_offset, L):
    ## (pCH) C, C, R, R (MAC)
    ## write input vector to gemv buffer
    # number of partition = (R parallel units)

    # Data broadcasting for pch, rank, bg, and ba
    for ba_idx in range(n_bank): # number of partitions
      for col_idx in range(math.ceil(dhead / n_bank / n_mac)):
        for lch in range(math.ceil(valid_channel)):
          # GEMV buffer address, col granularity = 1
          addr = addr_offset + lch * HBM_GS['ch'] + ba_idx * HBM_GS['ba'] + col_idx
          hex_addr = hex(addr)[2:]
          cmd = "PIM_WR_GB 0x{0:0>8}".format(hex_addr)
          append_command(cmd_score_wrgb[itr], cmd)

  def score_mac(addr_offset, L):
    ## (pCH) C, C, R, R (MAC)
    # MAC and move output vector to softmax buffer
    ## Vector (1 x k) x Matrix (k x n) multiplication
    ## GEMV unit = adder tree mode
    tiles_per_n = math.ceil(dhead / n_bank / n_mac)
    max_score_tiles = math.ceil(L / n_pch / n_rank / n_bg)
    flush_period = 16
    for n_idx in range(max_score_tiles):# 16 
      cmd_score_mac[itr].append([])
      for k_idx in range(tiles_per_n): # 2
        idx = k_idx + n_idx * tiles_per_n 

        # All bank command (legacy channel)
        for lch in range(math.ceil(valid_channel)):
          addr = addr_offset + lch * HBM_GS['ch'] + idx * HBM_GS['col']
          hex_addr = hex(addr)[2:]
          cmd = "PIM_MAC_AB 0x{0:0>8}".format(hex_addr)
          append_command(cmd_score_mac[itr][-1], cmd)
         ## parallelization

      ## MVSB command (Move to Softmax buffer) 
      ## A output element is generated for every n_idx
      if n_idx % flush_period == flush_period - 1 or n_idx == max_score_tiles - 1:
        cmd_score_mvsb[itr].append([])
        for bg_idx in range(n_bg):   
          for rank in range(n_rank):
            for lch in range(math.ceil(valid_channel)):
              bank_addr = addr_offset + lch * HBM_GS['ch'] + rank * HBM_GS['rank'] + \
                          bg_idx * HBM_GS['bg']
              hex_addr = hex(bank_addr)[2:]
              cmd = "PIM_MV_SB 0x{0:0>8}".format(hex_addr)
              append_command(cmd_score_mvsb[itr][-1], cmd)

  score_cpvec(key_addr, L)

  score_mac(key_addr, L)


# n_head and n_req = n_req per a HBM 
def run_attention(dhead, n_head_per_hbm, L, trace_file_name):
  partition_size = math.ceil(max_L * dhead / (n_pch * n_rank * n_bg * n_bank))
  head_offset = partition_size
  v_offset = pow(2, 23) 
  

  cmd_list_reset()
  ##-- Generate Commands --##
  num_itr = math.ceil(n_head_per_hbm / (n_channel))
  for itr in range(num_itr):
    remainder = 0
    if (n_head_per_hbm / ((itr+1) * n_channel) < 1):
      remainder = n_head_per_hbm % n_channel
    key_addr = itr * partition_size 
    val_addr = key_addr + v_offset
    if remainder == 0:
      Attention(L, key_addr, val_addr, itr)
    else:
      Attention(L, key_addr, val_addr, itr, remainder)


  ##-- Ovelapping Commands --##
  barrier = []
  for lch in range(n_channel):
    addr = lch * HBM_GS['ch']
    hex_addr = hex(addr)[2:]
    cmd = "PIM_BARRIER 0x{0:0>8}".format(hex_addr)
    append_command(barrier, cmd)

  total_cmd = []
  for i in range(0, num_itr -1, 2):

    # Head0: Score
      ## WRGB
    total_cmd.extend(cmd_score_wrgb[i])
      ## dummy MAC
    if i == 0:
      for j in range(valid_channels[i]):
        total_cmd.append(cmd_score_mac[i][0][j])
      ## BARRIER
    total_cmd.extend(barrier)

    length = math.ceil(L/n_pch/n_rank/n_bg/16)
    for j in range(0, length+1):
      ## MAC (Head0)
      if not j == length:
        stride = 16;
        for k in range(stride):
          if (j*stride+k) >= len(cmd_score_mac[i]):
            break;
          total_cmd.extend(cmd_score_mac[i][j*stride+k])
      ## MVSB (Head0)
      if not j == 0:
        total_cmd.extend(cmd_score_mvsb[i][j-1])
      ## WRGB (Head1)
      if not j == length:
        stride = int(n_bank*math.ceil(dhead /n_bank /n_mac)*math.ceil(valid_channels[i+1])/length);
        for k in range(stride):
          if (j*stride+k) >= len(cmd_score_wrgb[i+1]):
            break;
          total_cmd.append(cmd_score_wrgb[i+1][j*stride + k])
      ## BARRIER
      if not j == length:
        total_cmd.extend(barrier)

  if num_itr % 2 != 0:
    i = num_itr - 1

    # Score
      ## WRGB
    total_cmd.extend(cmd_score_wrgb[i])
      ## BARRIER
    total_cmd.extend(barrier)

    length = math.ceil(L/n_pch/n_rank/n_bg/16)
    for j in range(0, length+1):
      ## MAC
      if not j == length:
        stride = 16;
        for k in range(stride):
          if (j*stride+k) >= len(cmd_score_mac[i]):
            break;
          total_cmd.extend(cmd_score_mac[i][j*stride+k])
      ## MVSB
      if not j == 0:
        total_cmd.extend(cmd_score_mvsb[i][j-1])
      ## BARRIER
      if not j == length:
        total_cmd.extend(barrier)

  trace_file = open(trace_file_name, 'w')
  for cmd in total_cmd:
    trace_file.write(cmd + "\n")
  trace_file.close()

def main():
  global dhead, max_L, data_size, n_mac


  parser = argparse.ArgumentParser(description="Output path and operation infos",
                               formatter_class=argparse.ArgumentDefaultsHelpFormatter)
 
  parser.add_argument("-dh", "--dhead", type=int, default=128, 
                      help="dhead, default= 128")
  # Do not exceed 16 for simplicity
  parser.add_argument("-nh", "--nhead", type=int, default=16,
                      help="Number of heads, default=16")
  parser.add_argument("-l", "--seqlen", type=int, default=2048,
                      help="Sequence length L, default= 2048")
  parser.add_argument("-maxl", "--maxlen", type=int, default=4096, 
                      help="maximum L, default= 4096")
  parser.add_argument("-db", "--dbyte", type=int, default=2, 
                      help="data type (B), default= 2")
  parser.add_argument("-o", "--output", type=str, default="attacc_bank.trace", 
                      help="output path")

  args = parser.parse_args()

  dhead = args.dhead
  max_L = args.maxlen
  L = args.seqlen
  n_head_per_hbm = args.nhead 

  data_size = args.dbyte
  n_mac = int(HBM_GS['col'] / data_size)

  print("------   Make a trace of bank-level AttAcc   ------")

  args_dict = vars(args)
  print("All Arguments:")
  for key, value in args_dict.items():
      print(f"     {key}: {value}")
  print("---------------------------------------------------")

  run_attention(dhead, n_head_per_hbm, L, args.output)



if __name__ == "__main__":
  main()

# pruning, use predictor
python -OO compile_predictor.py -A aim -S 4096 4096 4096 1 -O 1 -Q -K 30
# pruning, no predictor
python -OO compile_predictor.py -A aim -S 4096 4096 4096 1 -O 2
# no pruning, use predictor
python -OO compile_predictor.py -A aim -S 4096 4096 4096 1 -O 3 -Q -K 30 -UU

python -OO compile_predictor.py -A upmem -S 1 4096 4096 1 -O upmem_pred

python -OO compile_predictor.py -A axdimm -S 1 4096 4096 1 -O axdimm_pred

python -OO compile_predictor.py -A axdimm -S 1 4096 4096 1 -O axdimm_pred_1 -RP

python -OO compile_predictor.py -A hbm-pim -S 1 4096 4096 1 -O hbmpim_pred

# -NS
python -OO compile_predictor.py -A aim      -S 4096 4096 4096 1 -O aim_NS       -Q -K 30 -NS
python -OO compile_predictor.py -A hbm-pim  -S 4096 4096 4096 1 -O hbmpim_NS    -Q -K 30 -NS
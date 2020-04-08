# ######################
# #    cartesian 468
# ######################

# CUDA_VISIBLE_DEVICES=2 \
#     python models/TOF_NET/train_tofnet.py \
#         -nc 6 -nd 5 -nf 64 \
#         -mask-style cartesian_1d -cf 0.08 -acc 4 -mf 16 \
#         -exp-dir exp_dir/TOF_NET/Cartesian/D5C6_CH64_MF16_X4/ \
#         -num-epochs 200 -lr-step 175 -eval \
&& sleep 3m && CUDA_VISIBLE_DEVICES=2 \
    python models/TOF_NET/train_tofnet.py \
        -nc 6 -nd 5 -nf 64 \
        -mask-style cartesian_1d -cf 0.06 -acc 6 -mf 16 \
        -exp-dir exp_dir/TOF_NET/Cartesian/D5C6_CH64_MF16_X6/ \
        -num-epochs 200 -lr-step 175 -eval \
# && sleep 3m && CUDA_VISIBLE_DEVICES=2 \
#     python models/TOF_NET/train_tofnet.py \
#         -nc 6 -nd 5 -nf 64 \
#         -mask-style cartesian_1d -cf 0.04 -acc 8 -mf 16 \
#         -exp-dir exp_dir/TOF_NET/Cartesian/D5C6_CH64_MF16_X8/ \
#         -num-epochs 200 -lr-step 175 -eval

# CUDA_VISIBLE_DEVICES=2 \
#     python models/TOF_NET/train_tofnet.py \
#         -nc 6 -nd 5 -nf 64 \
#         -mask-style cartesian_1d -cf 0.06 -acc 6 -mf 16 \
#         -exp-dir exp_dir/TOF_NET/Cartesian/D5C6_CH64_MF16_X6/ \
#         -num-epochs 200 -lr-step 175 -eval \
# && CUDA_VISIBLE_DEVICES=2 \
#     python models/TOF_NET/train_tofnet.py \
#         -nc 6 -nd 5 -nf 64 \
#         -mask-style poisson_2d -cf 0.04 -acc 9.754 -mf 16 \
#         -exp-dir exp_dir/TOF_NET/Poisson/D5C6_CH64_MF16_X8_Poisson \
#         -num-epochs 200 -lr-step 175 -eval
# ###########################
# #    Gap 234 & Poisson
# ###########################

CUDA_VISIBLE_DEVICES=2 \
    python models/TOF_NET/train_tofnet.py \
        -nc 6 -nd 5 -nf 64 \
        -mask-style cartesian_1d -cf 0.06 -acc 6 -mf 16 \
        -exp-dir exp_dir/TOF_NET/GAP/D5C6_CH64_MF16_X6_g2/ \
        -num-epochs 200 -lr-step 175 -gap 2 -eval \
&& sleep 3m && CUDA_VISIBLE_DEVICES=0 \
    python models/TOF_NET/train_tofnet.py \
        -nc 6 -nd 5 -nf 64 \
        -mask-style cartesian_1d -cf 0.06 -acc 6 -mf 16 \
        -exp-dir exp_dir/TOF_NET/GAP/D5C6_CH64_MF16_X6_g3/ \
        -num-epochs 200 -lr-step 175  -gap 3 -eval \
&& sleep 3m && CUDA_VISIBLE_DEVICES=3 \
    python models/TOF_NET/train_tofnet.py \
        -nc 6 -nd 5 -nf 64 \
        -mask-style cartesian_1d -cf 0.06 -acc 6 -mf 16 \
        -exp-dir exp_dir/TOF_NET/GAP/D5C6_CH64_MF16_X6_g4_0/ \
        -num-epochs 200 -lr-step 175  -gap 4 -eval

# ###########################
# #          ITER
# ###########################
CUDA_VISIBLE_DEVICES=2 \
    python models/TOF_NET/train_tofnet.py \
        -nc 1 -nd 5 -nf 64 \
        -mask-style cartesian_1d -cf 0.04 -acc 8 -mf 16 \
        -exp-dir exp_dir/TOF_NET/ITER/X8/D5C1_CH64_MF16_X8/ \
        -num-epochs 200 -lr-step 175 -eval \
&& CUDA_VISIBLE_DEVICES=2 \
    python models/TOF_NET/train_tofnet.py \
        -nc 2 -nd 5 -nf 64 \
        -mask-style cartesian_1d -cf 0.04 -acc 8 -mf 16 \
        -exp-dir exp_dir/TOF_NET/ITER/X8/D5C2_CH64_MF16_X8/ \
        -num-epochs 200 -lr-step 175 -eval \
&& CUDA_VISIBLE_DEVICES=2 \
    python models/TOF_NET/train_tofnet.py \
        -nc 3 -nd 5 -nf 64 \
        -mask-style cartesian_1d -cf 0.04 -acc 8 -mf 16 \
        -exp-dir exp_dir/TOF_NET/ITER/X8/D5C3_CH64_MF16_X8/ \
        -num-epochs 200 -lr-step 175 -eval \
&& CUDA_VISIBLE_DEVICES=2 \
    python models/TOF_NET/train_tofnet.py \
        -nc 4 -nd 5 -nf 64 \
        -mask-style cartesian_1d -cf 0.04 -acc 8 -mf 16 \
        -exp-dir exp_dir/TOF_NET/ITER/X8/D5C4_CH64_MF16_X8/ \
        -num-epochs 200 -lr-step 175 -eval \
&& CUDA_VISIBLE_DEVICES=2 \
    python models/TOF_NET/train_tofnet.py \
        -nc 5 -nd 5 -nf 64 \
        -mask-style cartesian_1d -cf 0.04 -acc 8 -mf 16 \
        -exp-dir exp_dir/TOF_NET/ITER/X8/D5C5_CH64_MF16_X8/ \
        -num-epochs 200 -lr-step 175 -eval
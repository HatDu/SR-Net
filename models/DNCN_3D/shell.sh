# CUDA_VISIBLE_DEVICES=1 \
#     python models/DNCN_3D/run_dncn3d.py \
#         -ckpt exp_dir/DNCN_3D/Gaps/D5C5_CH16_MF32_X6_gap4/best_model.pt \
#         -mask-style cartesian_1d -cf 0.06 -acc 6 -gap 4\
#         -out-dir exp_dir/DNCN_3D/Gaps/D5C5_CH16_MF32_X6_gap4/infer/

# CUDA_VISIBLE_DEVICES=0 \
#     python models/DNCN_3D/train_dncn3d.py \
#         -nc 5 -nd 5 -nf 16 \
#         -mask-style cartesian_1d -cf 0.06 -acc 6 -mf 32 \
#         -exp-dir exp_dir/DNCN_3D/D5C5_CH16_MF32_X6_gap4/ \
#         -num-epochs 800 -lr-step 780 -eval -gap 4

# python utils/eval.py -pred-dir exp_dir/DNCN_3D/Gaps/D5C5_CH16_MF32_X6_gap4/infer/ \
#     -out-dir exp_dir/DNCN_3D/Gaps/D5C5_CH16_MF32_X6_gap4/
# ! python utils/eval_2d.py --pred-dir exp_dir/TOF_NET/Cartesian/D5C5_CH32_MF32_X6/infer/ \
#     --out-path exp_dir/TOF_NET/Cartesian/D5C5_CH32_MF32_X6/

# ######################
# #    cartesian 468
# ######################

CUDA_VISIBLE_DEVICES=0 \
    python models/DNCN_3D/train_dncn3d.py \
        -nc 5 -nd 6 -nf 64 \
        -mask-style cartesian_1d -cf 0.08 -acc 4 -mf 16 \
        -exp-dir exp_dir/TOF_NET/Cartesian/D5C6_CH64_MF16_X4/ \
        -num-epochs 200 -lr-step 175 -eval \
&& sleep 3m && CUDA_VISIBLE_DEVICES=0 \
    python models/DNCN_3D/train_dncn3d.py \
        -nc 5 -nd 6 -nf 64 \
        -mask-style cartesian_1d -cf 0.06 -acc 6 -mf 16 \
        -exp-dir exp_dir/TOF_NET/Cartesian/D5C6_CH64_MF16_X6/ \
        -num-epochs 200 -lr-step 175 -eval \
&& sleep 3m && CUDA_VISIBLE_DEVICES=0 \
    python models/DNCN_3D/train_dncn3d.py \
        -nc 5 -nd 6 -nf 64 \
        -mask-style cartesian_1d -cf 0.04 -acc 8 -mf 16 \
        -exp-dir exp_dir/TOF_NET/Cartesian/D5C6_CH64_MF16_X8/ \
        -num-epochs 200 -lr-step 175 -eval \
&& sleep 3m && CUDA_VISIBLE_DEVICES=0 \
    python models/DNCN_3D/train_dncn3d.py \
        -nc 5 -nd 7 -nf 64 \
        -mask-style cartesian_1d -cf 0.06 -acc 6 -mf 16 \
        -exp-dir exp_dir/TOF_NET/Cartesian/D5C6_CH64_MF16_X8/ \
        -num-epochs 200 -lr-step 175 -eval

# ###########################
# #    Gap 234 & Poisson
# ###########################

CUDA_VISIBLE_DEVICES=0 \
    python models/DNCN_3D/train_dncn3d.py \
        -nc 5 -nd 6 -nf 64 \
        -mask-style cartesian_1d -cf 0.06 -acc 6 -mf 16 \
        -exp-dir exp_dir/DNCN_3D/GAP/D5C6_CH64_MF16_X6_g2/ \
        -num-epochs 200 -lr-step 175 -gap 2 \
&& sleep 3m && CUDA_VISIBLE_DEVICES=0 \
    python models/DNCN_3D/train_dncn3d.py \
        -nc 5 -nd 6 -nf 64 \
        -mask-style cartesian_1d -cf 0.06 -acc 6 -mf 16 \
        -exp-dir exp_dir/DNCN_3D/GAP/D5C6_CH64_MF16_X6_g3/ \
        -num-epochs 200 -lr-step 175  -gap 3 \
&& sleep 3m && CUDA_VISIBLE_DEVICES=0 \
    python models/DNCN_3D/train_dncn3d.py \
        -nc 5 -nd 6 -nf 64 \
        -mask-style cartesian_1d -cf 0.06 -acc 6 -mf 16 \
        -exp-dir exp_dir/DNCN_3D/GAP/D5C6_CH64_MF16_X6_g4/ \
        -num-epochs 200 -lr-step 175  -gap 4 
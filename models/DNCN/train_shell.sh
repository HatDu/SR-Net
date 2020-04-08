CUDA_VISIBLE_DEVICES=1 \
    python models/DNCN/train_dncn.py \
        -nc 5 -nd 5 -nf 64 \
        -mask-style cartesian_1d -cf 0.08 -acc 4 -mf 16 \
        -exp-dir exp_dir/DNCN/Cartesian/D5C5_CH64_MF16_X4 \
        -num-epochs 200 -lr-step 175 -eval \
&& CUDA_VISIBLE_DEVICES=1 \
    python models/DNCN/train_dncn.py \
        -nc 5 -nd 5 -nf 64 \
        -mask-style cartesian_1d -cf 0.06 -acc 6 -mf 16 \
        -exp-dir exp_dir/DNCN/Cartesian/D5C5_CH64_MF16_X6 \
        -num-epochs 200 -lr-step 175 -eval \
&& CUDA_VISIBLE_DEVICES=1 \
    python models/DNCN/train_dncn.py \
        -nc 5 -nd 5 -nf 64 \
        -mask-style cartesian_1d -cf 0.04 -acc 8 -mf 16 \
        -exp-dir exp_dir/DNCN/Cartesian/D5C5_CH64_MF16_X8 \
        -num-epochs 200 -lr-step 175 -eval \
&& CUDA_VISIBLE_DEVICES=2 \
    python models/DNCN/train_dncn.py \
        -nc 5 -nd 5 -nf 64 \
        -mask-style poisson_2d -cf 0.04 -acc 9.754 -mf 16 \
        -exp-dir exp_dir/DNCN/Cartesian/D5C5_CH64_MF16_X8_Poisson \
        -num-epochs 200 -lr-step 175 -eval \
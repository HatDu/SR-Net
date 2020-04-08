CUDA_VISIBLE_DEVICES=0 \
    python models/SR_NET/run_tofnet.py \
        -ckpt exp_dir/SR_NET/Cartesian/D5C6_CH64_MF16_X6/best_model.pt \
        -mask-style cartesian_1d -cf 0.06 -acc 6\
        -out-dir exp_dir/SR_NET/Cartesian/D5C6_CH64_MF16_X6/infer/
python utils/eval.py -pred-dir exp_dir/SR_NET/Cartesian/D5C6_CH64_MF16_X6/infer/ \
    -out-dir exp_dir/SR_NET/Cartesian/D5C6_CH64_MF16_X6/

python utils/eval_2d.py --pred-dir exp_dir/SR_NET/Cartesian/D5C6_CH64_MF16_X6/infer/ \
    --out-path exp_dir/SR_NET/Cartesian/D5C6_CH64_MF16_X6/
rm -r exp_dir/SR_NET/Cartesian/D5C6_CH64_MF16_X6/infer/

# CUDA_VISIBLE_DEVICES=0 \
#     python models/SR_NET/train_tofnet.py \
#         -nc 5 -nd 5 -nf 32 \
#         -mask-style cartesian_1d -cf 0.08 -acc 4 -mf 32 \
#         -exp-dir exp_dir/SR_NET/D5C7_CH32_MF32_X4/ \
#         -num-epochs 100 -lr-step 80

# CUDA_VISIBLE_DEVICES=3 \
#     python models/SR_NET/train_tofnet.py \
#         -nc 6 -nd 5 -nf 32 \
#         -mask-style cartesian_1d -cf 0.04 -acc 8 -mf 32 \
#         -exp-dir exp_dir/SR_NET/X4/D5C8_CH32_MF32_X8/ \
#         -num-epochs 100 -lr-step 80 -eval
# ######################
# #    cartesian 468
# ######################

CUDA_VISIBLE_DEVICES=0 \
    python models/CRNN/train_crnn.py \
        -nc 5 -nd 5 -nf 64 \
        -mask-style cartesian_1d -cf 0.08 -acc 4 -mf 15 \
        -exp-dir exp_dir/CRNN/Cartesian/D5C5_CH64_MF16_X4/ \
        -num-epochs 200 -lr-step 175 -eval \
&& sleep 3m && CUDA_VISIBLE_DEVICES=2 \
    python models/CRNN/train_crnn.py \
        -nc 5 -nd 5 -nf 64 \
        -mask-style cartesian_1d -cf 0.06 -acc 6 -mf 15 \
        -exp-dir exp_dir/CRNN/Cartesian/D5C5_CH64_MF16_X6/ \
        -num-epochs 200 -lr-step 175 -eval \
&& sleep 3m && CUDA_VISIBLE_DEVICES=2 \
    python models/CRNN/train_crnn.py \
        -nc 5 -nd 5 -nf 64 \
        -mask-style cartesian_1d -cf 0.04 -acc 8 -mf 15 \
        -exp-dir exp_dir/CRNN/Cartesian/D5C5_CH64_MF16_X8/ \
        -num-epochs 200 -lr-step 175 -eval \
&& CUDA_VISIBLE_DEVICES=3 \
    python models/CRNN/train_crnn.py \
        -nc 5 -nd 5 -nf 64 \
        -mask-style poisson_2d -cf 0.04 -acc 9.754 -mf 15 \
        -exp-dir exp_dir/CRNN/D5C5_CH64_MF16_Poisson_X8/ \
        -num-epochs 200 -lr-step 175 -eval

# ###########################
# #    Gap 234 & Poisson
# ###########################

CUDA_VISIBLE_DEVICES=1 \
    python models/CRNN/train_crnn.py \
        -nc 5 -nd 5 -nf 64 \
        -mask-style cartesian_1d -cf 0.06 -acc 6 -mf 15 \
        -exp-dir exp_dir/CRNN/GAP/D5C5_CH64_MF16_X6_g2/ \
        -num-epochs 200 -lr-step 175 -gap 2 -eval \
&& sleep 3m && CUDA_VISIBLE_DEVICES=1 \
    python models/CRNN/train_crnn.py \
        -nc 5 -nd 5 -nf 64 \
        -mask-style cartesian_1d -cf 0.06 -acc 6 -mf 15 \
        -exp-dir exp_dir/CRNN/GAP/D5C5_CH64_MF16_X6_g3/ \
        -num-epochs 200 -lr-step 175  -gap 3 -eval \
&& sleep 3m && CUDA_VISIBLE_DEVICES=1 \
    python models/CRNN/train_crnn.py \
        -nc 5 -nd 5 -nf 64 \
        -mask-style cartesian_1d -cf 0.06 -acc 6 -mf 15 \
        -exp-dir exp_dir/CRNN/GAP/D5C5_CH64_MF16_X6_g4/ \
        -num-epochs 200 -lr-step 175 -gap 4 -eval

# python models/CRNN/run_crnn.py \
#     -ckpt exp_dir/CRNN/D5C1_CH64_MF16_X4/best_model.pt \
#     -mask-style cartesian_1d -cf 0.08 -acc 4 \
#     -out-dir exp_dir/CRNN/D5C1_CH64_MF16_X4/infer/
# ! python utils/eval.py -pred-dir exp_dir/CRNN/D5C1_CH64_MF16_X4/infer/ \
#     -out-dir exp_dir/CRNN/D5C1_CH64_MF16_X4/
# ! python utils/eval_2d.py --pred-dir exp_dir/TOF_NET/Cartesian/D5C5_CH64_MF16_X6/infer/ \
#     --out-path exp_dir/TOF_NET/Cartesian/D5C5_CH64_MF16_X6/

sleep 5s \
&& ls -l \
&& sleep 2s

ls -l \


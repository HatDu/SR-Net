
CUDA_VISIBLE_DEVICES=1 \
    python models/KIKI/run_kiki.py \
        -ckpt exp_dir/KIKI/Cartesian/D5C5_CH64_MF16_X6/best_model.pt \
        -mask-style cartesian_1d -cf 0.06 -acc 6 \
        -out-dir exp_dir/KIKI/Cartesian/D5C5_CH64_MF16_X6/infer/

python utils/eval.py -pred-dir exp_dir/KIKI/Cartesian/D5C5_CH64_MF16_X6/infer/ \
    -out-dir exp_dir/KIKI/Cartesian/D5C5_CH64_MF16_X6/

python utils/eval_2d.py --pred-dir exp_dir/KIKI/Cartesian/D5C5_CH64_MF16_X6/infer/ \
    --out-path exp_dir/KIKI/Cartesian/D5C5_CH64_MF16_X6/

rm -r exp_dir/KIKI/Cartesian/D5C5_CH64_MF16_X6/infer/
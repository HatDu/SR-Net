CUDA_VISIBLE_DEVICES=1 python models/CRNN/run_crnn.py \
    -ckpt exp_dir/CRNN/Cartesian/D5C5_CH64_MF16_X8/best_model.pt \
    -mask-style cartesian_1d -cf 0.04 -acc 8 \
    -out-dir exp_dir/CRNN/Cartesian/D5C5_CH64_MF16_X8/infer/

python utils/eval.py -pred-dir exp_dir/CRNN/Cartesian/D5C5_CH64_MF16_X8/infer/ \
    -out-dir exp_dir/CRNN/Cartesian/D5C5_CH64_MF16_X8/

python utils/eval_2d.py --pred-dir exp_dir/CRNN/Cartesian/D5C5_CH64_MF16_X8/infer/ \
    --out-path exp_dir/CRNN/Cartesian/D5C5_CH64_MF16_X8/

rm -r exp_dir/CRNN/Cartesian/D5C5_CH64_MF16_X8/infer
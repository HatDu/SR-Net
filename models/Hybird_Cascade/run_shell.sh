### Run
CUDA_VISIBLE_DEVICES=3 \
    python models/Hybird_Cascade/run_hybird.py \
        -ckpt exp_dir/Hybird_Cascade/Cartesian/ikikii_CH16_MF32_X8_Poisson/best_model.pt \
        -mask-style poisson_2d -cf 0.04 -acc 9.754 \
        -out-dir exp_dir/Hybird_Cascade/Cartesian/ikikii_CH16_MF32_X8_Poisson/infer/
python utils/eval.py -pred-dir exp_dir/Hybird_Cascade/Cartesian/ikikii_CH16_MF32_X8_Poisson/infer/ \
    -out-dir exp_dir/Hybird_Cascade/Cartesian/ikikii_CH16_MF32_X8_Poisson/

python utils/eval_2d.py --pred-dir exp_dir/Hybird_Cascade/Cartesian/ikikii_CH16_MF32_X8_Poisson/infer/ \
    --out-path exp_dir/Hybird_Cascade/Cartesian/ikikii_CH16_MF32_X8_Poisson/
rm -r exp_dir/Hybird_Cascade/Cartesian/ikikii_CH16_MF32_X8_Poisson/infer/
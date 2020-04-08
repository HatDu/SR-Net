CUDA_VISIBLE_DEVICES=0 python models/DNCN_3D/run_dncn3d.py \
    -ckpt exp_dir/DNCN_3D/test2/best_model.pt \
    -mask-style cartesian_1d -cf 0.06 -acc 6 \
    -out-dir exp_dir/DNCN_3D/test2/infer/
python utils/eval.py -pred-dir exp_dir/DNCN_3D/test2/infer/ \
    -out-dir exp_dir/DNCN_3D/test2/
# python utils/eval_2d.py --pred-dir exp_dir/DNCN_3D/test2/infer/ \
#     --out-path exp_dir/DNCN_3D/test2/
rm -r exp_dir/DNCN_3D/test2/infer
# SR-Net
The Source code of SR-Net.

## Step 1: install packages and folders
```
# packages
pip install tensorboardX
pip install sigpy

# deformable convolutions
cd ./models/SR_NET/dcn
python setup.py develop
```

```
mkdir exp_dir
mkdir data
```

## Step 2: Prepare data

Downlad data from https://sites.google.com/view/calgary-campinas-dataset/home/download. Unzip the zip files into "data/" folder.

## Step 3: Train model
```
CUDA_VISIBLE_DEVICES=2 \
    python models/SR_NET/train_srnet.py \
        -nc 5 -nd 5 -nf 64 \
        -mask-style cartesian_1d -cf 0.06 -acc 6 -mf 16 \
        -exp-dir exp_dir/D5C5_CH64_MF16_X6/ \
        -num-epochs 200 -lr-step 175 -eval
```
## Step 4: Run and Eval model
### 4.1 Run model
```
CUDA_VISIBLE_DEVICES=0 \
    python models/SR_NET/run_srfnet.py \
        -ckpt exp_dir/D5C5_CH64_MF16_X6/best_model.pt \
        -mask-style cartesian_1d -cf 0.06 -acc 6\
        -out-dir exp_dir/D5C5_CH64_MF16_X6/infer/
```
### 4.2 Eval 
```
python utils/eval.py -pred-dir exp_dir/D5C5_CH64_MF16_X6/infer/ \
    -out-dir exp_dir/D5C5_CH64_MF16_X6/
```

### 4.3 Eval on each slice

```
python utils/eval_2d.py --pred-dir exp_dir/D5C5_CH64_MF16_X6/infer/ \
    --out-path exp_dir/D5C5_CH64_MF16_X6/
```
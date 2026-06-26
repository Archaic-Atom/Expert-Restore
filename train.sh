CUDA_VISIBLE_DEVICES=4 nohup python -u ./train.py --epochs 150 --batch_size 4 --lr 5e-4 --de_type denoise_15 denoise_25 denoise_50 derain dehaze --patch_size 128 --num_workers 16 \
    --data_file_dir ./data_dir/ \
    --denoise_dir /data/zzd/Datasets/UIR/WaveUIR/Train/Denoise/ \
    --derain_dir /data/zzd/Datasets/UIR/WaveUIR/Train/Derain/ \
    --dehaze_dir /data/zzd/Datasets/UIR/WaveUIR/Train/Dehaze/ \
    --gopro_dir /data/zzd/Datasets/UIR/WaveUIR/Train/Deblur/ \
    --enhance_dir /data/zzd/Datasets/UIR/WaveUIR/Train/Enhance/ \
    --raindrop_dir /data/zzd/Datasets/NTIRE/RainDrop/train/ \
    --output_path output/ \
    --ckpt_dir ./ckpt_three/ --num_gpus 1 >> train_three.log 2>&1 &


python test.py --test_data_path /home/notebook/data/personal/S9050086/flickr_soundnet/Data/ \
    --test_gt_path /home/notebook/data/personal/S9050086/flickr_soundnet/Annotations/ \
    --model_dir /home/notebook/code/personal/S9050086/EZ-VSL/checkpoints \
    --experiment_name vgg10k_034 \
    --testset 'flickr' \
    --alpha 0.4 \
    --gpu 0 \
    --model asy \
    --use_momentum \
    --use_mom_eval \
    --m_img 0.999 \
    --m_aud 0.999 \
    --dropout_img 0.9 \
    --dropout_aud 0 \
    # --save_visualizations \
#     # --multiprocessing_distributed True




EXPERIMENT=heard_test

rm -rf checkpoints/$EXPERIMENT

echo $EXPERIMENT

python train.py --multiprocessing_distributed \
    --train_data_path /home/notebook/data/personal/S9050086/vggsound \
    --test_data_path /home/notebook/data/personal/S9050086/flickr_soundnet/Data/ \
    --test_gt_path /home/notebook/data/personal/S9050086/flickr_soundnet/Annotations/ \
    --experiment_name $EXPERIMENT \
    --trainset 'vggss_heard' \
    --testset 'flickr' \
    --epochs 30 \
    --warmup 5 \
    --batch_size 256 \
    --init_lr 0.0001 \
    --port 1234 \
    --weight_decay 0.0001 \
    --fnac_loss1_weight 100 \
    --fnac_loss2_weight 100 \
    --fnac_loss3_weight 100 \
    --dropout_img 0.9 \
    --dropout_aud 0 \
    --gpu 0 \
    --seed 42 \



# heard
python test.py 
    --test_data_path [test data path] \
    --test_gt_path [test gt path] \
    --model_dir checkpoints/ \
    --experiment_name ez_vsl_heard_110 \
    --testset 'vggss_heard' \
    --alpha 0.4 \
    --gpu 0 \
    --use_mom_eval \
    --dropout_img 0.9 \
    --dropout_aud 0 \
    # --save_visualizations \
    # --multiprocessing_distributed True

# unheard
python test.py 
    --test_data_path [test data path] \
    --test_gt_path [test gt path] \
    --model_dir checkpoints/ \
    --experiment_name ez_vsl_heard_110 \
    --testset 'vggss_unheard' \
    --alpha 0.4 \
    --gpu 0 \
    --use_mom_eval \
    --dropout_img 0.9 \
    --dropout_aud 0 \
    # --save_visualizations \
    # --multiprocessing_distributed True

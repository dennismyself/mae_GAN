export MASTER_ADDR=localhost  # or the appropriate master node address
export MASTER_PORT=12355      # or another available port

torchrun --nproc_per_node=$SLURM_NTASKS_PER_NODE main_pretrain.py \
    --batch_size 32 \
    --norm_pix_loss \
    --mask_ratio 0.75 \
    --epochs 200 \
    --warmup_epochs 10 \
    --data_path "/home/jq271/rds/hpc-work/Dissertation/LLaVA/data/VLMPretrain" \
    --lr 1e-3 \
    --cuda "CUDA" \
    --model mae_vit_base_patch16 \

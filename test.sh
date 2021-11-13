 python -m torch.distributed.launch --nproc_per_node=4 --use_env main.py --model divervit_disable_d32_patch32_dim192_h6_r3_224 --batch-size 256 --data-path ../Dataset/imagenet/ILSVRC/Data/CLS-LOC --output_dir ./checkpoint_divervit_disable_d32_patch32_dim192_h6_r3_224/  --num_workers=8 --epochs=500 --divervit_alpha=0


 python -m torch.distributed.launch --nproc_per_node=4 --use_env main.py --model divervit_enable_d32_patch32_dim192_h6_r3_224 --batch-size 256 --data-path ../Dataset/imagenet/ILSVRC/Data/CLS-LOC --output_dir ./checkpoint_divervit_enable_head_diversity_d32_patch32_dim192_h6_r3_224_correct_loss_0.1/  --num_workers=8 --epochs=500 --divervit_alpha=0.1


 python -m torch.distributed.launch --nproc_per_node=4 --use_env main.py --model divervit_enable_d32_patch32_dim192_h6_r3_224 --batch-size 256 --data-path ../Dataset/imagenet/ILSVRC/Data/CLS-LOC --output_dir ./checkpoint_divervit_enable_head_diversity_d32_patch32_dim192_h6_r3_224_correct_loss_0.05/  --num_workers=8 --epochs=500 --divervit_alpha=0.05


 python -m torch.distributed.launch --nproc_per_node=4 --use_env main.py --model divervit_enable_d32_patch32_dim192_h6_r3_224 --batch-size 256 --data-path ../Dataset/imagenet/ILSVRC/Data/CLS-LOC --output_dir ./checkpoint_divervit_enable_head_diversity_d32_patch32_dim192_h6_r3_224_correct_loss_0.01/  --num_workers=8 --epochs=500 --divervit_alpha=0.01


 python -m torch.distributed.launch --nproc_per_node=4 --use_env main.py --model divervit_enable_d32_patch32_dim192_h6_r3_224 --batch-size 256 --data-path ../Dataset/imagenet/ILSVRC/Data/CLS-LOC --output_dir ./checkpoint_divervit_enable_head_diversity_d32_patch32_dim192_h6_r3_224_correct_loss_0.005/  --num_workers=8 --epochs=500 --divervit_alpha=0.005


 python -m torch.distributed.launch --nproc_per_node=4 --use_env main.py --model divervit_enable_d32_patch32_dim192_h6_r3_224 --batch-size 256 --data-path ../Dataset/imagenet/ILSVRC/Data/CLS-LOC --output_dir ./checkpoint_divervit_enable_head_diversity_d32_patch32_dim192_h6_r3_224_correct_loss_0.001/  --num_workers=8 --epochs=500 --divervit_alpha=0.001


 python -m torch.distributed.launch --nproc_per_node=4 --use_env main.py --model divervit_enable_d32_patch32_dim192_h6_r3_224 --batch-size 256 --data-path ../Dataset/imagenet/ILSVRC/Data/CLS-LOC --output_dir ./checkpoint_divervit_enable_head_diversity_d32_patch32_dim192_h6_r3_224_correct_loss_0.0005/  --num_workers=8 --epochs=500 --divervit_alpha=0.0005


 python -m torch.distributed.launch --nproc_per_node=4 --use_env main.py --model divervit_enable_d32_patch32_dim192_h6_r3_224 --batch-size 256 --data-path ../Dataset/imagenet/ILSVRC/Data/CLS-LOC --output_dir ./checkpoint_divervit_enable_head_diversity_d32_patch32_dim192_h6_r3_224_correct_loss_0.0001/  --num_workers=8 --epochs=500 --divervit_alpha=0.0001

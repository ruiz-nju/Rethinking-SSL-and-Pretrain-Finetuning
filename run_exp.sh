# #!/bin/bash

# sh scripts/coop/main.sh cifar10 vit_b16_ep100 8 0 &
# sh scripts/coop/main.sh cifar100 vit_b16_ep100 3 0 &
# sh scripts/coop/main.sh cifar100 vit_b16_ep100 8 0 &
# sh scripts/coop/main.sh stl10 vit_b16_ep100 8 0 &
# sh scripts/coop/main.sh stl10 vit_b16_ep100 23 0 &

# sh scripts/promptsrc/few_shot.sh cifar10 8 1 &
# sh scripts/promptsrc/few_shot.sh cifar100 8 1 &
# sh scripts/promptsrc/few_shot.sh stl10 8 1 &
# sh scripts/promptsrc/few_shot.sh stl10 23 1 &

# sh scripts/promptsrc/few_shot_one_seed.sh cifar100 3 1 1 &
# sh scripts/promptsrc/few_shot_one_seed.sh cifar100 3 1 2 &
# sh scripts/promptsrc/few_shot_one_seed.sh cifar100 3 1 3 &

# sh scripts/coop/main_one_seed.sh cifar10 rn101 25 0 1 &
# sh scripts/coop/main_one_seed.sh cifar10 rn101 25 0 2 &
# sh scripts/coop/main_one_seed.sh cifar10 rn101 25 0 3 &

# sh scripts/coop/main_one_seed.sh cifar100 rn101 25 0 1 &
# sh scripts/coop/main_one_seed.sh cifar100 rn101 25 0 2 &
# sh scripts/coop/main_one_seed.sh cifar100 rn101 25 0 3 &

sh scripts/coop/main_one_seed.sh cifar10 vit_b32 25 0 1 &
sh scripts/coop/main_one_seed.sh cifar10 vit_b32 25 0 2 &
sh scripts/coop/main_one_seed.sh cifar10 vit_b32 25 0 3 &

sh scripts/coop/main_one_seed.sh cifar100 vit_b32 25 1 1 &
sh scripts/coop/main_one_seed.sh cifar100 vit_b32 25 1 2 &
sh scripts/coop/main_one_seed.sh cifar100 vit_b32 25 1 3 &


wait
echo "All jobs completed!"
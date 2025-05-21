DEVICE=0

# zero-shot CLIP
sh scripts/zsclip/zeroshot.sh cifar10 vit_b16 all $DEVICE
sh scripts/zsclip/zeroshot.sh cifar100 vit_b16 all $DEVICE
sh scripts/zsclip/zeroshot.sh stl10 vit_b16 all $DEVICE
sh scripts/zsclip/zeroshot.sh imagenet127 vit_b16 all $DEVICE

# CoOp
sh scripts/coop/main.sh cifar10 vit_b16_ep100 4 all $DEVICE
sh scripts/coop/main.sh cifar10 vit_b16_ep100 5 all $DEVICE
sh scripts/coop/main.sh cifar10 vit_b16_ep100 8 all $DEVICE
sh scripts/coop/main.sh cifar10 vit_b16_ep100 15 all $DEVICE

sh scripts/coop/main.sh cifar100 vit_b16_ep100 3 all $DEVICE
sh scripts/coop/main.sh cifar100 vit_b16_ep100 4 all $DEVICE
sh scripts/coop/main.sh cifar100 vit_b16_ep100 5 all $DEVICE
sh scripts/coop/main.sh cifar100 vit_b16_ep100 8 all $DEVICE
sh scripts/coop/main.sh cifar100 vit_b16_ep100 15 all $DEVICE

sh scripts/coop/main.sh stl10 vit_b16_ep100 4 all $DEVICE
sh scripts/coop/main.sh stl10 vit_b16_ep100 8 all $DEVICE
sh scripts/coop/main.sh stl10 vit_b16_ep100 15 all $DEVICE
sh scripts/coop/main.sh stl10 vit_b16_ep100 23 all $DEVICE
sh scripts/coop/main.sh stl10 vit_b16_ep100 45 all $DEVICE

sh scripts/coop/main.sh imagenet127 vit_b16_ep100 25 all $DEVICE
sh scripts/coop/main.sh imagenet127 vit_b16_ep100 90 all $DEVICE

# PromptSRC
sh scripts/promptsrc/few_shot.sh cifar10 4 $DEVICE
sh scripts/promptsrc/few_shot.sh cifar10 5 $DEVICE
sh scripts/promptsrc/few_shot.sh cifar10 8 $DEVICE
sh scripts/promptsrc/few_shot.sh cifar10 15 $DEVICE

sh scripts/promptsrc/few_shot.sh cifar100 3 $DEVICE
sh scripts/promptsrc/few_shot.sh cifar100 4 $DEVICE
sh scripts/promptsrc/few_shot.sh cifar100 5 $DEVICE
sh scripts/promptsrc/few_shot.sh cifar100 8 $DEVICE
sh scripts/promptsrc/few_shot.sh cifar100 15 $DEVICE

sh scripts/promptsrc/few_shot.sh stl10 4 $DEVICE
sh scripts/promptsrc/few_shot.sh stl10 8 $DEVICE
sh scripts/promptsrc/few_shot.sh stl10 15 $DEVICE
sh scripts/promptsrc/few_shot.sh stl10 23 $DEVICE
sh scripts/promptsrc/few_shot.sh stl10 45 $DEVICE

sh scripts/promptsrc/few_shot.sh imagenet127 25 $DEVICE
sh scripts/promptsrc/few_shot.sh imagenet127 90 $DEVICE









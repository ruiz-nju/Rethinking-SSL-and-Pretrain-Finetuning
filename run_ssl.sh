DEVICE=0

# zero-shot CLIP
sh scripts/zsclip/zeroshot.sh cifar10 vit_b16 all $DEVICE
sh scripts/zsclip/zeroshot.sh cifar100 vit_b16 all $DEVICE
sh scripts/zsclip/zeroshot.sh stl10 vit_b16 all $DEVICE
sh scripts/zsclip/zeroshot.sh imagenet vit_b16 all $DEVICE

# CoOp
sh scripts/coop/main.sh cifar10 vit_b16_ep100 4 all $DEVICE
sh scripts/coop/main.sh cifar100 vit_b16_ep100 4 all $DEVICE
sh scripts/coop/main.sh stl10 vit_b16_ep100 4 all $DEVICE
sh scripts/coop/main.sh imagenet vit_b16_ep100 4 all $DEVICE

sh scripts/coop/main.sh cifar10 vit_b16_ep200 25 all $DEVICE
sh scripts/coop/main.sh cifar100 vit_b16_ep200 25 all $DEVICE
sh scripts/coop/main.sh stl10 vit_b16_ep200 25 all $DEVICE
sh scripts/coop/main.sh imagenet vit_b16_ep200 25 all $DEVICE

sh scripts/coop/main.sh cifar10 vit_b16_ep200 400 all $DEVICE
sh scripts/coop/main.sh cifar100 vit_b16_ep200 100 all $DEVICE
sh scripts/coop/main.sh stl10 vit_b16_ep200 100 all $DEVICE
sh scripts/coop/main.sh imagenet vit_b16_ep200 100 all $DEVICE


# PromptSRC
sh scripts/promptsrc/few_shot.sh cifar10 4 $DEVICE
sh scripts/promptsrc/few_shot.sh cifar100 4 $DEVICE
sh scripts/promptsrc/few_shot.sh stl10 4 $DEVICE
sh scripts/promptsrc/few_shot.sh imagenet 4 $DEVICE

sh scripts/promptsrc/few_shot.sh cifar10 25 $DEVICE
sh scripts/promptsrc/few_shot.sh cifar100 25 $DEVICE
sh scripts/promptsrc/few_shot.sh stl10 25 $DEVICE
sh scripts/promptsrc/few_shot.sh imagenet 25 $DEVICE

sh scripts/promptsrc/few_shot.sh cifar10 400 $DEVICE
sh scripts/promptsrc/few_shot.sh cifar100 100 $DEVICE
sh scripts/promptsrc/few_shot.sh stl10 100 $DEVICE
sh scripts/promptsrc/few_shot.sh imagenet 100 $DEVICE









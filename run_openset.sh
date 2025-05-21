DEVICE=0

# zero-shot CLIP
sh scripts/zsclip/zeroshot.sh cifar10 vit_b16 all $DEVICE
sh scripts/zsclip/zeroshot.sh imagenet30 vit_b16 all $DEVICE
sh scripts/zsclip/zeroshot.sh imagenet100 vit_b16 all $DEVICE

# CoOp
sh scripts/coop/main.sh cifar10 vit_b16_ep200 100 all $DEVICE
sh scripts/coop/main.sh imagenet30 vit_b16_ep200 50 all $DEVICE
sh scripts/coop/main.sh imagenet100 vit_b16_ep200 50 all $DEVICE

# PromptSRC
sh scripts/promptsrc/few_shot.sh cifar10 100 $DEVICE
sh scripts/promptsrc/few_shot.sh imagenet30 50 $DEVICE
sh scripts/promptsrc/few_shot.sh imagenet100 50 $DEVICE
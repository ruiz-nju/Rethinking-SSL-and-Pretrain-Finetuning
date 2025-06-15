DEVICE=0

# # zero-shot CLIP
# sh scripts/zsclip/zeroshot.sh cifar10_ow vit_b16 base $DEVICE
# sh scripts/zsclip/zeroshot.sh cifar10_ow vit_b16 new $DEVICE
# sh scripts/zsclip/zeroshot.sh cifar10_ow vit_b16 all $DEVICE

# sh scripts/zsclip/zeroshot.sh cifar100_ow vit_b16 base $DEVICE
# sh scripts/zsclip/zeroshot.sh cifar100_ow vit_b16 new $DEVICE
# sh scripts/zsclip/zeroshot.sh cifar100_ow vit_b16 all $DEVICE

# sh scripts/zsclip/zeroshot.sh imagenet100 vit_b16 base $DEVICE
# sh scripts/zsclip/zeroshot.sh imagenet100 vit_b16 new $DEVICE
# sh scripts/zsclip/zeroshot.sh imagenet100 vit_b16 all $DEVICE

# # CoOp
# sh scripts/coop/main_ow.sh cifar10_ow vit_b16_ep50 16 base $DEVICE
# sh scripts/coop/eval_ow.sh cifar10_ow vit_b16_ep50 16 base $DEVICE
# sh scripts/coop/eval_ow.sh cifar10_ow vit_b16_ep50 16 new $DEVICE
# sh scripts/coop/eval_ow.sh cifar10_ow vit_b16_ep50 16 all $DEVICE

# sh scripts/coop/main_ow.sh cifar100_ow vit_b16_ep50 16 base $DEVICE
# sh scripts/coop/eval_ow.sh cifar100_ow vit_b16_ep50 16 base $DEVICE
# sh scripts/coop/eval_ow.sh cifar100_ow vit_b16_ep50 16 new $DEVICE
# sh scripts/coop/eval_ow.sh cifar100_ow vit_b16_ep50 16 all $DEVICE

# sh scripts/coop/main_ow.sh imagenet100 vit_b16_ep50 16 base $DEVICE
# sh scripts/coop/eval_ow.sh imagenet100 vit_b16_ep50 16 base $DEVICE
# sh scripts/coop/eval_ow.sh imagenet100 vit_b16_ep50 16 new $DEVICE
# sh scripts/coop/eval_ow.sh imagenet100 vit_b16_ep50 16 all $DEVICE

# PromptSRC
for seed in 1 2 3
do
    sh scripts/promptsrc/base2new_train.sh cifar10_ow $seed 16 base  $DEVICE
    sh scripts/promptsrc/base2new_test.sh cifar10_ow $seed 16 base  $DEVICE
    sh scripts/promptsrc/base2new_test.sh cifar10_ow $seed 16 new  $DEVICE
    sh scripts/promptsrc/base2new_test.sh cifar10_ow $seed 16 all  $DEVICE

    sh scripts/promptsrc/base2new_train.sh cifar100_ow $seed 16 base  $DEVICE
    sh scripts/promptsrc/base2new_test.sh cifar100_ow $seed 16 base  $DEVICE
    sh scripts/promptsrc/base2new_test.sh cifar100_ow $seed 16 new  $DEVICE
    sh scripts/promptsrc/base2new_test.sh cifar100_ow $seed 16 all  $DEVICE

    sh scripts/promptsrc/base2new_train.sh imagenet100 $seed 16 base  $DEVICE
    sh scripts/promptsrc/base2new_test.sh imagenet100 $seed 16 base  $DEVICE
    sh scripts/promptsrc/base2new_test.sh imagenet100 $seed 16 new  $DEVICE
    sh scripts/promptsrc/base2new_test.sh imagenet100 $seed 16 all  $DEVICE
done



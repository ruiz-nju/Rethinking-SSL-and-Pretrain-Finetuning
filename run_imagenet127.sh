# sh scripts/zsclip/zeroshot.sh imagenet127 vit_b16 all 0 

# sh scripts/coop/main.sh imagenet127 vit_b16_ep200 25 1 &

# sh scripts/promptsrc/few_shot.sh imagenet127 25 1 &

sh scripts/coop/main_one_seed.sh imagenet127 vit_b16_ep200 90 0 1 &
sh scripts/coop/main_one_seed.sh imagenet127 vit_b16_ep200 90 0 2 &
sh scripts/coop/main_one_seed.sh imagenet127 vit_b16_ep200 90 0 3 &

sh scripts/promptsrc/few_shot_one_seed.sh imagenet127 90 1 1 &
sh scripts/promptsrc/few_shot_one_seed.sh imagenet127 90 1 2 &
sh scripts/promptsrc/few_shot_one_seed.sh imagenet127 90 1 3 &
wait
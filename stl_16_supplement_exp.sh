sh scripts/coop/main_one_seed.sh stl10_16 vit_b16 100 1 1 &
sh scripts/coop/main_one_seed.sh stl10_16 vit_b16 100 1 2 &
sh scripts/coop/main_one_seed.sh stl10_16 vit_b16 100 1 3 &

sh scripts/promptsrc/few_shot_one_seed.sh stl10_16 100 1 1 &
sh scripts/promptsrc/few_shot_one_seed.sh stl10_16 100 1 2 &
sh scripts/promptsrc/few_shot_one_seed.sh stl10_16 100 1 3 &

sh scripts/zsclip/zeroshot.sh stl10_16 vit_b16 all 1 &
wait
echo "All jobs completed!"
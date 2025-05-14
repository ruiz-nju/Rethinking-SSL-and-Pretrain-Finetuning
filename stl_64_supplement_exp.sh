# stl10 32x32 100shots
sh scripts/coop/main_one_seed.sh stl10_64 vit_b16 100 1 1 &
sh scripts/coop/main_one_seed.sh stl10_64 vit_b16 100 1 2 &
sh scripts/coop/main_one_seed.sh stl10_64 vit_b16 100 1 3 &

sh scripts/promptsrc/few_shot_one_seed.sh stl10_64 100 1 1 &
sh scripts/promptsrc/few_shot_one_seed.sh stl10_64 100 1 2 &
sh scripts/promptsrc/few_shot_one_seed.sh stl10_64 100 1 3 &

sh scripts/zsclip/zeroshot.sh stl10_64 vit_b16 all 1 &
wait
echo "All jobs completed!"
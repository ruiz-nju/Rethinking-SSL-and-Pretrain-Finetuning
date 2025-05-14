
# "eurosat", "mlrsnet", "optimal", "patternnet", "resisc45", "rsicb128", "rsicb256", "whurs19"
for SEED in 1 2 3
do
  bash scripts/promptsrc/base2new_train.sh cifar10 ${SEED} 50 all
  bash scripts/promptsrc/base2new_test.sh cifar10 ${SEED} 50 all
  bash scripts/promptsrc/base2new_train.sh cifar100 ${SEED} 500 all
  bash scripts/promptsrc/base2new_test.sh cifar100 ${SEED} 500 all
  # bash scripts/promptsrc/base2new_train.sh imagenet ${SEED} 25
  # bash scripts/promptsrc/base2new_test.sh imagenet ${SEED} 25
  bash scripts/promptsrc/base2new_train.sh stl10 ${SEED} 150 all
  bash scripts/promptsrc/base2new_test.sh stl10 ${SEED} 150 all
done
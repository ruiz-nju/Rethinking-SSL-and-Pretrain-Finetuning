
# "eurosat", "mlrsnet", "optimal", "patternnet", "resisc45", "rsicb128", "rsicb256", "whurs19"
for SEED in 1
do
  # bash scripts/promptsrc/base2new_train.sh cifar101 ${SEED} 25 base
  # bash scripts/promptsrc/base2new_test.sh cifar101 ${SEED} 25 new
  # bash scripts/promptsrc/base2new_test.sh cifar101 ${SEED} 25 all
  # bash scripts/promptsrc/base2new_train.sh cifar1001 ${SEED} 25 base
  # bash scripts/promptsrc/base2new_test.sh cifar1001 ${SEED} 25 new
  # bash scripts/promptsrc/base2new_test.sh cifar1001 ${SEED} 25 all
  bash scripts/promptsrc/base2new_train.sh imagenet100 ${SEED} 25 base
  bash scripts/promptsrc/base2new_test.sh imagenet100 ${SEED} 25 new
  bash scripts/promptsrc/base2new_test.sh imagenet100 ${SEED} 25 all
  # bash scripts/promptsrc/base2new_train.sh stl10 ${SEED} 25 base
  # bash scripts/promptsrc/base2new_test.sh stl10 ${SEED} 25 new
  # bash scripts/promptsrc/base2new_test.sh stl10 ${SEED} 25 all
  # bash scripts/promptsrc/base2new_train.sh rsicb128 ${SEED} cuda:1
  # bash scripts/promptsrc/base2new_test.sh rsicb128 ${SEED} cuda:1
  # bash scripts/promptsrc/base2new_train.sh resisc45 ${SEED} cuda:1
  # bash scripts/promptsrc/base2new_test.sh resisc45 ${SEED} cuda:1
  # bash scripts/promptsrc/base2new_train.sh rsicb256 ${SEED} cuda:1
  # bash scripts/promptsrc/base2new_test.sh rsicb256 ${SEED} cuda:1
  # bash scripts/promptsrc/base2new_train.sh whurs19 ${SEED} cuda:1
  # bash scripts/promptsrc/base2new_test.sh whurs19 ${SEED} cuda:1
done
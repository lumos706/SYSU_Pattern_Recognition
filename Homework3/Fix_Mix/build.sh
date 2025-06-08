# python fixmatch.py --num-labeled 40 --seed 5 --out results/fixmatch/cifar10@40.5
# python fixmatch.py --num-labeled 250 --seed 5 --out results/fixmatch/cifar10@250.5
# python fixmatch.py --num-labeled 4000 --seed 5 --out results/fixmatch/cifar10@4000.5
python mixmatch.py --seed 5  --num-labeled 40 --out ./results/mixmatch/cifar10@40.5
python mixmatch.py --seed 5  --num-labeled 250 --out ./results/mixmatch/cifar10@250.5
python mixmatch.py --seed 5  --num-labeled 4000 --out ./results/mixmatch/cifar10@4000.5
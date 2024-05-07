for i in {1..30}
do
    sbatch -p seas_gpu -t 20 -n 1 --mem 20G --gres gpu:1 -o bash-outputs/%A-%a.out -e bash-errors/%A-%a.err run_cnn.sh fuse $i
done

for i in {1..30}
do
    sbatch -p seas_gpu -t 20 -n 1 --mem 20G --gres gpu:2 -o bash-outputs/%A-%a.out -e bash-errors/%A-%a.err run_cnn.sh three_channel $i
done

for i in {1..30}
do
    sbatch -p seas_gpu -t 20 -n 1 --mem 20G --gres gpu:2 -o bash-outputs/%A-%a.out -e bash-errors/%A-%a.err run_cnn.sh png $i
done

for i in {1..30}
do
    sbatch -p seas_gpu -t 20 -n 1 --mem 60G --gres gpu:6 -o bash-outputs/%A-%a.out -e bash-errors/%A-%a.err run_cnn.sh original $i
done

for i in {1..30}
do
    sbatch -p seas_gpu -t 20 -n 1 --mem 40G --gres gpu:2 -o bash-outputs/%A-%a.out -e bash-errors/%A-%a.err run_cnn.sh five_channel $i
done

for i in {1..30}
do
    sbatch -p seas_gpu -t 20 -n 1 --mem 20G --gres gpu:4 -o bash-outputs/%A-%a.out -e bash-errors/%A-%a.err run_cnn.sh png_moe $i
done

for i in {1..30}
do
    sbatch -p seas_gpu -t 30 -n 1 --mem 60G --gres gpu:9 -o bash-outputs/%A-%a.out -e bash-errors/%A-%a.err run_cnn.sh original_moe $i
done

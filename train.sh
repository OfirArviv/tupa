#!/usr/bin/env bash
#SBATCH --mem=50G
#SBATCH --time=7-0
#SBATCH --gres=gpu:1
#SBATCH -c16

if [[ $# -lt 1 ]]; then
    SUFFIX=`date '+%Y%m%d'`
else
    SUFFIX="$1"
fi
if [[ -n "${FRAMEWORK}" ]]; then
    SUFFIX=${FRAMEWORK}-${SUFFIX}
fi

mkdir -p models
for FRAMEWORK in ${FRAMEWORK:-`grep -Po '(?<="framework": ")\w+(?=")' ../mrp/2019/training/training.mrp | sort -u`}; do
    grep '"framework": "'${FRAMEWORK}'"' ../mrp/2019/training/training.mrp | shuf > models/mrp-${SUFFIX}.train_dev.${FRAMEWORK}.mrp
done
head -n 500 -q models/mrp-${SUFFIX}.train_dev.*.mrp > models/mrp-${SUFFIX}.dev.mrp
tail -n+501 -q models/mrp-${SUFFIX}.train_dev.*.mrp | shuf > models/mrp-${SUFFIX}.train.mrp
rm -f models/mrp-${SUFFIX}.train_dev.*.mrp

echo $SUFFIX
python -m tupa --seed $RANDOM --cores=15 --use-bert --dynet-gpu --pytorch-gpu --no-validate-oracle --save-every=50000 --timeout=20 \
    -t models/mrp-${SUFFIX}.train.mrp -d models/mrp-${SUFFIX}.dev.mrp \
    --conllu ../mrp/2019/companion/udpipe.mrp --alignment ../mrp/2019/companion/isi.mrp -m models/mrp-${SUFFIX} -v

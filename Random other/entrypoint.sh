source /conda/etc/profile.d/conda.sh
conda activate rapids

echo "Mounting GCSFuse: $1/$2"
mkdir -p /rapids/dataset
gcsfuse --only-dir $2 \
        --implicit-dirs $1 /rapids/dataset

echo "Running: main.py ${@:3}"
python main.py ${@:3}

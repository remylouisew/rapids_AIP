source /conda/etc/profile.d/conda.sh
conda activate rapids

echo "Mounting GCSFuse: $1/$2"
mkdir -p /rapids/dataset
gcsfuse --only-dir $2 \
        --implicit-dirs $1 /rapids/dataset

echo "Running: rapids.py ${@:3}"
python rapids.py ${@:3}

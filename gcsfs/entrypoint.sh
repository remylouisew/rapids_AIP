source /conda/etc/profile.d/conda.sh
conda activate rapids

echo "Running: rapids.py $@"
python rapids.py $@

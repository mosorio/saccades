#!/bin/bash
#SBATCH --job-name=map_2to7_count_total       # name shown in the SLURM queue (squeue)
#SBATCH --account=f202509086cpcaa1g           # which allocation account to charge compute time to
#SBATCH --partition=normal-a100-40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=32
#SBATCH --time=08:00:00               
#SBATCH --output=logs/%x-%j.out                
#SBATCH --error=logs/%x-%j.err

module purge
module load Mamba/23.1.0-4
module load CUDA/11.8.0

eval "$(conda shell.bash hook)"
conda activate /projects/F202509086CPCAA1/.conda/conda_envs/newsac

cd /projects/F202509086CPCAA1/saccades

echo "Running on: $(hostname)"
echo "Python: $(which python)"
python --version

python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA build:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')"

bash datasets/make_curriculum_datasets.sh 2to7 count_total 30000 5000 42
srun bash run_curriculum_map.sh 2to7 count_total 30000 5000 42 1500 512 fixed \
  dataset_based sqrt_inv gru 1 Adam step 1 0.0005 "256 512"

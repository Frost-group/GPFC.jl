#!/bin/sh -xu
#SBATCH --job-name="SomebodyElsesProblem"
#SBATCH --get-user-env
#SBATCH --output=vasp.stdout
#SBATCH --error=vasp.stderr

#SBATCH --partition=standard
#SBATCH --account=e05-bulk-fro
#SBATCH --qos=standard

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=128
#SBATCH --cpus-per-task=1
#SBATCH --time=00:20:00 
# 2 minutes to midnight
# Set the number of threads to 1
#   This prevents any threaded system libraries from automatically
#   using threading.
export OMP_NUM_THREADS=1
module load PrgEnv-cray
# More efficient (Sean Jan 2021) options for VASP 6 on ARCHER2
srun --cpu-bind=rank --hint=nomultithread --distribution=block:block /work/e05/e05/jarvist/bin/ARCHER2-VASP/vasp6.3.0-Feb2022/vasp.6.3.0/bin/vasp_std
echo "For us, there is only the trying. The rest is not our business. ~T.S.Eliot"

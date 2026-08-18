#!/bin/bash
#SBATCH --job-name=RGBcrops
#SBATCH --time=24:00:00 #
#SBATCH --mail-type=ALL
#SBATCH --output=./outfiles/RGBcrops.out
#SBATCH --account=PUOM0017
#SBATCH --mem=64G

module purge
module load gcc/12.3.0
module load R/4.4.0
module load proj/9.2.1
module load gdal/3.7.3

Rscript Grid_and_crop_HARV_RGB.R 

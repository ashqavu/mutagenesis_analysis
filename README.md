# Deep mutational scanning (DMS) analysis
This pipeline has been developed for the purpose of analyzing sequencing data retrieved from deep mutational scanning (DMS) experiments. It takes paired-end Illumina sequencing data, merges and maps reads, then generates data matrices showing how many times each mutation in the library appeared in the sample.

## Tools
The tools featured in this pipeling include:
1. **FLASh** (T. Magoc, Salzberg S., 2011) for paired-end merging
2. **bowtie2** (Langmead B, Salzberg S., 2012) for alignment mapping
3. **SAMtools** (Danecek et. al., 2021) for alignment file processing

## Input Requirements
* a designated project folder containing paired-end reads from Illumina sequencing in a `raw_data` folder
* a reference genome in GENBANK format (for bowtie2 indexing) with the annotated gene of interest as a Gene feature

## Output
| Folder | Contents |
|---|---|
| job_outputs | `.err` and `.out` logs for each SLURM job |
| flash_merged | output files from FLASh merging |
| alignments | bowtie2 alignment files sorted and indexed by SAMtools and `total_reads.csv` with total reads found after sequence mapping|
| results/counts | `.csv` files for mutation count matrices and `.pkl` pickled forms of DataFrames from the pandas library|
| results/mutations | `.csv` files with list of all mutations found in data and `.pkl` pickled forms of DataFrames from the pandas library |
| results/mutations/quality_filtered | `.csv` files with list of all mutations that pass quality score filtering and `.pkl` pickled forms of DataFrames from the pandas library |
| results/mutations/quality_filtered/seq_lengths | `.csv` files with a list of sequence lengths found in the filtered mutations list and `.pkl` pickled forms of DataFrames from the pandas library |

## Usage
### Quick Start
```
bash start_slurm.sh <INPUT_FOLDER> <GENE_NAME>
```
### start_slurm.sh
The `start_slurm.sh` is a bash script written for the SLURM job scheduler used by UTSW's BioHPC and takes advantage of job arrays to initiate parallel processing of the samples, and is intended to be the only script run by the user. Given a `raw_data` folder in the project folder, the script will generate a `SampleNames.txt` file from the Illumina filenames and then launch a SLURM script in the format

```
sbatch --array <ARRAY_IDS> mutation_scanning.sh <INPUTFOLDER> <GENE_NAME>
```

### mutation_scanning.sh
`mutation_scanning.sh` is a SLURM script written for HPC platforms. It starts a job array with 1 sample per node and will use all CPUs available.

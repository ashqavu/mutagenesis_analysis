#!/bin/bash

# This script iterates through Illumina sequencing data output and creates a text file
# SampleNames.txt that matches the name of the sample to the sample number. It takes required
# arguments as described in the usage below. The script will then launch a SLURM job array for
# all sample simultaneously and in parallel.

# check that required flags (all) have been provided
if [[ ( $@ == "--help") ||  ( $@ == "-h") ]]
then
  echo "Usage:"
  echo "  bash start_slurm.sh -i <project_path> -g <gene_name> -r <gbk_file> -x <bowtie_index> -f <flash_path>"
  echo
  echo -e "  <project_path>\t\tPath to project folder" 
  echo -e "  <gene_name>\t\t\tGene name"
  echo -e "  <gbk_file>\t\t\t.gbk reference file with annotated gene"
  echo -e "  <bowtie_index>\t\tPath to bowtie index folder"
  echo -e "  <flash_path>\t\t\tPath to FLASh tool"
  echo "Options:"
  echo -e "  -h/--help\t\t\tPrint this usage message"
  exit 0
fi

while getopts 'i:g:r:x:f:' OPTION; do
  case "${OPTION}" in
    i) export INPUTFOLDER=$(realpath ${OPTARG}) ;;
    g) export GENE="${OPTARG}" ;;
    r) export REF_GBK=$(realpath ${OPTARG}) ;;
    x) export BOWTIE2_INDEXES=$(realpath ${OPTARG}) ;;
    f) export FLASH=$(realpath ${OPTARG}) ;;
  esac
done
export SCRIPT_FOLDER=$(dirname $(realpath $0))

# build SampleNames.txt output file matching sample number to sample name
for FILE in $INPUTFOLDER/raw_data/*; do
  if [[ $FILE == *fastq.gz ]]; then
    FILE=$(basename $FILE)
    SAMPLENAME=$(echo $FILE | awk -F '_S' '{ print $1 }');
    NUM=$(echo $FILE | awk -F '_S' '{ print $2 }' | awk -F '_' '{ print $1 }');
    if [[ $FILE =~ .*_R1 ]]; then
      echo -e $NUM'\t'$SAMPLENAME;
    fi
  fi
done > $INPUTFOLDER/raw_data/SampleNames.txt

sort -ho $INPUTFOLDER/raw_data/SampleNames.txt $INPUTFOLDER/raw_data/SampleNames.txt

# use SampleNames.txt matching to generate array_ids to feed to SLURM script
while read LINE; do
  array_ids=${array_ids}$(echo -e $(echo $LINE | awk '{ print $1 }'),)
done < $INPUTFOLDER/raw_data/SampleNames.txt

mkdir -p alignments
> alignments/total_reads.tsv
# switch to folder to hold SLURM job output files
mkdir -p $INPUTFOLDER/job_outputs
cd $INPUTFOLDER/job_outputs

# start SLURM job array
echo -e "sbatch --array $array_ids $SCRIPT_FOLDER/mutation_scanning.sh"
sbatch --array $array_ids $SCRIPT_FOLDER/mutation_scanning.sh
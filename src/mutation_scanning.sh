#!/bin/bash
#SBATCH --job-name mutation_scanning
#SBATCH -p 256GBv1
#SBATCH -e %a.err
#SBATCH -o %a.out
#SBATCH -t 1:00:0

#SBATCH --mail-type NONE

# USAGE: sbatch --array/-a [indexes] mutation_scanning.sh

# This script will process the data of one set of paired-end reads from Illumina data sequencing
# 1) Paired-end reads are merged with FLASh
# 2) bowtie2 and samtools are used to align the sequences to the reference sequence and sort
# 3) Total number of reads after alignment is calculated
# 4) Custom Python script (sequence_mapping.py) is used to count mutations and save the results

STARTTIME=$SECONDS

NUMCORES="$(nproc)"

echo
echo -e "Date: $(date +%Y-%m-%d)"
echo "Project folder: "$INPUTFOLDER""
echo "Using "$NUMCORES" cores..."
echo

BOWTIE_FILES=$(realpath $BOWTIE2_INDEXES/*)
BOWTIE_BASENAME=$(basename -- "${BOWTIE_FILES[0]%%.*}")

SAMPLE_NAMES=$INPUTFOLDER/raw_data/SampleNames.txt
if ! grep -o "^$SLURM_ARRAY_TASK_ID\s.*\w" $SAMPLE_NAMES; then
echo "Skipped number in SampleNames.txt"
  exit
fi
SAMPLE=$(grep -o "^$SLURM_ARRAY_TASK_ID\s.*\w" $SAMPLE_NAMES | awk -F '\t' '{ print $2 }')

module load python
conda activate TEM-1

cd $INPUTFOLDER
mkdir -p flash_merged
mkdir -p results/mutations
mkdir -p results/counts

FWD=$(realpath 'raw_data/'$(ls -v1 raw_data/ --color=never | grep "_S"$SLURM_ARRAY_TASK_ID"_.*R1"))
REV=$(realpath 'raw_data/'$(ls -v1 raw_data/ --color=never | grep "_S"$SLURM_ARRAY_TASK_ID"_.*R2"))

# merge paired-end sequences
# echo "[$(date +"%T")] flash --min-overlap=10 --max-overlap=151 --output_directory flash_merged --output-prefix=$SAMPLE --compress $FWD $REV"
# flash --min-overlap=10 --max-overlap=151 --output-directory=flash_merged --output-prefix=$SAMPLE --compress $FWD $REV
# echo

# align and sort unpaired sequences
echo "[$(date +"%T")] bowtie2 -x $BOWTIE_BASENAME -t -q -1 $FWD -2 $REV --very-sensitive-local --no-unal --ma 2 --rfg 1000,1000 -p $NUMCORES |"
echo "[$(date +"%T")] samtools view -h --threads $NUMCORES -b |"
echo "[$(date +"%T")] samtools sort --thread $NUMCORES -o alignments/"$SAMPLE".bam"
echo "[$(date +"%T")] samtools index -b -@ $NUMCORES alignments/"$SAMPLE".bam"

bowtie2 -x $BOWTIE_BASENAME -t -q -1 $FWD -2 $REV --very-sensitive-local --no-unal --ma 2 --rfg 1000,1000 -p $NUMCORES |
samtools view -h --threads $NUMCORES -b |
samtools sort --thread $NUMCORES -o alignments/$SAMPLE.bam 
samtools index -b -@ $NUMCORES alignments/$SAMPLE.bam

# count total number of sequences that aligned to the entire contig (plasmid map)
TOTAL_READS=$(samtools idxstats alignments/$SAMPLE.bam -@ $NUMCORES | grep -i $BOWTIE_BASENAME | cut -f 3)
# echo -e $SAMPLE'\t'$TOTAL_READS >> alignments/total_reads.tsv
echo "[$(date +"%T")] $TOTAL_READS reads found for $SAMPLE"
echo

echo "[$(date +"%T")] python $SCRIPT_FOLDER/sequence_mapping.py alignments/$SAMPLE.bam $REF_GBK $GENE"
python $SCRIPT_FOLDER/sequence_mapping.py alignments/$SAMPLE.bam $REF_GBK $GENE

ENDTIME=$SECONDS
echo "[$(date +"%T")] Total time: $(($ENDTIME - $STARTTIME)) seconds"

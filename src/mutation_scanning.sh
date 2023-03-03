#!/bin/bash
#SBATCH --job-name mutation_scanning
#SBATCH -p 256GBv1
#SBATCH -e %a.err
#SBATCH -o %a.out
#SBATCH -t 0:30:0

#SBATCH --mail-type NONE

#USAGE bash --array <1-96> <SCRIPT.SH> <PATH_TO_PROJECT_FOLDER> <GENE_NAME>

STARTTIME=$SECONDS

INPUTFOLDER=$(realpath $1)
GENE=$2

NUMCORES="$(nproc)"
FLASH="/work/greencenter/s426833/TEM-1/flash-1.2.11/flash"
#TODO: change hard-coded path references
REF_FOLDER="/work/greencenter/s426833/TEM-1/ref_data" 
REF_GBK="$REF_FOLDER/pBR322_AvrII_rc.gbk"
SCRIPT_PATH="/work/greencenter/s426833/TEM-1/src/sequence_mapping.py"

export BOWTIE2_INDEXES="$REF_FOLDER/pBR322_bowtie_index"

SAMPLE_NAMES=$INPUTFOLDER/raw_data/SampleNames.txt
if ! grep -o "^$SLURM_ARRAY_TASK_ID\s.*\w" $SAMPLE_NAMES; then
echo "Skipped number in SampleNames.txt"
  exit
fi
SAMPLE=$(grep -o "^$SLURM_ARRAY_TASK_ID\s.*\w" $SAMPLE_NAMES | awk -F '\t' '{ print $2 }')

module load bowtie2/2.4.2
module load samtools/intel/1.10
module load python/3.8.x-anaconda

source activate TEM-1

cd $INPUTFOLDER
mkdir -p flash_merged
mkdir -p alignments
mkdir -p results/mutations
mkdir -p results/counts

echo
echo -e "Date: $(date +%Y-%m-%d)"
echo "Project folder: "$INPUTFOLDER""
echo "Using "$NUMCORES" cores..."
echo

FWD=$(realpath 'raw_data/'$(ls -v1 raw_data/ --color=never | grep "_S"$SLURM_ARRAY_TASK_ID"_.*R1"))
REV=$(realpath 'raw_data/'$(ls -v1 raw_data/ --color=never | grep "_S"$SLURM_ARRAY_TASK_ID"_.*R2"))

echo "[$(date +"%T")] flash --min-overlap=10 --max-overlap=151 --output_directory flash_merged --output-prefix=$SAMPLE --compress $FWD $REV"
flash --min-overlap=10 --max-overlap=151 --output-directory=flash_merged --output-prefix=$SAMPLE --compress $FWD $REV
echo
# TODO: also change hard-coded here
echo "[$(date +"%T")] bowtie2 -x pBR322-blaTEM1 -t --very-sensitive-local --no-unal --ma 2 --rfg 1000,1000 -p $NUMCORES -q -U flash_merged/"$SAMPLE".extendedFrags.fastq.gz |"
echo "[$(date +"%T")] samtools view -h --threads $NUMCORES -b |"
echo "[$(date +"%T")] samtools sort --thread $NUMCORES -o alignments/"$SAMPLE".bam"
echo "[$(date +"%T")] samtools index -b -@ $NUMCORES alignments/"$SAMPLE".bam"

# TODO: change hard-coded reference to pBR322-blaTEM1 here
bowtie2 -x pBR322-blaTEM1 -t --very-sensitive-local --no-unal --ma 2 --rfg 1000,1000 -p $NUMCORES -q -U flash_merged/"$SAMPLE".extendedFrags.fastq.gz |
samtools view -h --threads $NUMCORES -b |
samtools sort --thread $NUMCORES -o alignments/$SAMPLE.bam 
samtools index -b -@ $NUMCORES alignments/$SAMPLE.bam

TOTAL_READS=$(samtools idxstats alignments/$SAMPLE.bam -@ 72 | grep -i pbr322 | cut -f 3)
echo -e $SAMPLE'\t'$TOTAL_READS >> alignments/total_reads.tsv
echo "[$(date +"%T")] $TOTAL_READS reads found for $SAMPLE"
echo

echo "[$(date +"%T")] python /work/greencenter/s426833/TEM-1/src/sequence_mapping.py alignments/$SAMPLE.bam $REF_GBK $GENE"
python $SCRIPT_PATH alignments/$SAMPLE.bam $REF_GBK $GENE

ENDTIME=$SECONDS
echo "[$(date +"%T")] Total time: $(($ENDTIME - $STARTTIME)) seconds"

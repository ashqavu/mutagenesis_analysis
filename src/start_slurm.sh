#!/bin/bash
INPUTFOLDER=$(realpath $1)
GENE=$2

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

while read LINE; do
  array_ids=${array_ids}$(echo -e $(echo $LINE | awk '{ print $1 }'),)
done < $INPUTFOLDER/raw_data/SampleNames.txt

mkdir -p $INPUTFOLDER/alignments
> $INPUTFOLDER/alignments/total_reads.tsv
mkdir -p $INPUTFOLDER/job_outputs
cd $INPUTFOLDER/job_outputs

echo -e 'sbatch --array '$array_ids' /work/greencenter/Toprak_lab/s426833/TEM-1/src/mutation_scanning.sh '$INPUTFOLDER' '$GENE''
sbatch --array $array_ids /work/greencenter/s426833/TEM-1/src/mutation_scanning.sh $INPUTFOLDER $GENE
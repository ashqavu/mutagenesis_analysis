#!/usr/bin/env python
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pysam
from Bio.Data import CodonTable, IUPACData
from tqdm import tqdm

from plasmid_map import Gene


def parse_args():
    parser = argparse.ArgumentParser(
        description="Finds nucleotide changes in an indexed in.bam file.",
        add_help=False,
    )

    required_args = parser.add_argument_group("required")
    required_args.add_argument(
        "bam", metavar="in.bam", type=str, help="Indexed input BAM file"
    )
    required_args.add_argument(
        "ref", metavar="ref.gbk", type=str, help=".gbk file with annotated CDS feature"
    )
    required_args.add_argument(
        "gene", type=str, help="Name of gene CDS sequence feature found in ref.gbk file"
    )

    optional_args = parser.add_argument_group("optional")
    optional_args.add_argument(
        "-c",
        "--contig",
        type=str,
        help="Input reference contig. Analysis defaults to the first contig name found in the SAM header, so these options need to be specified to search for a different region.",
        required=False,
    )
    optional_args.add_argument(
        "-q",
        "--quality-filter",
        dest="q",
        metavar="INT",
        type=int,
        help="Minimum accepted phred score for each mutation position (default: 30)",
        default=30,
    )
    optional_args.add_argument(
        "-o", "--output", type=str, help="Specify the output folder", required=False
    )
    optional_args.add_argument(
        "-h", "--help", action="help", help="show this help message and exit"
    )
    args = parser.parse_args()
    return args


translation_table = CodonTable.standard_dna_table.forward_table
stop_codons = CodonTable.standard_dna_table.stop_codons
translation_table.update({stop_codon: "*" for stop_codon in stop_codons})


def fGetTime():
    return f"""[{time.strftime("%H:%M:%S")}]"""


def mutation_finder(alignments, gene):
    cds_start = gene.cds.location.start
    cds_end = gene.cds.location.end
    insertions = []
    deletions = []
    wildtypes = []
    mutations = []
    for aln in alignments:
        # record and discard indels (mostly deletions from AT-rich regions of gene)
        if any(a == 1 for a, _ in aln.cigartuples):
            insertions.append(aln)
            continue
        elif any(a == 2 for a, _ in aln.cigartuples):
            deletions.append(aln)
            continue

        # iterate over aligned pairs to find mutation
        aligned_pairs = aln.get_aligned_pairs(matches_only=True, with_seq=True)
        # lowercase letter indicates substitution
        if aln.get_reference_sequence().isupper():
            wildtypes.append(aln)
            continue
        # proceed along sequence until the CDS region
        for query_pos, ref_pos, ref_base in aligned_pairs:
            if ref_base.islower():
                mutations.append(
                    (
                        aln.query_name,
                        ref_pos,
                        ref_base,
                        query_pos,
                        aln.query_sequence[query_pos],
                        aln.query_qualities[query_pos],
                        aln.query_sequence,
                        aln.get_overlap(
                            start=cds_start, end=cds_end
                        ),
                    )
                )
    return mutations


def read_mutations(mutations, gene):
    df = pd.DataFrame(
        mutations,
        columns=[
            "read_id",
            "ref_pos",
            "ref_base",
            "query_pos",
            "query_base",
            "base_quality",
            "query_seq",
            "cds_overlap_length",
        ],
    )
    # set dtypes, we're trying to save memory
    # if you have a bigger genome you might have to upcast ref_pos to Int32
    df = df.astype(
        {
            "ref_pos": "Int16",
            "ref_base": "string",
            "query_base": "string",
            "base_quality": "Int8",
        }
    )
    df = df.astype({"ref_base": "category", "query_base": "category"})

    # find position of codon's residue in protein
    df["aa_pos"] = (
        df["ref_pos"].sub(gene.cds.location.start).floordiv(3).astype("Int16")
    )
    df["ref_codon"] = (
        df["aa_pos"].map(gene.cds_codon_dict).astype("string").astype("category")
    )
    df["ref_aa"] = df["ref_codon"].map(translation_table).astype("category")
    # pull up position for the first base of the codon of the nucleotide
    df["codon_pos"] = df["aa_pos"].map(gene.codon_starts).astype("Int16")
    # adjust the codon position from the read to set the reading frame
    df["query_codon_pos"] = df["query_pos"].add(df["codon_pos"] - df["ref_pos"])
    df["query_codon"] = [
        seq[pos : pos + 3] if pd.notnull(pos) else pd.NA
        for seq, pos in zip(df["query_seq"], df["query_codon_pos"])
    ]
    df["query_codon"] = df["query_codon"].astype("string").astype("category")
    df["query_aa"] = (
        df["query_codon"].map(translation_table).astype("string").astype("category")
    )
    df = df[
        [
            "read_id",
            "ref_pos",
            "ref_base",
            "query_base",
            "base_quality",
            "aa_pos",
            "ref_codon",
            "ref_aa",
            "query_codon",
            "query_aa",
            "query_seq",
            "cds_overlap_length",
        ]
    ]
    return df


def find_multiple_mutants(df):
    df_multiples = df[df.duplicated("read_id", keep=False)]
    df_singles = df.drop_duplicates("read_id", keep=False)
    return df_multiples, df_singles


def count_mutations(df, gene, quality_filter=30):
    # do a quality check first
    df_quality_filter = df.query("base_quality >= @quality_filter")
    print(
        f"{df_quality_filter.shape[0] / df.shape[0]:.2%} of all mutations found passed with quality scores >= {quality_filter}"
    )
    # then filter redundant mutations (when all mutated bases are in same codon)
    df_quality_filter = df_quality_filter.drop_duplicates(["read_id", "aa_pos"])

    df_multiples, df_singles = find_multiple_mutants(df_quality_filter)
    num_singles = df_singles.shape[0]
    num_multiples = df_multiples["read_id"].unique().shape[0]    

    print(f"Number of reads with one mutation passing quality check: {num_singles:,}")
    print(
        f"Number of reads with multiple mutations passing quality check: {num_multiples:,}"
    )
    print(f"Number of single mutants in CDS region: {df_singles.dropna().shape[0]:,}")

    df_counts = pd.DataFrame(
        index=np.arange(len(gene.cds_translation)),
        columns=list(IUPACData.protein_letters + "*∅"),
    )
    for (pos, aa), count in df_singles.groupby(["aa_pos", "query_aa"], observed=True).size().items():
        if gene.cds_translation[pos] == aa:
            df_counts.loc[pos, "∅"] = count
        else:
            df_counts.loc[pos, aa] = count
    df_counts.fillna(0, inplace=True)
    # df_counts.index = pd.Index(gene.numbering_scheme, dtype="int64")

    return df_counts, num_singles, num_multiples


def main():
    args = parse_args()
    gene = Gene(args.ref, args.gene)

    quality_filter = args.q
    cds_start = gene.cds.location.start
    cds_end = gene.cds.location.end

    input_file = Path(args.bam)
    input_folder = input_file.parent.parent
    sample_name = input_file.stem

    if args.output:
        output_folder = args.output
    else:
        output_folder = input_folder / "results"

    start_time = time.time()

    print(f"{fGetTime()} Finding mutations...")
    with pysam.AlignmentFile(input_file, "rb", threads=os.cpu_count()) as bam:
        if args.contig:
            contig = args.contig
        else:
            contig = bam.header.references[0]
        num_alignments = int(
            pysam.view(
                "-c", input_file.as_posix(), f"{contig}:{cds_start+1}-{cds_end}"
            ).strip()
        )
        alns = tqdm(
            bam.fetch(contig=contig, start=cds_start, stop=cds_end),
            total=num_alignments,
            file=sys.stdout,
        )
        mutations = mutation_finder(alns, gene)
    print(f"{fGetTime()} Done")

    print(f"{fGetTime()} Generating mutations table...")
    df_mutations = read_mutations(mutations, gene)
    df_mutations.name = sample_name
    df_mutations.to_csv(
        output_folder / f"mutations/{sample_name}_mutations.tsv", index=False, sep="\t"
    )
    df_mutations.to_pickle(output_folder / f"mutations/{sample_name}_mutations.pkl")
    print(f"{fGetTime()} Done")

    print(f"{fGetTime()} Calculating mutation counts...")
    df_counts, num_singles, num_multiples = count_mutations(df_mutations, gene, quality_filter)
    with open(output_folder / "multiple_mutants.tsv", "a") as f:
        f.write(f"{sample_name}\t{num_singles}\t{num_multiples}\n")
    df_counts.to_csv(output_folder / f"counts/{sample_name}_counts.tsv", sep="\t")
    df_counts.to_pickle(output_folder / f"counts/{sample_name}_counts.pkl")
    print(f"{fGetTime()} Done")

    total_runtime = round(time.time() - start_time, 3)
    print(f"Time: {total_runtime} seconds")

    print("-" * 50)


if __name__ == "__main__":
    main()

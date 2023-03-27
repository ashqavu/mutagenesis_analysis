#!/usr/bin/env python
"""
This script takes alignments in BAM format and then uses a GenBank sequence as reference
for finding mutations. All mutations, counts, and other results are reported as .csv tables
in an output folder.
"""
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
from natsort import natsorted

from plasmid_map import Gene


def parse_args():
    """
    Argument parser
    """
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


def get_time() -> str:
    """
    Returns
    -------
    str
        Get current time
    """
    return f"""[{time.strftime("%H:%M:%S")}]"""

# @profile
def mutation_finder(alignments, gene: Gene) -> list:
    """
    Find all mutations present in all alignments using Gene as reference

    Parameters
    ----------
    alignments : iterable
        Alignments to find mutations for
    gene : Gene
        Gene providing reference coding sequence

    Returns
    -------
    mutations : list
        All mutations found in alignments
    """
    cds_start = gene.cds.location.start
    cds_end = gene.cds.location.end
    insertions = []
    deletions = []
    wildtypes = []
    mutations = []
    for aln in alignments:
        # * record and discard indels (mostly deletions from AT-rich regions of gene)
        if any(a == 1 for a, _ in aln.cigartuples):
            insertions.append(aln)
        #     continue
        elif any(a == 2 for a, _ in aln.cigartuples):
            deletions.append(aln)
        #     continue

        # * lowercase letter indicates substitution
        if aln.get_reference_sequence().isupper():
            wildtypes.append(aln)
        #     continue

        # * get full length of reference sequence for alignment
        # ref_positions = aln.get_reference_positions(full_length=True)
        # ref_seq = plasmid_seq[ref_positions[0]:ref_positions[-1]]
        # if aln.query_sequence == ref_seq:
        #     wildtypes.append(aln)
        #     continue

        # * iterate over aligned pairs to find mutation
        aligned_pairs = aln.get_aligned_pairs(matches_only=False, with_seq=True)

        for query_pos, ref_pos, ref_base in aligned_pairs:
            if ref_pos is None:
                continue
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
                        aln.get_overlap(start=cds_start, end=cds_end),
                    )
                )
    print(f"{len(insertions):,} sequences found with insertions")
    print(f"{len(deletions):,} sequences found with deletions")
    print(f"{len(wildtypes):,} sequences found with wild-type")
    return mutations


def read_mutations(mutations: list, gene: Gene) -> pd.DataFrame:
    """
    Take list of nucleotide mutations found and determine query/reference codons,
    amino acids, reference positions, etc.

    Parameters
    ----------
    mutations : list
        Nucleotide retrieved from alignment files
    gene : Gene
        Gene with wild-type sequences

    Returns
    -------
    df : pd.DataFrame
        Mutation table
    """
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

    # * find position of codon's residue in protein
    df["aa_pos"] = (
        df["ref_pos"].sub(gene.cds.location.start).floordiv(3).astype("Int16")
    )
    df["ref_codon"] = (
        df["aa_pos"].map(gene.cds_codon_dict).astype("string").astype("category")
    )
    df["ref_aa"] = df["ref_codon"].map(translation_table).astype("category")
    # * pull up position for the first base of the codon of the nucleotide
    df["codon_pos"] = df["aa_pos"].map(gene.codon_starts).astype("Int16")
    # * adjust the codon position from the read to set the reading frame
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


def find_multiple_mutants(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Divide DataFrame into read ids with multiple mutations found and read ids with single
    mutations

    Parameters
    ----------
    df : pd.DataFrame
        Mutation table

    Returns
    -------
    df_multiples, df_singles : tuple[pd.DataFrame, pd.DataFrame]
    """
    df_multiples = df[df.duplicated("read_id", keep=False)]
    df_singles = df.drop_duplicates("read_id", keep=False)
    return df_multiples, df_singles


def count_mutations(df: pd.DataFrame, gene: Gene) -> tuple[pd.DataFrame, int, int]:
    """
    Count up how many of each mutant are present in mutation table and present as a
    single DataFrame. Also determine how many single mutants and how many multiple
    mutants are present

    Parameters
    ----------
    df : pd.DataFrame
        Mutation table
    gene : Gene
        Gene with wild-type sequences

    Returns
    -------
    df_counts, num_singles, num_mutants : tuple[pd.DataFrame, int, int]
        DataFrame with count values for each mutation as well as the number of single
        mutants and the number of multiple mutants
    """
    # * filter redundant mutations (when all mutated bases are in same codon)
    # TODO: change to group the multiple mutations into one rdecord instead of straight dropping
    df = df.drop_duplicates(["read_id", "aa_pos"])

    df_multiples, df_singles = find_multiple_mutants(df)
    num_singles = df_singles.shape[0]
    num_multiples = df_multiples["read_id"].unique().shape[0]

    print(f"Number of reads with one mutation passing quality check: {num_singles:,}")
    print(
        f"Number of reads with multiple mutations passing quality check: {num_multiples:,}"
    )
    print(
        f"Number of single mutants in CDS region passing quality check: {df_singles.dropna().shape[0]:,}"
    )

    df_counts = pd.DataFrame(
        index=np.arange(len(gene.cds_translation)),
        columns=list(IUPACData.protein_letters + "*∅"),
    )
    # ! only counting single mutants
    for (pos, aa), count in (
        df_singles.groupby(["aa_pos", "query_aa"], observed=True).size().items()
    ):
        if gene.cds_translation[pos] == aa:
            df_counts.loc[pos, "∅"] = count
        else:
            df_counts.loc[pos, aa] = count
    df_counts.fillna(0, inplace=True)
    # df_counts.index = pd.Index(gene.numbering_scheme, dtype="int64")

    return df_counts, num_singles, num_multiples


def main() -> None:
    """
    Main function
    """
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

    # make sure folders exists
    if not os.path.exists(output_folder / "counts"):
        os.makedirs(output_folder / "counts")
    if not os.path.exists(output_folder / "mutations/quality_filtered/seq_lengths"):
        os.makedirs(output_folder / "mutations/quality_filtered/seq_lengths")

    start_time = time.time()

    print(f"{get_time()} Finding mutations for {sample_name}...")
    with pysam.AlignmentFile(input_file, "rb", threads=os.cpu_count()) as bam: # pylint: disable=no-member
        if args.contig:
            contig = args.contig
        else:
            contig = bam.header.references[0]
        num_alignments = int(
            pysam.view( # pylint: disable=no-member
                # restrict search to gene region
                # "-c", input_file.as_posix(), f"{contig}:{cds_start+1}-{cds_end}"
                "-c", input_file.as_posix()
            ).strip()
        )
        alns = tqdm(
            bam.fetch(contig=contig, start=cds_start, stop=cds_end),
            total=num_alignments,
            file=sys.stdout,
        )
        mutations = mutation_finder(alns, gene)
    print(f"{get_time()} Done")

    print(f"{get_time()} Generating mutations table...")
    df_mutations = read_mutations(mutations, gene)
    df_mutations.name = sample_name
    df_mutations.to_csv(
        output_folder / f"mutations/{sample_name}_all_mutations.tsv",
        index=False,
        sep="\t",
    )
    df_mutations.to_pickle(output_folder / f"mutations/{sample_name}_all_mutations.pkl")

    # do a quality check
    df_quality_filter = df_mutations.query("base_quality >= @quality_filter")
    df_quality_filter.to_csv(
        output_folder
        / f"mutations/quality_filtered/{sample_name}_filtered_mutations.tsv",
        index=False,
        sep="\t",
    )
    df_quality_filter.to_pickle(
        output_folder
        / f"mutations/quality_filtered/{sample_name}_filtered_mutations.pkl"
    )
    print(f"{get_time()} Done")

    # ! analysis performed on prefiltered data
    print(f"{get_time()} Calculating lengths of mapped query sequences...")
    df_read_lengths = (
        df_quality_filter[["read_id", "query_seq"]]
        .drop_duplicates("read_id")
        .set_index("read_id")["query_seq"]
        .transform(len)
    )
    df_read_lengths.name = sample_name
    df_read_lengths.to_csv(
        output_folder
        / f"mutations/quality_filtered/seq_lengths/{sample_name}_filtered_seq_lengths.tsv",
        index=False,
        sep="\t",
    )
    df_read_lengths.to_pickle(
        output_folder
        / f"mutations/quality_filtered/seq_lengths/{sample_name}_filtered_seq_lengths.pkl"
    )
    print(f"{get_time()} Done")

    print(f"{get_time()} Calculating mutation counts...")
    print(f"Number of mutations found: {df_mutations.shape[0]:,}")
    print(
        f"{df_quality_filter.shape[0]:,} ({df_quality_filter.shape[0] / df_mutations.shape[0]:.2%}) of all nucleotide mutations found passed with quality scores >= {quality_filter}"
    )
    df_counts, num_singles, num_multiples = count_mutations(df_quality_filter, gene)
    with open(
        output_folder / "mutations/quality_filtered/multiple_mutants.tsv",
        "a+",
        encoding="utf-8",
    ) as f:
        header = "sample_name\tnum_singles\tnum_multiples\n"
        if f.readline() != header:
            f.write(header)
        f.write(f"{sample_name}\t{num_singles}\t{num_multiples}\n")
    with open(
        output_folder / "mutations/quality_filtered/multiple_mutants.tsv",
        "r",
        encoding="utf-8",
    ) as f:
        sorted_lines = "".join(natsorted(f.readlines()[1:]))
    # ! not sure if this works, check this
    with open(
        output_folder / "mutations/quality_filtered/multiple_mutants.tsv",
        "w",
        encoding="utf-8",
    ) as f:
        header = "sample_name\tnum_singles\tnum_multiples\n"
        f.write(header)
        f.write(sorted_lines)
    df_counts.to_csv(output_folder / f"counts/{sample_name}_counts.tsv", sep="\t")
    df_counts.to_pickle(output_folder / f"counts/{sample_name}_counts.pkl")
    print(f"{get_time()} Done")

    total_runtime = round(time.time() - start_time, 3)
    print(f"Time: {total_runtime} seconds")

    print("-" * 50)


if __name__ == "__main__":
    main()

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

import pysam
from tqdm import tqdm
from natsort import natsorted

from plasmid_map import Gene
from utils.seq_mapping_utils import get_time, mutation_finder, read_mutations, count_mutations


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

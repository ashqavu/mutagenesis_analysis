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

from plasmid_map import Gene
from utils.seq_mapping_utils import (
    count_mutations,
    count_wildtype,
    get_time,
    mutation_finder,
    read_mutations,
    revert_poor_quality
)


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

    input_file = Path(args.bam)
    input_folder = input_file.parent.parent
    sample_name = input_file.stem

    if args.output:
        output_folder = args.output
    else:
        output_folder = input_folder / "results"

    # make sure folders exist
    (output_folder / "counts").mkdir(parents=True, exist_ok=True)
    (output_folder / "mutations").mkdir(parents=True, exist_ok=True)
    (output_folder / "wildtypes").mkdir(parents=True, exist_ok=True)
    (output_folder / "mutations/quality_filtered/seq_lengths").mkdir(parents=True, exist_ok=True)

    start_time = time.time()

    print(f"{get_time()} Finding all base mutations for {sample_name}...")
    with pysam.AlignmentFile(  # pylint: disable=no-member
        input_file, "rb", threads=os.cpu_count()
    ) as bam:  # pylint: disable=no-member
        if args.contig:
            contig = args.contig
        else:
            contig = bam.header.references[0]
        num_alignments = int(
            pysam.view(  # pylint: disable=no-member
                "-c",
                input_file.as_posix(),
                # restrict search to gene region
                # f"{contig}:{cds_start+1}-{cds_end}",
                f"{contig}",
            ).strip()
        )
        with open(
            input_folder / "alignments/total_reads.csv", "a", encoding="utf-8"
        ) as f:
            f.write(f"{sample_name},{num_alignments}\n")
        alns = tqdm(
            bam.fetch(contig=contig),# start=cds_start, stop=cds_end),
            total=num_alignments,
            file=sys.stdout,
        )
        wildtype, mutations = mutation_finder(alns, gene)
    print(f"{get_time()} Done")

    print(f"{get_time()} Generating wild-types table...")
    df_wildtypes = read_mutations(wildtype, gene)
    df_wildtypes.name = sample_name
    df_wildtypes.to_csv(
        output_folder / f"wildtypes/{sample_name}_all_wildtype_positions.csv",
        index=False
    )
    df_wildtypes.to_pickle(output_folder / f"wildtypes/{sample_name}_all_wildtype_positions.pkl")
    print(f"{get_time()} Done.")

    print(f"{get_time()} Generating mutations table...")
    df_mutations = read_mutations(mutations, gene)
    df_mutations.name = sample_name
    df_mutations.to_csv(
        output_folder / f"mutations/{sample_name}_all_mutations.csv",
        index=False
    )
    df_mutations.to_pickle(output_folder / f"mutations/{sample_name}_all_mutations.pkl")
    print(f"{get_time()} Done.")

    print(f"{get_time()} Filtering poor quality base mutations...")
    df_quality_filter = revert_poor_quality(df_mutations, quality_filter=quality_filter)
    df_quality_filter.to_csv(
        output_folder
        / f"mutations/quality_filtered/{sample_name}_filtered_mutations.csv",
        index=False
    )
    df_quality_filter.to_pickle(
        output_folder
        / f"mutations/quality_filtered/{sample_name}_filtered_mutations.pkl"
    )
    print(f"{get_time()} Done.")

    # ! analysis performed on prefiltered data
    print(f"{get_time()} Calculating lengths of all mapped and filtered query sequences...")
    df_read_lengths = (
        df_quality_filter[["read_id", "query_seq"]]
        .drop_duplicates("read_id", keep="first")
        .set_index("read_id")["query_seq"]
        .transform(len)
    )
    df_read_lengths.name = sample_name
    df_read_lengths.to_csv(
        output_folder
        / f"mutations/quality_filtered/seq_lengths/{sample_name}_filtered_seq_lengths.csv",
        index=False
    )
    df_read_lengths.to_pickle(
        output_folder
        / f"mutations/quality_filtered/seq_lengths/{sample_name}_filtered_seq_lengths.pkl"
    )
    print(f"{get_time()} Done")

    print(f"{get_time()} Calculating base mutation counts...")
    print(f"Number of base mutations found in whole plasmid: {df_mutations.shape[0]:,}")
    print(
        f"{df_quality_filter.shape[0]:,} ({df_quality_filter.shape[0] / df_mutations.shape[0]:.2%}) of all mutations in whole plasmid found passed with quality scores >= {quality_filter}"
    )
    df_mutations_cds = df_mutations.dropna(subset="ref_codon")
    df_quality_filter_cds = df_quality_filter.dropna(subset="ref_codon")
    print(f"Number of base mutations found in CDS region: {df_mutations_cds.shape[0]:,}")
    print(
        f"{df_quality_filter_cds.shape[0]:,} ({df_quality_filter_cds.shape[0] / df_mutations_cds.shape[0]:.2%}) of all mutations in CDS region found passed with quality scores >= {quality_filter}"
    )

    print(f"{get_time()} Calculating amino acid mutation counts after quality filtering...")
    df_mutation_counts, num_mutation_singles, num_mutation_multiples = count_mutations(df_quality_filter_cds, gene)
    print(f"Number of sequences with one amino acid mutation in CDS passing quality check: {num_mutation_singles:,}")
    print(f"Number of sequences with multiple amino acid mutations in CDS passing quality check: {num_mutation_multiples:,}")

    df_wildtype_counts = count_wildtype(df_wildtypes, gene)
    df_mutation_counts = df_mutation_counts.add(df_wildtype_counts)

    with open(
        output_folder / "mutations/quality_filtered/multiple_mutants.csv",
        "a+",
        encoding="utf-8",
    ) as f:
        f.write(f"{sample_name},{num_mutation_singles},{num_mutation_multiples}\n")

    df_mutation_counts.to_csv(output_folder / f"counts/{sample_name}_counts.csv")
    df_mutation_counts.to_pickle(output_folder / f"counts/{sample_name}_counts.pkl")
    print(f"{get_time()} Done")

    total_runtime = round(time.time() - start_time, 3)
    print(f"Time: {total_runtime} seconds")

    print("-" * 50)


if __name__ == "__main__":
    main()

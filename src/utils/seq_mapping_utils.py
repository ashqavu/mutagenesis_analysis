#!/usr/bin/env python
"""
Utils functions for counting mutation data from alignment files
"""
import sys
import time
# from typing import List

from Bio.Data import CodonTable, IUPACData
import numpy as np
import pandas as pd
from tqdm import tqdm

from plasmid_map import Gene

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


def mutation_finder(
    alignments, gene: Gene
) -> list[str, int, str, int, str, str, str, int]:
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
    mutations : list[str, int, str, int, str, str, str, int]
        All mutations found in alignments with data for read_id, nucleotide position in
        reference, reference nucleotide, nucleotide position in query sequence,
        query nucleotide, query base quality, full query sequence, and number of nucleotides
        overlapping the CDS region of the gene
    """
    cds_start = gene.cds.location.start
    cds_end = gene.cds.location.end
    insertions = []
    deletions = []
    wildtypes = []
    wildtypes_table = []
    mutations = []
    for aln in alignments:
        # * if alignment doesn't overlap with CDS region for gene
        if aln.get_reference_positions()[0] < cds_start or aln.get_reference_positions()[0] > cds_end:
            continue

        # * record indels (mostly deletions from AT-rich regions of gene)
        if any(a == 1 for a, _ in aln.cigartuples):
            insertions.append(aln)
        if any(a == 2 for a, _ in aln.cigartuples):
            deletions.append(aln)

        # * lowercase letter indicates substitution; all upper-case indicates
        # * a wild-type sequence
        is_wildtype = aln.get_reference_sequence().isupper()
        if is_wildtype:
            wildtypes.append(aln)

        # # * iterate over aligned pairs to find mutation
        aligned_pairs = aln.get_aligned_pairs(matches_only=False, with_seq=True)
        for query_pos, ref_pos, ref_base in aligned_pairs:
            if ref_pos is None or query_pos is None:
                continue
            sequence_entry = (
                aln.query_name,
                ref_pos,
                ref_base,
                query_pos,
                aln.query_sequence[query_pos],
                aln.query_qualities[query_pos],
                aln.query_sequence,
                aln.get_overlap(start=cds_start, end=cds_end),
            )
            if is_wildtype:
                wildtypes_table.append(sequence_entry)
            elif is_wildtype is False and ref_base.islower():
                mutations.append(sequence_entry)

    df_wildtype = pd.DataFrame(
        wildtypes_table,
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

    df_mutations = pd.DataFrame(
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

    print(f"{len(insertions):,} sequences found with insertions before quality filtering")
    print(f"{len(deletions):,} sequences found with deletions before quality filtering")
    print(f"{df_wildtype.drop_duplicates('read_id', keep='first').shape[0]:,} sequences found with wild-type CDS before quality filtering")
    return df_wildtype, df_mutations


def read_mutations(
    df_mutations: pd.DataFrame,
    gene: Gene,
) -> pd.DataFrame:
    """
    Take list of nucleotide mutations found and determine query/reference codons,
    amino acids, reference positions, etc.

    Parameters
    ----------
    df_mutations: pd.DataFrame
        DataFrame of mutation records
    gene : Gene
        Gene with wild-type sequences

    Returns
    -------
    df : pd.DataFrame
        Mutation table
    """
    # set dtypes, we're trying to save memory
    # if you have a bigger genome you might have to upcast ref_pos to Int32
    df = df_mutations.astype(
        {
            "read_id": "string",
            "ref_pos": "Int16",
            "ref_base": "string",
            "query_base": "string",
            "base_quality": "Int8",
            "query_seq": "string",
        }
    )
    df = df.astype({"ref_base": "category", "query_base": "category"})
    df = df.sort_values(["read_id", "ref_pos"])

    # * set categories for amino acid list
    residue_category = pd.api.types.CategoricalDtype(categories=set(translation_table.values()))
    codon_category = pd.api.types.CategoricalDtype(categories=translation_table.keys())

    # * find position of codon's residue in protein
    df["aa_pos"] = (
        df["ref_pos"].sub(gene.cds.location.start).floordiv(3).astype("Int16")
    )
    df["ref_codon"] = (
        df["aa_pos"].map(gene.cds_codon_dict).astype("string").astype(codon_category)
    )
    df["ref_aa"] = df["ref_codon"].map(translation_table).astype(residue_category)
    # * pull up position for the first base of the codon of the nucleotide
    df["ref_codon_start"] = df["aa_pos"].map(gene.codon_starts).astype("Int16")
    # * adjust the codon position from the read to set the reading frame
    df["query_codon_start"] = df["query_pos"].add(df["ref_codon_start"] - df["ref_pos"])
    df["query_codon"] = [
        seq[pos : pos + 3] if pd.notnull(pos) else pd.NA
        for seq, pos in zip(df["query_seq"], df["query_codon_start"])
    ]
    df["intra_codon_pos"] = df["ref_pos"] - df["ref_codon_start"]
    df["query_codon"] = df["query_codon"].astype("string").astype(codon_category)
    df["query_aa"] = (
        df["query_codon"].map(translation_table).astype("string").astype(residue_category)
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
            "ref_codon_start",
            "ref_aa",
            "query_codon",
            "query_codon_start",
            "query_aa",
            "query_seq",
            "cds_overlap_length",
            "intra_codon_pos"
        ]
    ]
    df = df.drop_duplicates(["read_id", "ref_pos"])
    return df


def revert_poor_quality(df: pd.DataFrame, quality_filter: int = 30) -> pd.DataFrame:
    """
    Filter/revert mutations from base calls with poor quality

    Parameters
    ----------
    df : pd.DataFrame
        Mutation info table
    quality_filter : int, optional
        Minimum base quality score, by default 30

    Returns
    -------
    pd.DataFrame
        Mutation info table with reverted query codons
    """
    df_reverted = df.copy(deep=True)
    read_ref_codon_groups = df_reverted.groupby(["read_id", "ref_codon_start"])
    for _, data in tqdm(
        read_ref_codon_groups, total=len(read_ref_codon_groups.size()), file=sys.stdout
    ):
        # * if all bases pass quality check
        if all(data["base_quality"].ge(quality_filter)):
            continue
        # * if there is only one mutation and it is also low quality we will drop it later
        if len(data["base_quality"]) == 1:
            # df = df.drop(data.index)
            continue
        ref_codon = data["ref_codon"].values[0]
        query_codon = data["query_codon"].values[0]
        # * base unclear from sequencing so residue cannot be determined
        if isinstance(query_codon, float):
            data["query_codon"] = [
                seq[pos : pos + 3] if pd.notnull(pos) else pd.NA
                for seq, pos in zip(data["query_seq"], data["query_codon_start"])
            ]
            query_codon = data["query_codon"].values[0]
        intra_codon_positions_list = data["intra_codon_pos"].agg(list)
        base_quality_list = data["base_quality"].agg(list)
        corrected_query_codon = list(ref_codon)
        for pos, quality in zip(intra_codon_positions_list, base_quality_list):
            if quality >= quality_filter:
                corrected_query_codon[pos] = query_codon[pos]
        corrected_query_codon = "".join(corrected_query_codon)

        try:
            df_reverted.loc[data.index, "query_aa"] = translation_table[corrected_query_codon]
            df_reverted.loc[data.index, "query_codon"] = corrected_query_codon
        except KeyError:
            # * shouldn't be used anymore because of the category dtype but included as a safety net
            df_reverted.loc[data.index, "query_aa"] = pd.NA
            df_reverted.loc[data.index, "query_codon"] = pd.NA
    df_reverted = df_reverted.drop(df_reverted.query("base_quality < @quality_filter").index)
    num_wildtype_flips = df_reverted[df_reverted['ref_codon'] == df_reverted['query_codon']]
    num_wildtype_flips = num_wildtype_flips.drop_duplicates(["read_id", "aa_pos"]).shape[0]
    print(f"{num_wildtype_flips} mutated codons flipped back to wild-type after quality filter")
    df_reverted = df_reverted.drop(df_reverted[df_reverted["ref_codon"] == df_reverted["query_codon"]].index)
    return df_reverted



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
    df_multiples : pd.DataFrame
        Dataframe containing mutations found in read_ids with multiple amino acid mutations
    df_singles : pd.DataFrame
        Dataframe containing mutations found in read_ids with only one amino acid mutation
    """
    # * filter to only mutations in CDS region
    df = df.dropna(subset=["ref_aa", "query_codon"])
    # * filter redundant mutations (when all mutated bases are in same codon)
    df = df.drop_duplicates(["read_id", "aa_pos"])
    df_singles = df.drop_duplicates("read_id", keep=False)
    df_multiples = df[df.duplicated("read_id", keep=False)]
    return df_singles, df_multiples


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
    df_counts : pd.DataFrame
        Dataframe with count values for each mutation
    num_singles : int
        Number of single mutants
    num_mutants : int
        Number of multiple mutants
    """
    # * filter to only mutations in CDS region
    df = df.dropna(subset=["ref_aa", "query_codon"])
    # TODO: change to group the multiple mutations into one record instead of straight dropping
    # * filter redundant mutations (when all mutated bases are in same codon)
    df = df.drop_duplicates(["read_id", "aa_pos"])

    df_singles, df_multiples = find_multiple_mutants(df)
    num_singles = df_singles.shape[0]
    num_multiples = df_multiples["read_id"].unique().shape[0]

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


def count_wildtype(df: pd.DataFrame, gene: Gene) -> pd.DataFrame:
    """
    Fill in wild-type positions with total number from sequences in CDS region

    Parameters
    ----------
    df : pd.DataFrame
        Wild-type data table
    gene : Gene
        Gene with wild-type sequence annotated

    Returns
    -------
    df_counts : pd.DataFrame
        Dataframe with count values for each mutation
    """
    df = df.dropna(subset=["ref_aa", "query_codon"])
    # * filter redundant mutations (when all mutated bases are in same codon)
    df = df.drop_duplicates(["read_id", "aa_pos"])
    
    df_counts = pd.DataFrame(
        index=np.arange(len(gene.cds_translation)),
        columns=list(IUPACData.protein_letters + "*∅"),
    )
    for (pos, aa), count in (
        df.groupby(["aa_pos", "query_aa"], observed=True).size().items()
    ):
        df_counts.loc[pos, aa] = count
    df_counts.fillna(0, inplace=True)

    return df_counts

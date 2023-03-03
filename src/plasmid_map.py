#!/usr/bin/env python
"""
This script creates a class that holds relevant information about the gene
(e.g. sequence, translated peptide, codon positions) extracted from the GenBank
file provided.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from Bio import SeqIO
import Bio


@dataclass
class Gene:
    """
    Object class built on the basis of Biopython's methods. Attributes
    gbk_record and cds refer to the Biopython object classes. Genes have coding
    sequences (cds_seq), translated proteins (cds_translation), and nucleotide
    positions indicating the beginning of each codon in the reading frame of
    the gene.
    """
    _gbk_record: str
    gene_name: str

    def get_feature_type(self, SeqRecord: SeqIO.SeqRecord, feature_type: str) -> Bio.SeqFeature:
        """
        From plasmid map, retrieve gene features to extract gene sequencing data from.

        Parameters
        ----------
        SeqRecord : SeqIO.SeqRecord
            Record from annotated GenBank file
        feature_type : str
            What type of feature to extract from the sequence (mostly gene features)

        Returns
        -------
        Bio.SeqFeature
        """
        for f in SeqRecord.features:
            if f.type == feature_type:
                for key, value in f.qualifiers.items():
                    if key == "gene" and self.gene_name in value:
                        return f

    @property
    def gbk_record(self) -> SeqIO.SeqRecord:
        """
        Record of full plasmid map from annotated .gbk sequence

        Returns
        -------
        SeqIO.SeqRecord
        """
        return SeqIO.read(self._gbk_record, "gb")

    @property
    def cds(self) -> Bio.SeqFeature:
        """
        Biopython feature from annotated .gbk sequence of coding region of the
        indicated gene

        Returns
        -------
        Bio.SeqFeature
        """
        if any(f.type == "CDS" for f in self.gbk_record.features):
            return self.get_feature_type(self.gbk_record, "CDS")

    @property
    def cds_seq(self) -> SeqIO.SeqRecord:
        """
        Nucleotide sequence of coding-region of gene

        Returns
        -------
        SeqIO.SeqRecord
        """
        if any(f.type == "CDS" for f in self.gbk_record.features):
            return self.cds.extract(self.gbk_record.seq)

    @property
    def cds_translation(self) -> SeqIO.SeqRecord:
        """
        Translated protein sequence of gene coding region

        Returns
        -------
        SeqIO.SeqRecord
        """
        if any(f.type == "CDS" for f in self.gbk_record.features):
            return self.cds_seq.translate()

    @property
    def codon_starts(self) -> pd.Series:
        """
        Nucleotide positions of each codon start position

        Returns
        -------
        pd.Series
        """
        if any(f.type == "CDS" for f in self.gbk_record.features):
            return pd.Series(
                np.arange(self.cds.location.start, self.cds.location.end, 3)
            )

    @property
    def cds_codon_dict(self) -> dict:
        """
        Dictionary matching codon-index to codon nucleotide position

        Returns
        -------
        dict
        """
        if any(f.type == "CDS" for f in self.gbk_record.features):
            cds_codons = [
                self.cds_seq[i : i + 3] for i in range(0, len(self.cds_seq), 3)
            ]
            return {idx: value for idx, value in enumerate(cds_codons)}


@dataclass
class TEM1_gene(Gene):
    """
    Bases: Gene

    Class specific for our blaTEM-1, which has a signal peptide that is cleaved
    from the mature protein (mat_peptide) and an odd residue-numbering system
    (Ambler et al., 1991). Also holds what positions are covered by each
    sublibrary in our mutagenesis library.
    """
    gene_name: str = "blaTEM-1"

    @property
    def mat_peptide(self) -> Bio.SeqFeature:
        """
        'Mature peptide' feature from annotated GenBank file

        Returns
        -------
        Bio.SeqFeature
        """
        return self.get_feature_type(self.gbk_record, "mat_peptide")

    @property
    def ambler_numbering(self) -> list:
        """
        Class A beta-lactamases have a standardized numbering schemes, which
        results in each beta-lactamase gene having weird start/stop numbers and
        also random number skips in the middle of the gene. Literature will
        reference the Ambler numbers and not the positional index numbers.

        Returns
        -------
        list
        """
        return list(range(3, 239)) + list(range(240, 253)) + list(range(254, 292))

    @property
    def sublibrary_positions(self) -> dict:
        """
        Our mutagenesis library is divided into 10 sublibraries covering
        specific mutations, as indicated here.

        Returns
        -------
        dict
        """
        sublibrary_names = [
            "MAS5",
            "MAS6",
            "MAS7",
            "MAS8",
            "MAS9",
            "MAS10",
            "MAS11",
            "MAS12",
            "MAS13",
            "MAS14",
        ]

        # amino acid positions covered by each sublibrary
        # e.g. MAS5 covers positions 26-51 in the gene
        MAS5 = list(range(26, 52))
        MAS6 = list(range(52, 79))
        MAS7 = list(range(79, 105))
        MAS8 = list(range(105, 133))
        MAS9 = list(range(133, 157))
        MAS10 = list(range(157, 184))
        MAS11 = list(range(184, 210))
        MAS12 = list(range(210, 237))
        MAS13 = list(range(237, 239)) + list(range(240, 253)) + list(range(254, 265))
        MAS14 = list(range(265, 292))

        return dict(
            zip(
                sublibrary_names,
                [MAS5, MAS6, MAS7, MAS8, MAS9, MAS10, MAS11, MAS12, MAS13, MAS14],
            )
        )

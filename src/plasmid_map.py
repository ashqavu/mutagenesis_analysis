#!/usr/bin/env python
from dataclasses import dataclass

import numpy as np
import pandas as pd
from Bio import SeqIO

"""
This script creates a class that holds relevant information about the gene (e.g. sequence, translated peptide, codon positions) extracted from the GenBank file provided.
"""


@dataclass
class Gene:
    _gbk_record: str
    gene_name: str

    def get_feature_type(self, SeqRecord: SeqIO.SeqRecord, feature_type: str):
        for f in SeqRecord.features:
            if f.type == feature_type:
                for key, value in f.qualifiers.items():
                    if key == "gene" and self.gene_name in value:
                        return f

    @property
    def gbk_record(self):
        return SeqIO.read(self._gbk_record, "gb")

    @property
    def cds(self):
        if any(f.type == "CDS" for f in self.gbk_record.features):
            return self.get_feature_type(self.gbk_record, "CDS")

    @property
    def cds_seq(self):
        if any(f.type == "CDS" for f in self.gbk_record.features):
            return self.cds.extract(self.gbk_record.seq)

    @property
    def cds_translation(self):
        if any(f.type == "CDS" for f in self.gbk_record.features):
            return self.cds_seq.translate()

    @property
    def codon_starts(self):
        if any(f.type == "CDS" for f in self.gbk_record.features):
            return pd.Series(
                np.arange(self.cds.location.start, self.cds.location.end, 3)
            )

    @property
    def cds_codon_dict(self):
        if any(f.type == "CDS" for f in self.gbk_record.features):
            cds_codons = [
                self.cds_seq[i : i + 3] for i in range(0, len(self.cds_seq), 3)
            ]
            return {idx: value for idx, value in enumerate(cds_codons)}


@dataclass
class TEM1_gene(Gene):
    gene_name: str = "blaTEM-1"

    @property
    def mat_peptide(self):
        return self.get_feature_type(self.gbk_record, "mat_peptide")

    @property
    def ambler_numbering(self):
        return list(range(3, 239)) + list(range(240, 253)) + list(range(254, 292))

    @property
    def sublibrary_positions(self):
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
#!/usr/bin/env python
from Bio import SeqIO
import numpy as np
import pandas as pd


class Gene:
    def __init__(self, genbank_path, gene_name: str):
        self.gbk_record = SeqIO.read(genbank_path, "gb")
        self.gene_name = gene_name
        if any(f.type == "CDS" for f in self.gbk_record.features):
            self.cds = self.get_feature_type(self.gbk_record, "CDS")
            self.cds_seq = self.cds.extract(self.gbk_record.seq)
            self.cds_translation = self.cds_seq.translate()
            self.codon_starts = pd.Series(
                np.arange(self.cds.location.start, self.cds.location.end, 3)
            )

            cds_codons = [
                self.cds_seq[i : i + 3] for i in range(0, len(self.cds_seq), 3)
            ]
            self.cds_codon_dict = {idx: value for idx, value in enumerate(cds_codons)}

    def get_feature_type(self, SeqRecord, feature_type):
        for f in SeqRecord.features:
            if f.type == feature_type:
                for key, value in f.qualifiers.items():
                    if key == "gene" and self.gene_name in value:
                        return f


class TEM1_gene(Gene):
    def __init__(self, genbank_path, gene_name: str = "blaTEM-1"):
        super().__init__(genbank_path, gene_name)

        self.mat_peptide = self.get_feature_type(self.gbk_record, "mat_peptide")
        self.numbering_scheme = self.get_numbering_scheme()
        self.sublibrary_positions = self.get_sublibrary_positions()

    def get_numbering_scheme(self):
        # beta-lactamase residue numbering is irregular for each protein
        # see Ambler et al. 1991
        numbering_scheme = (
            list(range(3, 239)) + list(range(240, 253)) + list(range(254, 292))
        )
        return numbering_scheme

    def get_sublibrary_positions(self):
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

        dict_sublibrary_positions = dict(
            zip(
                sublibrary_names,
                [MAS5, MAS6, MAS7, MAS8, MAS9, MAS10, MAS11, MAS12, MAS13, MAS14],
            )
        )
        return dict_sublibrary_positions

#!/usr/bin/env python
from Bio import SeqIO
import numpy as np
import pandas as pd
from dataclasses import dataclass
"""
This script creates a class that holds relevant information about the gene (e.g. sequence, translated peptide, codon positions) extracted from the GenBank file provided.
"""
@dataclass
class Gene:
    gbk_record: str
    gene_name: str
    
    def __post_init__(self):
        self.gbk_record = SeqIO.read(self.gbk_record, "gb")
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
            
    def get_feature_type(self, SeqRecord: SeqIO.SeqRecord, feature_type: str):
        for f in SeqRecord.features:
            if f.type == feature_type:
                for key, value in f.qualifiers.items():
                    if key == "gene" and self.gene_name in value:
                        return f


@dataclass
class TEM1_gene(Gene):
    """
    Beta-lactamases have an atypical residue numbering system created to standardize position numbers across all different classes of beta-lactamases. This class defines that numbering scheme. This class also adds additional information specifically relevant to our TEM-1 saturation mutagenesis library. This class defines the amino acid positions covered by each sublibrary and the mature protein with the signal peptide removed, as the signal sequence is not covered by the library.
    """
    gene_name: str = "blaTEM-1"
    
    def __post_init__(self):
        super().__post_init__()
        self.mat_peptide = self.get_feature_type(self.gbk_record, "mat_peptide")
        self.ambler_numbering = self.get_ambler_numbering()
        self.sublibrary_positions = self.get_sublibrary_positions()
        
    def get_ambler_numbering(self):
        # beta-lactamase residue numbering is irregular for each protein
        # see Ambler et al. 1991
        ambler_numbering = (
            list(range(3, 239)) + list(range(240, 253)) + list(range(254, 292))
        )
        return ambler_numbering

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

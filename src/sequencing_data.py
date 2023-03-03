#!/usr/bin/env python
"""
This script is written to parse the mutation data from the read sequences. It
provides a base SequencingData object class that holds information about the
dataset (e.g. sample names, reference gene) and calculates counts, frequency,
enrichment, and fitness data. Two additional subclasses are provided for
parsing different sequencing schemes, such as when the sublibraries are
submitted as one pooled library sample as opposed to separate sublibraries for
sequencing.
"""

import copy
import csv
import glob
import re
from pathlib import Path

from Bio.Data import IUPACData
import numpy as np
import pandas as pd
from natsort import natsorted
from scipy.stats import norm

from plasmid_map import Gene

# TODO: get_pairs, match_treated_untreated, filter_fitness_read_noise can probably go in a separate tools .py # pylint: disable=fixme,line-too-long
# TODO: fix @fitness to not be a property i guess


def get_pairs(treatment: str, samples: list) -> tuple[str, str]:
    """
    Given a drug, extract the replicas from the list of samples

    Parameters
    ----------
    treatment : str
        Drug to find replicates of
    samples : list
        Reference for fitness values of all samples

    Returns
    -------
    replica_one, replica_two : tuple[str, str]
        Strings of replica sample names
    """
    treatment_pair = [sample for sample in samples if treatment in sample]
    if not treatment_pair:
        raise KeyError(f"No fitness data: {treatment}")
    if len(treatment_pair) > 2:
        raise IndexError("Treatment has more than 2 replicates to compare")
    replica_one, replica_two = treatment_pair[0], treatment_pair[1]
    return replica_one, replica_two


def match_treated_untreated(sample: str) -> str:
    """
    Takes name of treated sample (e.g. CefX3) and matches it to the
    corresponding untreated sample name (UT3) for proper comparisons.

    Parameters
    ----------
    sample : str
        Name of sample

    Returns
    -------
    untreated : str
        Name of corresponding untreated smple
    """
    num = re.sub(r"[A-Za-z]*", "", sample)
    untreated = "UT" + num
    return untreated


def heatmap_table(gene: Gene) -> pd.DataFrame:
    """
    Returns DataFrame for plotting heatmaps with position indices and residue
    columns (ACDEFGHIKLMNPQRSTVWY*∅)

    Parameters
    ----------
    gene : Gene
        Gene object with translated protein sequence

    Returns
    -------
    df : pd.DataFrame
        DataFrame of Falses
    """
    df = pd.DataFrame(
        False,
        index=np.arange(len(gene.cds_translation)),
        columns=list(IUPACData.protein_letters + "*∅"),
    )
    return df


def heatmap_masks(gene: Gene) -> pd.DataFrame:
    """
    Returns a bool DataFrame with wild-type cells marked as True for heatmap
    plotting

    Parameters
    ----------
    gene : Gene
        Object providing translated protein sequence

    Returns
    -------
    df_wt : pd.DataFrame
        DataFrame to use for marking wild-type cells on heatmaps
    """
    df_wt = heatmap_table(gene)
    for position, residue in enumerate(gene.cds_translation):
        df_wt.loc[position, residue] = True
    return df_wt


def filter_fitness_read_noise(
    counts_dict: dict,
    fitness_dict: dict,
    read_threshold: int = 20,
) -> dict:
    """
    Takes DataFrames for treated sample and returns a new DataFrame with cells
    with untreated counts under the minimum read threshold filtered out

    Parameters
    ----------
    counts_dict : dict
        Reference with counts dataframes for all samples
    fitness_dict : dict
        Reference with fitness dataframes for all samples
    read_threshold : int, optional
        Minimum number of reads required to be included, by default 20

    Returns
    -------
    df_treated_filtered : dict
        Fitness tables with insufficient counts filtered out
    """
    dfs_filtered = {}
    for sample in sorted(fitness_dict):
        untreated = match_treated_untreated(sample)
        df_counts_untreated = counts_dict[untreated]
        df_counts_sample = counts_dict[sample]
        df_fitness_sample = fitness_dict[sample]
        dfs_filtered[sample] = df_fitness_sample.where(
            df_counts_sample.ge(read_threshold) & df_counts_untreated.ge(read_threshold)
        )
    return dfs_filtered


class SequencingData:
    r"""A class created to compile fitness data about each sequencing dataset.

    Parameters
    ---------
    gene : Gene
        Gene object from `plasmid_map`
    inputfolder : str
        Project folder
    read_threshold : int, optional
        Minimum of reads for a fitness value to be included, by default 20
    extinct_add : int, optional
        Amount to add to counts when calculating frequencies in order to separate
        out the extinct mutations, by default 0.001

    """

    def __init__(
        self,
        gene: Gene,
        inputfolder: str,
        read_threshold: int = 20,
        extinct_add: int = 0.001,
    ):
        self.gene = gene
        self._inputfolder = inputfolder
        self.read_threshold = read_threshold
        self.extinct_add = extinct_add
        self._samples = self._get_sample_names(self._inputfolder)
        self._treatments = self._get_treatments(self._samples)
        self._counts = self._get_counts(self._samples, self._inputfolder)
        self._total_reads = self._get_total_reads(self._inputfolder)
        self._frequencies = self._get_frequencies(
            self._counts, self._total_reads, self.extinct_add
        )
        if len([sample for sample in self.samples if "UT" in sample]) == 1:
            self._enrichment = self._get_enrichment(self._frequencies)
            self._fitness = self._get_fitness(self._enrichment)
        else:
            self._enrichment = None
            self._fitness = None

    def copy(self):
        """
        Copy method

        Returns
        -------
        self : SequencingData
        """

        return copy.deepcopy(self)

    def _get_sample_names(self, inputfolder: str) -> list:
        """
        Extract the names of the samples from the Illumina filenames in the raw_data folder

        Parameters
        ----------
        inputfolder : str
            Project folder

        Returns
        -------
        samples : list
            Names of samples submitted for sequencing
        """

        samples_file = Path(inputfolder) / "raw_data/SampleNames.txt"
        samples = []

        with open(samples_file, "r", encoding="utf-8") as file:
            for line in file:
                line = line.rstrip()
                name = line.split("\t")[1]
                samples.append(name)
        return natsorted(samples)

    def _get_treatments(self, samples: list) -> list:
        """
        Find treatment conditions from sample names

        Parameters
        ----------
        samples : list
            List of samples to pull treatments from

        Returns
        -------
        treatments : list
            List of treatments used in experiment
        """

        treatments = []
        for name in samples:
            treatment = re.sub(r"([^A-Za-z0-9])?\d+$", "", name)
            treatments.append(treatment)
        treatments = natsorted(list(set(treatments)))
        return treatments

    # keeps original numbers
    def _get_counts(self, sample_list: list, inputfolder: str) -> dict:
        """
        Load all matrices from the results folder as pandas.DataFrame objects
        and compile a dictionary

        Parameters
        ----------
        sample_list : list
            List of samples to get counts for
        inputfolder : str
            Project folder

        Returns
        -------
        counts_dict : dict
            Dictionary with sample names and dataframes
        """

        counts_dict = {}
        datafolder = Path(inputfolder) / "results/counts"

        filetype = "*.pkl"
        try:
            next(glob.iglob((datafolder / filetype).as_posix()))
        except StopIteration:
            filetype = "*.tsv"
        for sample in sample_list:
            for file in glob.iglob((datafolder / filetype).as_posix()):
                if sample in file:
                    if filetype == "*.pkl":
                        df = pd.read_pickle(file)
                    elif filetype == "*.tsv":
                        # * if dataframe wasn't pickled before saving
                        df = pd.read_table(file, index_col=0)

                    # ? determine whether to mask extinct mutations or not for mapping here
                    df.name = sample
                    counts_dict[sample] = df
        return dict(natsorted(counts_dict.items()))

    def _get_total_reads(self, inputfolder: str) -> dict:
        """
        Retrieve total number of reads found for each sample after alignment
        (by `samtools idxstats`) and create dictionary

        Parameters
        ----------
        inputfolder : str
            Data folder

        Returns
        -------
        counts_dict : dict
            Dictionary with sample names to total number of reads
        """

        reads_file = Path(inputfolder) / "alignments/total_reads.tsv"

        with open(reads_file, "r", encoding="utf-8") as csvfile:
            reader = csv.reader(csvfile, delimiter="\t")
            total_reads = {row[0]: int(row[1]) for row in reader}
        return total_reads

    # adds 0.001 / extinct_add, has signal peptide
    def _get_frequencies(
        self, counts: dict, total_reads: dict, extinct_add: int = 0.001
    ) -> dict:
        """
        Calculate frequecy (f) for mutation (i) by :math:`\frac{N^i}{N^{total reads}}`.

        When calculating frequencies, counts will have an addition of `extinct_add` to
        separate out mutations with untreated counts that go to below-threshold counts
        (i.e. extinct) in the untreated sample.

        Parameters
        ----------
        counts : dict
            Dataframes of counts for samples
        total_reads : dict
            Total number of reads in each sequencing run

        Returns
        -------
        frequencies : dict
            Dictionary with sample names as keys and dataframes as values
        """
        frequencies = dict.fromkeys(counts)
        for name in frequencies:
            adjusted_counts = counts[name].add(
                extinct_add  # ! added 0.001 here for extinct mutations
            )
            df_freqs = adjusted_counts.divide(total_reads[name])
            # if "UT" in name:
            #     df_freqs = df_freqs[adjusted_counts != 0]
            # ! we will now be filtering out counts with insufficient UT reads when
            # ! after calculating fitness in self._get_fitness
            df_freqs.name = name
            frequencies[name] = df_freqs
        return frequencies

    # used 0.001 added, has signal peptide
    def _get_enrichment(self, frequencies: dict) -> dict:
        """
        Calculate enrichment (e) of each mutation (i) by equation
        :math:`log_{10}(\frac{f^i{selected}}{f^i_{unselected}})`

        Parameters
        ----------
        frequencies : dict
            Dataframes of frequencies for samples

        Returns
        -------
        enrichment : dict
            Dictionary with sample names as keys and enrichment dataframes as values

        Raises
        ------
        LookupError
            Will not be able to calculate enrichment properly if there is more
            than one untreated sample to compare frequencies to
        """

        untreated = [x for x in frequencies if "UT" in x]
        num_untreated = len(untreated)
        if num_untreated > 1:
            raise LookupError("More than one untreated sample found in dataset")

        enrichment = {}
        for sample in frequencies:
            if "UT" in sample:
                continue
            untreated = match_treated_untreated(sample)
            df_untreated = frequencies[untreated]
            df_treated = frequencies[sample]
            df_enriched = df_treated.divide(df_untreated)
            with np.errstate(divide="ignore"):
                df_enriched = np.log10(df_enriched)
                df_enriched.name = sample
                enrichment[sample] = df_enriched
        return enrichment

    def _mask_untreated_zero(
        self, counts_dict: dict, fitness_df: pd.DataFrame, read_threshold: int = 20
    ) -> pd.DataFrame:
        """
        We are unable to really calculate a proper fitness value in cases where the number
        of observations in the untreated sample is 0 and the number of observations in the
        treated sample is greater than the read threshold (i.e. when a mutation is largely
        beneficial), so we hide those mutations from our values.

        Parameters
        ----------
        counts_dict : dict
            Reference with counts dataframes for all samples
        fitness_df : pd.DataFrame
            Fitness dataframe to mask
        read_threshold : int, optional
            Minimum of reads to be included, by default 20

        Returns
        -------
        fitness_Masked : pd.DataFrame
            Masked fitness dataframe
        """
        sample_name = fitness_df.name
        untreated = match_treated_untreated(sample_name)
        untreated_counts = counts_dict[untreated]
        treated_counts = counts_dict[sample_name]
        fitness_masked = fitness_df.mask(
            untreated_counts.lt(read_threshold) & treated_counts.ge(read_threshold)
        )
        return fitness_masked

    def _get_fitness(self, enrichment: dict) -> dict:
        """
        Calculate normalized fitness values (s) of each mutation (i) by subtracting the
        enrichment of synonymous wild-type mutations from the enrichment value of a
        mutation :math:`e^i - < e^{WT} >`

        Parameters
        ----------
        enrichment : dict
            Reference with enrichment dataframes for all samples

        Returns
        -------
        fitness : dict
            Dictionary with sample names as keys and fitness dataframes as values

        Raises
        ------
        LookupError
            Will not be able to calculate enrichment properly (and thus
            fitness) if there is more than one untreated sample to compare
            frequencies to
        """

        untreated = [x for x in enrichment if "UT" in x]
        num_untreated = len(untreated)
        if num_untreated > 1:
            raise LookupError("More than one untreated sample found in dataset")

        fitness = {}
        for sample in sorted(enrichment):
            if "UT" in sample:
                continue
            df_enriched = enrichment[sample]
            SynWT_enrichment = df_enriched["∅"]
            SynWT_mean, _ = norm.fit(SynWT_enrichment.dropna())
            normalized = df_enriched.subtract(SynWT_mean)
            normalized.name = sample
            fitness[sample] = normalized

            # ! mask out wild-type positions
            # fitness_masked = normalized.mask(heatmap_masks(self.gene))
            # fitness_masked.name = sample
            # * here mask where untreated counts are insufficient but treated counts
            # * are large (i.e. highly beneficial mutations)
            # fitness_masked = self._mask_untreated_zero(
            #     counts, fitness_masked, read_threshold=read_threshold
            # )
            # fitness_masked.name = sample
            # fitness[sample] = fitness_masked
        return fitness

    @property
    def samples(self) -> list:
        """Samples used in experiment"""

        return self._samples

    @samples.setter
    def samples(self, value):
        self._samples = value
        self._treatments = self._get_treatments(value)
        self._counts = self._get_counts(self._samples, self._inputfolder)
        self._total_reads = self._get_total_reads(self._inputfolder)
        self._frequencies = self._get_frequencies(
            self.counts, self.total_reads, self.extinct_add
        )
        if len([sample for sample in self.samples if "UT" in sample]) == 1:
            self._enrichment = self._get_enrichment(self._frequencies)
            self._fitness = self._get_fitness(self._enrichment)

    @property
    def treatments(self):
        """Treatments used in experiment"""

        return self._treatments

    @property
    def counts(self):
        """Counts DataFrame for each sample"""

        return self._counts

    @property
    def total_reads(self):
        """Total number of reads found in each sample"""

        return self._total_reads

    @property
    def frequencies(self):
        r"""Frequencies  DataFrame for each sample calculated for mutation (i) by
        :math:`\frac{N^i}{N^{total reads}}`
        """

        return self._frequencies

    @property
    def enrichment(self):
        r"""Enrichment DataFrame for each sample calculated for mutation (i) by
        :math:`log_{10}(\frac{f^i{selected}}{f^i_{unselected}})`
        """
        if self._enrichment is None:
            print("More than one untreated sample found in dataset, cannot calculate enrichment. Reset samples to recalculate.")
        return self._enrichment

    @property
    def fitness(self):
        r"""Fitness DataFrame for each sample calculated for mutation (i) by
        :math:`e^i - < e^{WT} >`
        """
        if self._fitness is None:
            print("More than one untreated sample found in dataset, cannot calculate enrichment. Reset samples to recalculate.")
        return self._fitness


class SequencingDataReplicates(SequencingData):
    """
    Class used to divide sequencing results into replicate datasets (n)
    """

    def __init__(
        self,
        gene: Gene,
        inputfolder: str,
        read_threshold: int = 20,
        extinct_add: int = 0.001,
    ):
        super().__init__(gene, inputfolder, read_threshold, extinct_add)
        enrichment_all = {}
        fitness_all = {}
        for replicate in self.replicate_numbers:
            replicate_data = self.get_replicate_data(replicate)
            enrichment_all.update(replicate_data.enrichment)
            fitness_all.update(replicate_data.fitness)
        self._enrichment = enrichment_all
        self._fitness = fitness_all
        # self.enrichment = self._get_enrichment(self.frequencies)
        # self.fitness = self._get_fitness(self.enrichment)

    @property
    def replicate_numbers(self) -> list:
        """
        If multiple selection experiments were submitted,
        match the treatment samples to the correct untreated sample.

        Returns
        -------
        group_numbers : list
            Numbers of the groups from the sequencing set as string-type values
        """

        replicate_numbers = []

        r = re.compile(r"\d+")
        for name in self.samples:
            numbers = r.findall(name)
            replicate_numbers += numbers
        replicate_numbers = list(set(replicate_numbers))
        replicate_numbers = natsorted(replicate_numbers)
        replicate_numbers = [int(x) for x in replicate_numbers]
        return replicate_numbers

    def get_replicate_data(self, n: int) -> SequencingData:
        """
        If multiple selection experiments were submitted, match the treatment
        samples to the correct untreated sample.

        Parameters
        ----------
        n : int
            Replicate number to retrieve data for

        Returns
        -------
        dataset : SequencingData
            Data for the specified group number
        """
        data = copy.deepcopy(self)
        replicate_samples = []
        for sample in self.samples:
            replicate_number = re.sub("[a-zA-Z]*?", "", sample)
            if replicate_number == str(n):
                replicate_samples.append(sample)
        data.samples = replicate_samples
        return data


# ! unused class
class SequencingDataSublibraries:  # pylint: disable=too-few-public-methods
    """
    Class used to combine results when library is sequenced as multiple sublibraries
    """

    #     def __init__(self, SequencingData):
    #         self.fullset = SequencingData
    #         self.samples = self.fullset.samples
    #         self.treatments = self.fullset.treatments
    #         self.pools = self._get_sublibrary_pools(self.samples)
    #         self.counts = self._combine_tables("counts")
    #         self.fitness = self._combine_tables("fitness")

    #     def _get_sublibrary_pools(self, sample_names: list) -> list:
    #         """
    #         Determine which sublibraries were pooled given a set of samples named
    #         by convention <TREATMENT>56, 78, etc.

    #         This is specific to our TEM-1 mutagenesis library and will not be able
    #         to be generalized to other datasets

    #         Parameters
    #         ----------
    #         sample_names : list
    #             List of sample names fromd dataset

    #         Returns
    #         -------
    #         list
    #             List of pooled groups with numbers as string-ypes
    #         """

    #         # * need to strip library numbers in descending order to account for double digit numbers
    #         sublibrary_numbers = [
    #             n.removeprefix("MAS") for n in self.fullset.gene.sublibrary_positions.keys()
    #         ][::-1]

    #         data_pools = []
    #         for sample in sample_names:
    #             name = sample
    #             # * track the numbers per name
    #             sample_pool = []
    #             for n in sublibrary_numbers:
    #                 # * collect sublibrary numbers from end of name
    #                 if name.endswith(n):
    #                     # * insert at index 0 to create ascending order
    #                     sample_pool.insert(0, n)
    #                     name = name.removesuffix(n)
    #             if sample_pool not in data_pools:
    #                 data_pools.append(sample_pool)
    #         return data_pools

    #     def _get_pool_residues(self, pools: list) -> dict:
    #         """
    #         Retrieve list of covered positions from a list of pooled sublibrary numbers

    #         Parameters
    #         ----------
    #         pools : list
    #             List of pooled groups with numbers as string-types

    #         Returns
    #         -------
    #         sublibrary_residues : dict
    #             Covered library positions matched to a concatenated string
    #             representation of pooled library numbers
    #         """

    #         pool_residues = {}
    #         for pool in pools:
    #             sublibrary_names = ["MAS" + s for s in pool]
    #             residues = []
    #             for name in sublibrary_names:
    #                 residues += self.fullset.gene.sublibrary_positions[name]
    #             residues.sort()
    #             # * reverse back to 0-index sorting instead of Ambler numbering to match
    #             # * other functions
    #             indices = np.searchsorted(
    #                 self.fullset.gene.ambler_numbering, residues
    #             ).tolist()
    #             pool_residues["".join(pool)] = indices
    #         return pool_residues

    #     def _combine_tables(self, data_name: str) -> dict:
    #         """
    #         Go through sequencing pools and combine positions into one dataframe
    #         (minus signal peptide)

    #         Parameters
    #         ----------
    #         data_name : str
    #             Data to combine

    #         Returns
    #         -------
    #         combined_dict : dict
    #             Single combined data
    #         """

    #         if data_name == "fitness":
    #             treated = [x for x in self.treatments if "UT" not in x]
    #             df_dict = {key: [] for key in treated}
    #         elif data_name == "counts":
    #             df_dict = {key: [] for key in self.treatments}

    #         pool_residues = self._get_pool_residues(self.pools)
    #         for pool_name, residue_list in pool_residues.items():
    #             # * make a copy to not overwrite the original dataset
    #             pool_dataset = copy.deepcopy(self.fullset)
    #             # * reduce samples in dataset
    #             pool_dataset.samples = [x for x in self.fullset.samples if pool_name in x]
    #             if data_name == "fitness":
    #                 data = pool_dataset.fitness
    #             elif data_name == "counts":
    #                 data = pool_dataset.counts
    #             for name, df in data.items():
    #                 if data_name == "counts":
    #                     name = re.sub(r"([^A-Za-z0-9])?\d+$", "", name)
    #                 sublibrary_data = df.loc[residue_list]
    #                 df_dict[name].append(sublibrary_data)
    #         # * iterate back through each treatment and merge the data
    #         combined_dict = {}
    #         for treatment, df_list in df_dict.items():
    #             merged_df = pd.concat(df_list)
    #             merged_df = merged_df.reindex(np.arange(merged_df.index[-1]), fill_value=0)
    #             combined_dict[treatment] = merged_df
    #         return combined_dict

    #     def _combine_fitness(self, dataset: SequencingData) -> dict:
    #         """
    #         Go through sequencing pools and appropriate calculate fitness data by
    #         restricting the positions for each set

    #         Parameters
    #         ----------
    #         dataset : SequencingData
    #             Processed sequencing data

    #         Returns
    #         -------
    #         combined_fitness : dict
    #             Combined fitness data for each treatment (minus signal peptide)
    #         """

    #         treated = [x for x in self.fullset.treatments if "UT" not in x]
    #         fitness_dict = {treatment: [] for treatment in treated}
    #         # go through each pool and restrict the dataset to propertly
    #         # calculate enrichment/fitness
    #         pool_residues = self._get_pool_residues(self.pools)
    #         for pool_name, residue_list in pool_residues.items():
    #             # make a copy to not overwrite the original dataset
    #             pool_dataset = copy.deepcopy(self.fullset)
    #             # reduce samples in dataset
    #             pool_dataset.samples = [x for x in self.fullset.samples if pool_name in x]
    #             for treatment, fitness_df in pool_dataset.fitness.items():
    #                 sublibrary_data = fitness_df.loc[residue_list]
    #                 fitness_dict[treatment].append(sublibrary_data)
    #         # iterate back through each treatment and merge the data
    #         combined_fitness = {
    #             treatment: pd.concat(df_list) for treatment, df_list in fitness_dict.items()
    #         }

    #         return combined_fitness

    #     @property
    #     def enrichment(self):
    #         raise AttributeError(
    #             "Enrichment data cannot be accurately represented in combined \
    #             dataframes, see 'fullset' attribute"
    #         )
    pass  # pylint: disable=unnecessary-pass


# TEM1 = TEM1_gene(
#     "/work/greencenter/s426833/TEM-1/ref_data/pBR322_AvrII.gbk", "blaTEM-1"
# )
# data1 = SequencingData(
#     TEM1,
#     "/endosome/work/greencenter/s426833/TEM-1/experiments/20221102_TEM-1-Mutagenesis_AV",
# )
# data2 = SequencingDataReplicates(
#     TEM1,
#     "/endosome/work/greencenter/s426833/TEM-1/experiments/20221102_TEM-1-Mutagenesis_AV",
# )
# # data1.samples = ["AMP1", "UT1"]
# # data2.samples = ["AMP1", "UT1"]
# # print(data1.fitness)
# print(data2.fitness)

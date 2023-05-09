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
from functools import reduce
import glob
import re
from pathlib import Path

import numpy as np
import pandas as pd
from natsort import natsorted

from plasmid_map import Gene
from utils.seq_data_utils import heatmap_masks

# TODO: fix @fitness to not be a property i guess


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
                extinct_add
            )
            df_freqs = adjusted_counts.divide(total_reads[name])
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
        """

        untreated = [x for x in frequencies if "UT" in x]

        enrichment = {}
        for sample in frequencies:
            if "UT" in sample:
                continue
            untreated = self.match_treated_untreated(sample)
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
        fitness_masked : pd.DataFrame
            Masked fitness dataframe
        """
        sample_name = fitness_df.name
        untreated = self.match_treated_untreated(sample_name)
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
        """

        fitness = {}
        for sample in sorted(enrichment):
            if "UT" in sample:
                continue
            df_enriched = enrichment[sample]
            SynWT_enrichment = df_enriched["∅"]
            # ? calculate mean from fitting normalized curve or by nanmean
            SynWT_mean = np.nanmean(SynWT_enrichment)
            normalized = df_enriched.subtract(SynWT_mean)
            normalized.name = sample
            fitness[sample] = normalized

            # ! mask out wild-type positions
            fitness_masked = normalized.mask(heatmap_masks(self.gene))
            fitness_masked.name = sample
            # * here mask where untreated counts are insufficient but treated counts
            # * are large (i.e. highly beneficial mutations)
            # fitness_masked = self._mask_untreated_zero(
            #     counts, fitness_masked, read_threshold=read_threshold
            # )
            # fitness_masked.name = sample
            fitness[sample] = fitness_masked
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
            self._counts, self._total_reads, self.extinct_add
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
            print("Either more than one untreated sample found in dataset, cannot calculate enrichment. Reset samples to recalculate.")
        return self._enrichment

    @property
    def fitness(self):
        r"""Fitness DataFrame for each sample calculated for mutation (i) by
        :math:`e^i - < e^{WT} >`
        """
        if self._fitness is None:
            print("Either more than one untreated sample found in dataset, cannot calculate enrichment. Reset samples to recalculate.")
        return self._fitness

    def get_pairs(self, treatment: str, samples: list) -> tuple[str, str]:
        # TODO: implement ability to find more than 2 matches
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
        replica_one : str
            Name of first replicate
        replica_two : str
            Name of second replicate
        """
        treatment_pair = [sample for sample in samples if treatment in sample]
        if not treatment_pair:
            raise KeyError(f"No fitness data: {treatment}")
        if len(treatment_pair) > 2:
            raise IndexError("Treatment has more than 2 replicates to compare")
        replica_one, replica_two = treatment_pair[0], treatment_pair[1]
        return replica_one, replica_two


    def match_treated_untreated(self, sample: str) -> str:
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
        r = re.compile(r"_(\d+)")
        num = r.findall(sample)[0]
        untreated = f"UT_{num}"
        return untreated

    def filter_fitness_read_noise(
        self,
        read_threshold: int = 20,
    ) -> dict:
        """
        Takes DataFrames for treated sample and returns a new DataFrame with cells
        with untreated counts under the minimum read threshold filtered out

        Parameters
        ----------
        read_threshold : int, optional
            Minimum number of reads required to be included, by default 20

        Returns
        -------
        df_treated_filtered : dict
            Fitness tables with insufficient counts filtered out
        """
        counts_dict = self.counts
        fitness_dict = self.fitness

        dfs_filtered = {}
        for sample in sorted(fitness_dict):
            untreated = self.match_treated_untreated(sample)
            df_counts_untreated = counts_dict[untreated]
            df_counts_sample = counts_dict[sample]
            df_fitness_sample = fitness_dict[sample]
            dfs_filtered[sample] = df_fitness_sample.where(
                df_counts_sample.ge(read_threshold) | df_counts_untreated.ge(read_threshold)
            )
        return dfs_filtered


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

        r = re.compile(r"_(\d+)")
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
        r = re.compile(r"_(\d+)")
        for sample in self.samples:
            replicate_number = r.findall(sample)[0]
            if replicate_number == str(n):
                replicate_samples.append(sample)
        data.samples = replicate_samples
        return data

    @SequencingData.samples.setter
    def samples(self, value):
        self._samples = value
        self._treatments = self._get_treatments(value)
        self._counts = self._get_counts(self._samples, self._inputfolder)
        self._total_reads = self._get_total_reads(self._inputfolder)
        self._frequencies = self._get_frequencies(
            self._counts, self._total_reads, self.extinct_add
        )
        self._enrichment = self._get_enrichment(self._frequencies)
        self._fitness = self._get_fitness(self._enrichment)

class SequencingDataPools(SequencingData):
    r"""
    Class for pooling counts data and calculating fitness values thereafter

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

    @SequencingData.samples.setter
    def samples(self, value):
        self._samples = value
        self._treatments = value
        self._counts = self._get_pooled_counts()
        self._total_reads = self._get_pooled_total_reads()
        self._frequencies = self._get_frequencies(
            self._counts, self._total_reads, self.extinct_add
        )
        if len([sample for sample in self._samples if "UT" in sample]) == 1:
            self._enrichment = self._get_enrichment(self._frequencies)
            self._fitness = self._get_fitness(self._enrichment)

    def match_treated_untreated(self, sample: str) -> str:
        return "UT"

    def _get_pooled_counts(self) -> dict:
        """
        Override function pools counts for processing
        """
        counts_data = {}
        pooled_counts = {}
        for treatment in self.treatments:
            treatment_counts = [value for key, value in self.counts.items() if treatment in key]
            counts_data[treatment] = treatment_counts
        for treatment in self.treatments:
            pooled_counts_df = reduce(lambda a, b: a.add(b, fill_value=0), counts_data[treatment])
            pooled_counts[treatment] = pooled_counts_df
        return dict(natsorted(pooled_counts.items()))

    def _get_pooled_total_reads(self) -> dict:
        pooled_total_reads = {}
        for treatment in self.treatments:
            treatment_counts = [value for key, value in self.total_reads.items() if treatment in key]
            pooled_total_reads[treatment] = sum(treatment_counts)
        return dict(natsorted(pooled_total_reads.items()))

    def filter_fitness_read_noise(
        self,
        read_threshold: int = 20,
    ) -> dict:
        """
        Takes DataFrames for treated sample and returns a new DataFrame with cells
        with untreated counts under the minimum read threshold filtered out

        Parameters
        ----------
        read_threshold : int, optional
            Minimum number of reads required to be included, by default 20

        Returns
        -------
        df_treated_filtered : dict
            Fitness tables with insufficient counts filtered out
        """
        counts_dict = self.counts
        fitness_dict = self.fitness

        dfs_filtered = {}
        for sample in sorted(fitness_dict):
            untreated = self.match_treated_untreated(sample)
            df_counts_untreated = counts_dict[untreated]
            df_counts_sample = counts_dict[sample]
            df_fitness_sample = fitness_dict[sample]
            dfs_filtered[sample] = df_fitness_sample.where(
                df_counts_sample.ge(read_threshold) | df_counts_untreated.ge(read_threshold)
            )
        return dfs_filtered

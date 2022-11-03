#!/usr/bin/env python
"""
This script is written to parse the mutation data from the read sequences. It provides a base SequencingData object class that holds information about the dataset (e.g. sample names, reference gene) and calculates counts, frequency, enrichment, and fitness data. Two additional subclasses are provided for parsing different sequencing schemes, such as when the sublibraries are submitted as one pooled library sample as opposed to separate sublibraries for sequencing.
"""
import copy
import csv
import glob
import re
from pathlib import Path

import numpy as np
import pandas as pd
from natsort import natsorted
from scipy.stats import norm

import plasmid_map


class SequencingData:
    """
    A class created to compile fitness data about each sequencing dataset.
    """

    def __init__(self, gene: plasmid_map.Gene, inputfolder: str):
        self.gene = gene
        self._inputfolder = inputfolder
        self.samples = self._get_sample_names(self._inputfolder)

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

        with open(samples_file, "r") as file:
            for line in file:
                line = line.rstrip()
                name = line.split("\t")[1]
                samples.append(name)
        return natsorted(samples)

    def copy(self):
        """
        Copy method

        Returns
        -------
        self : SequencingData
        """
        return copy.deepcopy(self)

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
            treatment = re.sub("([^A-Za-z0-9])?\d+$", "", name)
            treatments.append(treatment)
        treatments = natsorted(list(set(treatments)))
        return treatments

    def _get_counts(self, sample_list: list, inputfolder: str) -> dict:
        """
        Load all matrices from the results folder as pandas.DataFrame objects and compile a dictionary

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
        Retrieve total number of reads found for each sample after alignment (by `samtools idxstats`) and create dictionary

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

        with open(reads_file, "r") as csvfile:
            reader = csv.reader(csvfile, delimiter="\t")
            total_reads = {row[0]: int(row[1]) for row in reader}
        return total_reads

    def _get_frequencies(self, counts: dict, total_reads: dict) -> dict:
        """
        Calculate frequecy (f) for mutation (i) by N^i / N^{total reads}.

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
            df_freqs = counts[name].divide(total_reads[name])
            if "UT" in name:
                df_freqs = df_freqs[counts[name] != 0]
            df_freqs.name = name
            frequencies[name] = df_freqs
        return frequencies

    def _get_enrichment(self, frequencies: dict) -> dict:
        """
        Calculate enrichment (e) of each mutation (i) by equation log_{10}(\frac{f^i{selected}}{f^i_{unselected}})

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
            Will not be able to calculate enrichment properly if there is more than one untreated sample to compare frequencies to
        """
        samples = frequencies.keys()
        treatments = self._get_treatments(samples)
        untreated = [x for x in samples if "UT" in x]
        num_untreated = len(untreated)
        if num_untreated > 1:
            raise LookupError("More than one untreated sample found in dataset")

        treated = [x for x in treatments if "UT" not in x]
        enrichment = dict.fromkeys(treated)
        df_untreated = [frequencies[x] for x in samples if "UT" in x][0]
        for treatment in treatments:
            if "UT" in treatment:
                continue
            df_treated = [frequencies[x] for x in samples if treatment in x][0]
            df_enriched = df_treated.divide(df_untreated)
            with np.errstate(divide="ignore"):
                df_enriched = df_enriched.where(df_enriched == 0, np.log10)
            df_enriched.name = treatment
            enrichment[treatment] = df_enriched
        return enrichment

    def _get_fitness(self, enrichment: dict) -> dict:
        """
        Calculate normalized fitness values (s) of each mutation (i) by subtracting the enrichment of synonymous wild-type mutations from the enrichment value of a mutation
        s^i = e^i - < e^{WT} >

        Parameters
        ----------
        enrichment : dict
            Dataframes of enrichment values for samples

        Returns
        -------
        fitness : dict
            Dictionary with sample names as keys and fitness dataframes as values

        Raises
        ------
        LookupError
            Will not be able to calculate enrichment properly (and thus fitness) if there is more than one untreated sample to compare frequencies to
        """
        samples = enrichment.keys()
        treatments = self._get_treatments(samples)
        untreated = [x for x in samples if "UT" in x]
        num_untreated = len(untreated)
        if num_untreated > 1:
            raise LookupError("More than one untreated sample found in dataset")

        treatments = [x for x in treatments if "UT" not in x]
        fitness = dict.fromkeys(treatments)

        for treatment in treatments:
            if "UT" in treatment:
                continue
            df_enriched = enrichment[treatment]
            SynWT_enrichment = df_enriched["∅"]
            SynWT_mean, _ = norm.fit(SynWT_enrichment.dropna())
            normalized = df_enriched - SynWT_mean
            normalized.name = treatment
            fitness[treatment] = normalized
        return fitness

    @property
    def samples(self) -> list:
        return self._samples

    @samples.setter
    def samples(self, value):
        self._samples = value

    @property
    def treatments(self):
        return self._get_treatments(self._samples)

    @property
    def counts(self):
        return self._get_counts(self.samples, self._inputfolder)

    @property
    def total_reads(self):
        return self._get_total_reads(self._inputfolder)

    @property
    def frequencies(self):
        return self._get_frequencies(self.counts, self.total_reads)

    @property
    def enrichment(self):
        return self._get_enrichment(self.frequencies)

    @property
    def fitness(self):
        return self._get_fitness(self.enrichment)


class SequencingDataReplicates(SequencingData):
    """
    Class used to divide sequencing results into replicate datasets (n)
    """

    def __init__(self, gene, inputfolder):
        super().__init__(gene, inputfolder)

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

        r = re.compile("\d+")
        for name in self.samples:
            numbers = r.findall(name)
            replicate_numbers += numbers
        replicate_numbers = list(set(replicate_numbers))
        replicate_numbers = natsorted(replicate_numbers)
        replicate_numbers = [int(x) for x in replicate_numbers]
        return replicate_numbers

    def get_replicate_data(self, n: int) -> SequencingData:
        """
        If multiple selection experiments were submitted, match the treatment samples to the correct untreated sample.

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


class SequencingDataSublibraries:
    """
    Class used to combine results when library is sequenced as multiple sublibraries
    """

    def __init__(self, SequencingData):
        self.fullset = SequencingData
        self.samples = self.fullset.samples
        self.treatments = self.fullset.treatments
        self.pools = self._get_sublibrary_pools(self.samples)
        self.counts = self._combine_tables("counts")
        self.fitness = self._combine_tables("fitness")

    def _get_sublibrary_pools(self, sample_names: list) -> list:
        """
        Determine which sublibraries were pooled given a set of samples named by convention <TREATMENT>56, 78, etc.

        This is specific to our TEM-1 mutagenesis library and will not be able to be generalized to other datasets

        Parameters
        ----------
        sample_names : list
            List of sample names fromd dataset

        Returns
        -------
        list
            List of pooled groups with numbers as string-ypes
        """
        # * need to strip library numbers in descending order to account for double digit numbers
        sublibrary_numbers = [
            n.removeprefix("MAS") for n in self.fullset.gene.sublibrary_positions.keys()
        ][::-1]

        data_pools = []
        for sample in sample_names:
            name = sample
            # * track the numbers per name
            sample_pool = []
            for n in sublibrary_numbers:
                # * collect sublibrary numbers from end of name
                if name.endswith(n):
                    # * insert at index 0 to create ascending order
                    sample_pool.insert(0, n)
                    name = name.removesuffix(n)
            if sample_pool not in data_pools:
                data_pools.append(sample_pool)
        return data_pools

    def _get_pool_residues(self, pools: list) -> dict:
        """
        Retrieve list of covered positions from a list of pooled sublibrary numbers

        Parameters
        ----------
        pools : list
            List of pooled groups with numbers as string-types

        Returns
        -------
        sublibrary_residues : dict
            Covered library positions matched to a concatenated string representation of pooled library numbers
        """
        pool_residues = {}
        for pool in pools:
            sublibrary_names = ["MAS" + s for s in pool]
            residues = []
            for name in sublibrary_names:
                residues += self.fullset.gene.sublibrary_positions[name]
            residues.sort()
            # * reverse back to 0-index sorting instead of Ambler numbering to match other functions
            indices = np.searchsorted(
                self.fullset.gene.ambler_numbering, residues
            ).tolist()
            pool_residues["".join(pool)] = indices
        return pool_residues

    def _combine_tables(self, data_name: str) -> dict:
        """
        Go through sequencing pools and combine positions into one dataframe (minus signal peptide)

        Parameters
        ----------
        data_name : str
            Data to combine

        Returns
        -------
        combined_dict : dict
            Single combined data
        """
        if data_name == "fitness":
            treated = [x for x in self.treatments if "UT" not in x]
            df_dict = {key: [] for key in treated}
        elif data_name == "counts":
            df_dict = {key: [] for key in self.treatments}

        pool_residues = self._get_pool_residues(self.pools)
        for pool_name, residue_list in pool_residues.items():
            # * make a copy to not overwrite the original dataset
            pool_dataset = copy.deepcopy(self.fullset)
            # * reduce samples in dataset
            pool_dataset.samples = [x for x in self.fullset.samples if pool_name in x]
            if data_name == "fitness":
                data = pool_dataset.fitness
            elif data_name == "counts":
                data = pool_dataset.counts
            for name, df in data.items():
                if data_name == "counts":
                    name = re.sub("([^A-Za-z0-9])?\d+$", "", name)
                sublibrary_data = df.loc[residue_list]
                df_dict[name].append(sublibrary_data)
        # * iterate back through each treatment and merge the data
        combined_dict = {}
        for treatment, df_list in df_dict.items():
            merged_df = pd.concat(df_list)
            merged_df = merged_df.reindex(np.arange(merged_df.index[-1]), fill_value=0)
            combined_dict[treatment] = merged_df
        return combined_dict

    def _combine_fitness(self, dataset: SequencingData) -> dict:
        """
        Go through sequencing pools and appropriate calculate fitness data by restricting the positions for each set

        Parameters
        ----------
        dataset : SequencingData
            Processed sequencing data

        Returns
        -------
        combined_fitness : dict
            Combined fitness data for each treatment (minus signal peptide)
        """
        treated = [x for x in self.fullset.treatments if "UT" not in x]
        fitness_dict = {treatment: [] for treatment in treated}
        # * go through each pool and restrict the dataset to propertly calculate enrichment/fitness
        pool_residues = self._get_pool_residues(self.pools)
        for pool_name, residue_list in pool_residues.items():
            # * make a copy to not overwrite the original dataset
            pool_dataset = copy.deepcopy(self.fullset)
            # * reduce samples in dataset
            pool_dataset.samples = [x for x in self.fullset.samples if pool_name in x]
            for treatment, fitness_df in pool_dataset.fitness.items():
                sublibrary_data = fitness_df.loc[residue_list]
                fitness_dict[treatment].append(sublibrary_data)
        # * iterate back through each treatment and merge the data
        combined_fitness = {
            treatment: pd.concat(df_list) for treatment, df_list in fitness_dict.items()
        }

        return combined_fitness

    @property
    def enrichment(self):
        raise AttributeError("Enrichment data cannot be accurately represented in combined dataframes, see 'fullset' attribute")
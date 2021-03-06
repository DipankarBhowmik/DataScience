# Glossary of protein world through deep learning.
## Problem description

This directory contains data to train a model to predict the function of protein domains, based on the PFam dataset. Domains are functional sub-parts of proteins; much like images in ImageNet are pre segmented to contain exactly one object class, this data is presegmented to contain exactly and only one domain. The purpose of the dataset is to repose the PFam seed dataset as a multiclass classification machine learning task. The task is: given the amino acid sequence of the protein domain, predict which class it belongs to. There are about 1 million training examples, and 18,000 output classes.

## Data structure

This data is more completely described by the publication "Can Deep Learning Classify the Protein Universe", Bileschi et al.

## Data split and layout

The approach used to partition the data into training/dev/testing folds is a random split.

- Training data should be used to train your models.
- Dev (development) data should be used in a close validation loop (maybe for hyperparameter tuning or model validation).
- Test data should be reserved for much less frequent evaluations - this helps avoid overfitting on your test data, as it should only be used infrequently.

## File content

Each fold (train, dev, test) has a number of files in it. Each of those files contains csv on each line, which has the following fields:

      sequence: HWLQMRDSMNTYNNMVNRCFATCIRSFQEKKVNAEEMDCTKRCVTKFVGYSQRVALRFAE 
      family_accession: PF02953.15
      sequence_name: C5K6N5_PERM5/28-87
      aligned_sequence: ....HWLQMRDSMNTYNNMVNRCFATCI...........RS.F....QEKKVNAEE.....MDCT....KRCVTKFVGYSQRVALRFAE 
      family_id: zf-Tim10_DDP

## Description of fields:

- sequence: These are usually the input features to your model. Amino acid sequence for this domain. There are 20 very common amino acids (frequency > 1,000,000), and 4 amino acids that are quite uncommon: X, U, B, O, Z.
- family_accession: These are usually the labels for your model. Accession number in form PFxxxxx.y (Pfam), where xxxxx is the family accession, and y is the version number. Some values of y are greater than ten, and so 'y' has two digits.
- family_id: One word name for family.
- sequence_name: Sequence name, in the form "uniprot_accession_id/start_index-and end_index".
- aligned_sequence: Contains a single sequence from the multiple sequence alignment (with the rest of the members of the family in seed, with gaps retained.

Generally, the family_accession field is the label, and the sequence (or aligned sequence) is the training feature. This sequence corresponds to a domain, not a full protein. The contents of these fields is the same as to the data provided in Stockholm format by PFam at ftp://ftp.ebi.ac.uk/pub/databases/Pfam/releases/Pfam32.0/Pfam-A.seed.gz

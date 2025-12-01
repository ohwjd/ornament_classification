# Search and classification of ornaments in 16th century lute tablature

In this bachelor thesis, a method for the search and classification of ornaments in lute tablature is presented. The aim is to develop an algorithm that can identify and categorize different types of ornaments in lute tablature based on a set of predefined rules taken from instructions by Vincenzo Galilei (1584) as summarized by Minamino (1988).
The work summarizes different approaches for search in polyphonic music and discusses the challenges of applying these approaches to lute tablature. The developed algorithm is a rule-based approach that analyzes the output of the tabmapper tool from the abtab toolbox (De Valk, 2024), which identifies notes belonging to the vocal model and those that are part of an ornamentation. The algorithm applies Galilei's rules to classify the identified ornaments.
This work is part of the process of developing a computer-aided analysis tool for the E-LAUTE project, which aims to study lute intabulations of vocal music. The findings of this thesis contribute to a better understanding of lute ornamentation and lute music in general.

------------------------------------------------------------------
After cloning the repository, cd into the folder. 
To run this code it is recommended to use a virtual environment to then installing the required libraries:

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

To then run the algorithm use the following command:

```bash
python -m search_for_rules.main

```

The output is then saved in the "output" folder. 
It contains subfolders corresponding to the individual pieces from the data set. 
Each ....

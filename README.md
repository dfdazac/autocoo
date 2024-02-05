# Comparing Ontology Engineering Approaches for Clinical Trial Outcomes

This repository contains the implementation to run and evaluate different approaches for building an ontology of outcome measures from a given clinical trial.

We are interested in clinical trials relevant to two diseases: **diabetes**, and **gastrointestinal (GI) cancer**.

In this project we explore three approaches towards building an ontology of outcome measures for clinical trials related to these diseases:

- Manual ontology engineering
- Semi-automatic construction
- Automatic construction

The ontologies were created based on clinical trials obtained from [clinicaltrials.gov](https://clinicaltrials.gov/). The data was exported in XML format.
It can be downloaded from [this link](https://drive.google.com/file/d/1zoMuw8QrLUPR-hxiEyWSqsBTeMLtkGNE/view?usp=drive_link).



### Manual construction

They can be found (to be confirmed) here:

- [Diabetes ontology](https://drive.google.com/file/d/1bAXBCzwZA8iQUpfNt7eZibUy6SQsgue-/view?usp=drive_link)
- [GI cancer](https://drive.google.com/file/d/1kpBUfrg3V9bzJ2gHbxo65srl1ZSXI-aM/view?usp=drive_link)



### Semi-automatic construction

The steps for reproducing this pipeline are:

**1.** Convert XML file into CSV

**Reproducible?** :x:

**Comments:** It's not clear how the CSV file was obtained. The thesis mentions a "data parser", but this is not in the code provided.



**2.** Apply the NER model to the CSV data.

**Reproducible?** :x:

**Comments**: This step requires running the notebook at `semi-automatic/biomedical_transformer.ipynb`, which runs an NER model to extract entities of type "diagnostic procedure". The notebook requires a PDF file (not mentioned in the thesis) called `ctg_pdf.pdf`. It's not clear how to obtain this file.



**3.** Build the ontology manually with the detected terms.

**Reproducible:** :white_check_mark: At least in principle.

**Comments:** Once step 2 is completed, the diagnostic procedure entities are manually examined by a person. They are classified into one of 1) biomarkers, 2) diabetic diseases, 3) endpoint scores, 4) outcome measurement tools, and 5) questionnaires. Definitions are added using external databases, and synonyms are added as well.




### Automatic construction

The steps for reproducing this pipeline are:

**1.** Preprocess the clinical trials.
**Reproducible?** :x:

**Comments:** This step removes everything after the first period or comma in descriptions of outcome measures, keeping nouns only, and removing measures endind with "and" and "is". It's not clear how the initial XML files were processed until getting only outcome measure fields, and the code for achieving this is not provided.



**2.** Extract concepts with GPT-3.5

**Reproducible?** :x:

GPT is used to do some curation of the terms from the previous step. The code for executing this step is not available. However, it seems from the thesis that this step could be skipped, because "the language model considered the form of the outcome measures at this stage to be already satisfactory" (Michael Becker's thesis, section 4.2).



**3.** Clustering outcome measures

**Reproducible?** 


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



**3.** Build the ontology with the detected terms.

**Reproducible:** Yes, in principle.

**Comments:** Once step 2 is completed, the diagnostic procedure entities are manually examined. They are classified into one of 1) biomarkers, 2) diabetic diseases, 3) endpoint scores, 4) outcome measurement tools, and 5) questionnaires. Definitions are added using external databases, and synonyms are added as well.






### Automatic construction

---
title: "CMP Random Error Estimates"
type: pages
---

This page contains replication materials and the error dataset providing estimates of random error in the Comparative Manifesto Project’s dataset of coded quasi-sentence from election programmes. These are described in:

*  2012\. Mikhaylov, Slava, Michael Laver, and Kenneth Benoit. “[Coder Reliability and Misclassification in the Human Coding of Party Manifestos](/pdfs/MLB_PA_2012.pdf).” **_Political Analysis  _**20(1): 78–91**.


## Replication Materials

The simplest method is to download both the error dataset and the replication code (in R), including the datasets from Adams et al (2006) and from Hix et al (2006), from the **[Dataverse located here](http://dvn.iq.harvard.edu/dvn/dv/Mikhaylov/faces/study/StudyPage.xhtml?globalId=hdl:1902.1/17073&studyListingIndex=0_5770dea2e7a7c2ac7d8938933f90)**.

## Logit-Scaled Dataset with BLM (2009) Errors

The simplest method to obtain all of the BLM (2009) errors plus the scales from Lowe et al (2011) is from the **[Dataverse located here](http://dvn.iq.harvard.edu/dvn/dv/Mikhaylov/faces/study/StudyPage.xhtml?globalId=hdl:1902.1/17073&studyListingIndex=0_5770dea2e7a7c2ac7d8938933f90)**.

All the analysis in the article can be reproduced by running the R scripts contained in that replication set. Part of our analysis is the replication of two _AJPS_ publications, listed below. In order to replicate the analysis in BLM (2009) you will need need the corrected datasets from both of these articles, available below but also found in the replication zip file. Both datasets are also available on this website with the kind permission of the authors.

1.  Adams, James, M. Clark, L. Ezrow & G. Glasgow. 2006\. “Are Niche Parties Fundamentally Different from Mainstream Parties? The Causes and the Electoral Consequences of Western European Parties’ Policy Shifts, 1976-1998.” _American Journal of Political Science_ 50(3):513â€“529\. [Corrected dataset here](/assets/cmp/Adams_etal_replication.dta).
2.  Hix, Simon, Abdul Noury & Gerard Roland. 2006\. “Dimensions of Politics in the European Parliament.” _American Journal of Political Science_ 50(2, April):494â€“511\. [Corrected dataset here](/assets/cmp/Hix_etal_replication.dta).

The key outcome of our analysis is to produce standard error estimates for the CMP data. Our results are contained in Stata format in the file BLM_CMP_uncertainty.dta contained in the replication zip file).

## Merging the error estimates and logit scales with your own data

These standard error estimates can and as we showed should be used in any research that utilizes the CMP data. Our standard error dataset contains two variables that uniquely identify the results: **cmp_party** (a numeric code identifying the country and party, using the CMP’s numeric party coding scheme) and **cmp_edate** (a date-formatted variable indicating the election). These two unique identifiers allow merging standard error estimates with any CMP based dataset. For example, in Stata this is done simply by typing the following commands:

```
use BLM_CMP_uncertainty_extended.dta, clear
merge party edate using YOUR_CMP_DATASET.dta, sort
```

A similar merge can be performed in R using the **merge()** function:

```{r}
require(foreign)
CMP.original <- read.dta("mds2005f.dta", convert.factors=FALSE)
CMP.BLMerror <- read.dta("BLM_CMP_uncertainty_extended.dta")
# convert CMP's edate to numeric so can be matched to BLM file
# and harmonize the names of the election date variable
require(date)
CMP.BLMerror$edate <- as.date(CMP.BLMerror$cmp_edate)
# harmonize name of the party variable
names(CMP.original)[which(names(CMP.original)=="party")] <- "cmp_party"
# merge the files
CMP.merged <- merge(CMP.original, CMP.BLMerror, c("cmp_party","edate"))
```

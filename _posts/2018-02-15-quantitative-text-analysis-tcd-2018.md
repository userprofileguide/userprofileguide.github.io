---
id: 1460
title: Quantitative Text Analysis (TCD 2018)
date: 2018-02-15T15:30:30+00:00
author: Ken
layout: single

permalink: /quantitative-text-analysis-tcd-2018/
categories:
- Course-related
- Quantitative Methods
- Text Analysis
---

**When and where:** 16 February, 23 February, 9 March, 16 March, 23 March 2018; Arts Building 3020.

**Instructor:** Professor Kenneth Benoit, kbenoit@tcd.ie.  Assignment marking by [Stefan Muüller](MULLERS@tcd.ie).

* This will become a table of contents (this text will be scraped).
{:toc}


## Overview

The course surveys methods for systematically extracting quantitative information from political text for social scientific purposes, starting with classical content analysis and dictionary-based methods, to classification methods, and state-of-the-art scaling methods and topic models for estimating quantities from text using statistical techniques. The course lays a theoretical foundation for text analysis but mainly takes a very practical and applied approach, so that students learn how to apply these methods in actual research. The common focus across all methods is that they can be reduced to a three-step process: first, identifying texts and units of texts for analysis; second, extracting from the texts quantitatively measured features - such as coded content categories, word counts, word types, dictionary counts, or parts of speech - and converting these into a quantitative matrix; and third, using quantitative or statistical methods to analyse this matrix in order to generate inferences about the texts or their authors. The course systematically covers these methods in a logical progression, with a practical, hands-on approach where each technique will be applied using appropriate software to real texts.

## Objectives

The course is also designed to cover many fundamental issues in quantitative text analysis such as inter-coder agreement, reliability, validation, accuracy, and precision. It focuses on methods of converting texts into quantitative matrixes of features, and then analysing those features using statistical methods. The course briefly covers the qualitative technique of human coding and an- notation but only for the purposes of creating a validation set for automated approaches. These automated approaches include dictionary construction and application, classification and machine learning, scaling models, and topic models. For each topic, we will systematically cover published applications and examples of these methods, from a variety of disciplinary and applied fields but focusing on political science. Lessons will consist of a mixture of theoretical grounding in content analysis approaches and techniques, with hands on analysis of real texts using content analytic and statistical software.

## Detailed Course Schedule

### [Week 1: Introduction and issues in quantitative text analysis](/assets/courses/tcd2018qta/QTA_TCD_Day1.pdf)

This session will cover fundamentals, including the continuum from traditional
(non-computer assisted) content analysis to fully automated quantitative text
analysis. We will cover the conceptual foundations of content analysis and
quantitative content analysis, discuss the objectives, the approach to
knowledge, and the particular view of texts when performing quantitative
analysis. We will also discuss issues including where to obtain textual data;
formatting and working with text files; indexing and meta-data; units of
analysis; and definitions of features and measures commonly extracted
from texts, including stemming, and stop-words.

#### Required Reading
{:.no_toc}

- Vignette and instructions at https://docs.quanteda.io  
- Tutorials at https://tutorials.quanteda.io/  
- Grimmer and Stewart (2013)  
- Manning, Raghavan and Schütze (2008, 117–120)  
- Krippendorff (2013, Chs. 9–10)  
- Dunning (1993)  

#### Recommended Reading
{:.no_toc}

- Krippendorff (2013, Ch. 1–2, 5, 7)  
- Wikipedia entry on Character encoding, http://en.wikipedia.org/wiki/Text_encoding  
- Browse the different text file formats at http://www.fileinfo.com/filetypes/text  
- Neuendorf (2002, Chs. 4–7)  
- Krippendorff (2013, Ch. 6)  
- Däubler et al. (2012)

#### Resources
{:.no_toc}

* [Week 1 slides](/assets/courses/tcd2018qta/QTA_TCD_Day1.pdf)  
* [Getting Started with quanteda and R](/assets/courses/tcd2018qta/getting_started.html) and the [`.Rmd` source](/assets/courses/tcd2018qta/getting_started.Rmd), which you should save and try to knit
* [Assignment 1: Working with Texts in quanteda](/assets/courses/tcd2018qta/assignment1_LASTNAME_FIRSTNAME.html) and [`.Rmd` source](https://cdn.rawgit.com/kbenoit/kbenoit.github.io/4bf16046/assets/courses/tcd2018qta/assignment1_LASTNAME_FIRSTNAME.Rmd)


### [Week 2: Quantitative methods for comparing texts](/assets/courses/tcd2018qta/QTA_TCD_Day2.pdf)

Here we focus on quantitative methods for describing texts, focusing on summary measures that highlight particular characteristics of documents and allowing these to be compared. These meth- ods include characterizing texts through concordances, co-occurrences, and keywords in context; complexity and readability measures; and an in-depth discussion of text types, tokens, and equiv- alencies. We will also discuss weighting strategies for features, such as tf-idf. The emphasis will be on comparing texts, through concordances and keyword identification, dissimilarity measures, association models, and vector-space models.

#### Required Reading
{:.no_toc}

- Krippendorff (2013, Ch. 10)  
- Lowe et al. (2011)  
- Manning, Raghavan and Schütze (2008, Section 6.3)

#### Recommended Reading
{:.no_toc}

- Seale, Ziebland and Charteris-Black (2006)

#### Resources
{:.no_toc}

- [Week 2 slides](/assets/courses/tcd2018qta/QTA_TCD_Day2.pdf)
- [Exercise 2: Describing and comparing texts and their features](/assets/courses/tcd2018qta/assignment2_LASTNAME_FIRSTNAME.html) and [`.Rmd` source](https://cdn.rawgit.com/kbenoit/kbenoit.github.io/73accef5/assets/courses/tcd2018qta/assignment2_LASTNAME_FIRSTNAME.Rmd)


### [Week 3: Automated dictionary-based approaches](/assets/courses/tcd2018qta/QTA_TCD_Day3.pdf)

Automatic dictionary-based methods involve association of pre-defined word lists with particular quantitative values assigned by the researcher for some characteristic of interest. This topic covers the design model behind dictionary construction, including guidelines for testing and refining dic- tionaries. Hand-on work will cover commonly used dictionaries such as LIWC, RID, and the Harvard IV-4, with applications. We will also review a variety of text pre-processing issues and textual data concepts such as word types, tokens, and equivalencies, including word stemming and trimming of words based on term and/or document frequency.

#### Required Reading
{:.no_toc}

- Neuendorf (2002, Ch. 6)  
- Laver and Garry (2000)  
- Rooduijn and Pauwels (2011)

#### Recommended Reading
{:.no_toc}

- Pennebaker and Chung (2008)  
- Tausczik and Pennebaker (2010)  
- Loughran and McDonald (2011)  

#### Resources
{:.no_toc}

- [Week 3 slides](/assets/courses/tcd2018qta/QTA_TCD_Day3.pdf)  
- [Exercise 3: Applying, modifying, and creating dictionaries for the analysis of political texts.](/assets/courses/tcd2018qta/assignment3_LASTNAME_FIRSTNAME.html) and [`.Rmd` source](https://cdn.rawgit.com/kbenoit/kbenoit.github.io/3ef69b09/assets/courses/tcd2018qta/assignment3_LASTNAME_FIRSTNAME.Rmd)


### [Week 4: Machine learning and scaling for texts](/assets/courses/tcd2018qta/QTA_TCD_Day4.pdf)

Classification methods permit the automatic classification of texts in a test set following machine learning from a training set. We will introduce machine learning methods for classifying documents, including one of the most popular classifiers, the Naive Bayes model. The topic also introduces validation and reporting methods for classifiers and discusses where these methods are applicable. Building on the Naive Bayes classifier, we introduce the “Wordscores” method of Laver, Benoit and Garry (2003) for scaling latent traits, and show the link between classification and scaling. We also cover applications of penalized regression to score and scale texts.

#### Resources
{:.no_toc}

- [Week 4 slides](/assets/courses/tcd2018qta/QTA_TCD_Day4.pdf)  
- [Exercise 4: Machine learning and scaling for texts.](/assets/courses/tcd2018qta/assignment4_LASTNAME_FIRSTNAME.html) and [`.Rmd` source](https://cdn.rawgit.com/kbenoit/kbenoit.github.io/5fb4ba46/assets/courses/tcd2018qta/assignment4_LASTNAME_FIRSTNAME.Rmd)

#### Required Reading
{:.no_toc}

- Manning, Raghavan and Schütze (2008, Ch. 13)  
- Lantz (2013, Ch. 3–4)  

#### Recommended Reading
{:.no_toc}

- Lantz (2013, Ch. 10)  
- Statsoft, "Naive Bayes Classifier Introductory Overview," http://www.statsoft.com/textbook/naive-bayes-classifier/.  
- An online article by Paul Graham on classifying spam e-mail. http://www.paulgraham. com/spam.html.  
- Bionicspirit.com, 9 Feb 2012, "How to Build a Naive Bayes Classifier," http://bionicspirit. com/blog/2012/02/09/howto-build-naive-bayes-classifier.html.  
- Yu, Kaufmann and Diermeier (2008)  
- Zumel and Mount (2014, Ch. 5–6)  
- Benoit and Nulty (2013)  
- Martin and Vanberg (2007)  
- Benoit and Laver (2008)  
- Lowe (2008)


### [Week 5: Scaling models](/assets/courses/tcd2018qta/QTA_TCD_Day5.pdf)

Building on the Naive Bayes classifier, we introduce the “Wordscores” method of Laver, Benoit and Garry (2003) for scaling latent traits, and show the link between classification and scaling.  We also cover the unsupervised "Wordfish" scaling method of Slapin and Proksch (2008).

#### Required Reading
{:.no_toc}

- Evans et al. (2007)  
- Laver, Benoit and Garry (2003)  
- Slapin and Proksch (2008)

#### Recommended Reading
{:.no_toc}

* Beil, Ester and Xu (2002)
* Chang et al. (2009)
* Gilardi et al. (2017)
* Lucas et al (2015)
* Manning, Raghavan and Schütze (2008, Ch. 16–17)
* James et al. (2013, Ch. 10.3)
* Zumel and Mount (2014, Ch. 8)

#### Resources
{:.no_toc}

- [Exercise 5: Classifying legal documents and legislative speeches.](/assets/courses/tcd2018qta/exercise5.html)  


## Final Assignment

#### Project Guidelines
{:.no_toc}

The final project is a written analysis of approximately 4,000-5,000 words. The project replaces a final exam, and is designed to give you a chance to analyze a set of texts that you have chosen, reflecting (hopefully) your own research interests. This can be, and probably makes the most sense to be, textual data from something you are already studying. Which texts you choose, what question you investigate, and how you analyze the texts is your choice, but you must justify the choice.

#### Content
{:.no_toc}

Your content should include the following:

1.  A cover sheet including the title and your name.  
2.  An abstract page, with an abstract of no more than 200 words.  
3.  Introduction.  An expanded version of your abstract, which introduces the question, states the rationale for trying to answer it, briefly describes your corpus, identifies the methods you apply to the texts, and summarizes the findings.
4.  Motivation. Why have you chosen to analyze this topic? Is there a compelling social reason? Does it contribute to scholarship? This can include a “literature review” (but don’t overdo it).
5.  Description of your corpus. You are free to choose any corpus you wish, of any size, although you must include a justification for the choice of texts, and acknowledge the source. You will need to document any format conversions or pre-processing steps you have applied to the texts prior to analysis. You should also present some basic summary statistics about the texts, prior to analysis.
6.  Description of your methods. What techniques will you apply to analyze the texts? Is there a precedent (in previous scholarly literature) for applying such methods to texts similar to yours? Defend why the application of the methods is appropriate.
7.  Results. Apply the methods, present the findings. Be sure to be explicit about any steps taken.
8.  Conclusions. What conclusions on the question can we draw from the results?

#### Formatting
{:.no_toc}

There is no rigid set of guidelines, but you should use a Chicago manual of style compatible referencing system (parenthetical references rather than footnotes). Tables should be formatted rather than consisting of pasted output from a statistical package. Think of trying to look (roughly) as if you were formatting a journal article.

#### Deadline
{:.no_toc}

30 April 2018, 5pm, by email to [kbenoit@tcd.ie](mailto:kbenoit@tcd.ie).


## References

Barberá, Pablo. 2015. "Birds of the Same Feather Tweet Together: Bayesian Ideal Point Estimation Using Twitter Data." _Political Analysis_ 23(1):76--91. doi: [`10.1093/pan/mpu011`](https://doi.org/10.1093/pan/mpu011).

Beauchamp, N. 2017. "[Predicting and Interpolating State‐Level Polls Using Twitter Textual Data.](http://onlinelibrary.wiley.com/doi/10.1111/ajps.12274/full)" _American Journal of Political Science_ 61(2), 490-503.

Beil, F, M Ester and X Xu. 2002. Frequent term-based text clustering. In _Eighth ACM SIGKDD international conference on Knowledge discovery and data mining_. pp. 436–442.

Benoit, K. and M. Laver. 2008. “Compared to What? A Comment on ‘A Robust Transformation
Procedure for Interpreting Political Text’ by Martin and Vanberg.” _Political Analysis_ 16(1):101–111. doi: [`10.1093/pan/mpm020`](https://doi.org/10.1093/pan/mpm020).

Benoit, Kenneth and Paul Nulty. 2013. “Classification Methods for Scaling Latent Political Traits.” Presented at the Annual Meeting of the Midwest Political Science Association, April 11–14, Chicago.

Blei, David M. 2012. “Probabilistic topic models.” _Communications of the ACM_ 55(4):77. doi: [`10.1145/2133806.2133826`](https://doi.org/10.1145/2133806.2133826).

Blei, D.M., A.Y. Ng and M.I. Jordan. 2003. “Latent dirichlet allocation.” _The Journal of Machine Learning Research_ 3:993–1022.

Chang, J., J. Boyd-Graber, S. Gerrish, C. Wang and D. Blei. 2009. Reading tea leaves: How humans interpret topic models. In _Neural Information Processing Systems._

Choi, Seung-Seok, Sung-Hyuk Cha and Charles C. Tappert. 2010. “A Survey of Binary Similarity and Distance Measures.” _Journal of Systemics, Cybernetics and Informatics_ 8(1):43–48.

Clinton, J., S. Jackman and D. Rivers. 2004. “The statistical analysis of roll call voting: A unified approach.” _American Journal of Political Science_ 98(2):355–370. doi: [`10.1017/s0003055404001194`](https://doi.org/10.1017/s0003055404001194).

Corley, Courtney and Rada Mihalcea. 2005. Measuring the semantic similarity of texts. In _Proceedings of the ACL Workshop on Empirical Modeling of Semantic Equivalence and Entailment - EMSEE ’05_.

Däubler, Thomas, Kenneth Benoit, Slava Mikhaylov and Michael Laver. 2012. “Natural Sentences as Valid Units for Coded Political Texts.” _British Journal of Political Science_ 42(4):937–951. doi: [`10.1017/S0007123412000105`](https://doi.org/10.1017/S0007123412000105).

DuBay, William. 2004. The Principles of Readability. Costa Mesa, California. <http://www.impact-information.com/impactinfo/readability02.pdf>.

Dunning, Ted. 1993. “Accurate methods for the statistics of surprise and coincidence.” _Computational Linguistics_ 19:61–74.

Evans, Michael, Wayne McIntosh, Jimmy Lin and Cynthia Cates. 2007. “Recounting the Courts?
Applying Automated Content Analysis to Enhance Empirical Legal Research.” _Journal of Empirical Legal Studies_ 4(4):1007–1039.

Gilardi, F., Shipan, C. R., & Wueest, B. 2017. "[Policy Diffusion: The Issue-Definition Stage.](https://www.fabriziogilardi.org/resources/papers/policy-diffusion-issue-definition.pdf)" Working paper, University of Zurich.

Ginsberg, Jeremy, Matthew H Mohebbi, Rajan S Patel, Lynnette Brammer, Mark S Smolinski and Larry Brilliant. 2008. “Detecting influenza epidemics using search engine query data.” _Nature_ 457(7232):1012–1014.

Grimmer, Justin and Brandon M. Stewart. 2013. “Text as Data: The Promise and Pitfalls of Automatic Content Analysis Methods for Political Texts.” _Political Analysis_ 21(3):267–297. doi: [`10.1093/pan/mps028`](https://doi.org/10.1093/pan/mps028).

Gurciullo, S., & Mikhaylov, S. J. 2016. "[Detecting Policy Preferences and Dynamics in the UN General Debate with Neural Word Embeddings](http://www.smikhaylov.net/wp-content/uploads/2017/04/UNembeddings.pdf)". Working paper, University College London.

James, Gareth, Daniela Witten, Trevor Hastie and Robert Tibshirani. 2013. _An Introduction to Statistical Learning with Applications in R_. Springer Science & Business Media.

Jürgens, Pascal and Andreas Jungherr. 2016. “A Tutorial for Using Twitter Data in the Social Sciences: Data Collection, Preparation, and Analysis.”

Klašnja, M., Barberá, P., Beauchamp, N., Nagler, J., & Tucker, J. 2016. "[Measuring public opinion with social media data.](http://www.oxfordhandbooks.com/view/10.1093/oxfordhb/9780190213299.001.0001/oxfordhb-9780190213299-e-3)" In The Oxford Handbook of Polling and Survey Methods.

Krippendorff, Klaus. 2013. _Content Analysis: An Introduction to Its Methodology_. 3rd ed. Thousand Oaks, CA: Sage.

Lampos, Vasileios, Daniel Preotiuc-Pietro and Trevor Cohn. 2013. A user-centric model of voting intention from Social Media. In _Proceedings of the 51st Annual Meeting of the Association for Computational Linguistics (ACL)_.

Lantz, Brett. 2013. _Machine Learning with R._ Packt Publishing Ltd.

Laver, M. and J. Garry. 2000. “Estimating policy positions from political texts.” _American Journal of Political Science_ 44(3):619–634. doi: [`10.2307/2669268`](https://doi.org/10.2307/2669268).

Laver, Michael, Kenneth Benoit and John Garry. 2003. “Estimating the policy positions of political actors using words as data.” _American Political Science Review_ 97(2):311–331. doi: [`10.1017/S0003055403000698`](https://doi.org/10.1017/S0003055403000698).

Loughran, Tim and Bill McDonald. 2011. “When Is a Liability Not a Liability? Textual Analysis, Dictionaries, and 10-Ks.” _The Journal of Finance_ 66(1):35–65.

Lowe, W. 2008. “Understanding Wordscores.” _Political Analysis_ 16(4):356–371. doi: [`10.1093/pan/mpn004`](https://doi.org/10.1093/pan/mpn004).

Lowe, William and Kenneth Benoit. 2013. “Validating Estimates of Latent Traits From Textual Data Using Human Judgment as a Benchmark.” _Political Analysis_ 21(3):298–313. doi: [`10.1093/pan/mpt002`](https://doi.org/10.1093/pan/mpt002).

Lowe, William, Kenneth Benoit, Slava Mikhaylov and Michael Laver. 2011. “Scaling Policy Preferences From Coded Political Texts.” _Legislative Studies Quarterly_ 26(1):123–155. doi: [`10.1111/j.1939-9162.2010.00006.x`](https://doi.org/10.1111/j.1939-9162.2010.00006.x).

Lucas, C., Nielsen, R. A., Roberts, M. E., Stewart, B. M., Storer, A., & Tingley, D. 2015. "[Computer-assisted text analysis for comparative politics.](http://scholar.princeton.edu/sites/default/files/bstewart/files/comparativepoliticstext.pdf)" Political Analysis, 23(2), 254-277.

Manning, C. D., P. Raghavan and H. Schütze. 2008. _Introduction to Information Retrieval_. Cambridge University Press.

Martin, L. W. and G. Vanberg. 2007. “A robust transformation procedure for interpreting political text.” _Political Analysis_ 16(1):93–100. doi: [`10.1093/pan/mpm010`](https://doi.org/10.1093/pan/mpm010).

Metaxas, Panagiotis T., Eni Mustafaraj and Daniel Gayo-Avello. 2011. How (not) to predict elections. In _Privacy, security, risk and trust (PASSAT), 2011 IEEE third international conference on social computing (SocialCom)_.

Mikolov, T., Chen, K., Corrado, G., & Dean, J. 2013. "Efficient estimation of word representations in vector space." [arXiv preprint arXiv:1301.3781.](https://arxiv.org/pdf/1301.3781.pdf)

Neuendorf, K. A. 2002. _The Content Analysis Guidebook_. Thousand Oaks CA: Sage.

Pennebaker, J. W. and C. K. Chung. 2008. Computerized text analysis of al-Qaeda transcripts. In _The Content Analysis Reader_, ed. K. Krippendorf and M. A. Bock. Thousand Oaks, CA: Sage.

Roberts, Margaret E, Brandon M Stewart, Dustin Tingley, Christopher Lucas, Jetson Leder-Luis, Shana Kushner Gadarian, Bethany Albertson and David G Rand. 2014. “Structural Topic Models for Open-Ended Survey Responses.” _American Journal of Political Science_ 58(4):1064–1082. doi: [`10.1080/01621459.2016.1141684`](https://doi.org/10.1080/01621459.2016.1141684).

Rooduijn, Matthijs and Teun Pauwels. 2011. “Measuring Populism: Comparing Two Methods of
Content Analysis.” _West European Politics_ 34(6):1272–1283.

Ruths, D., & Pfeffer, J. 2014. "[Social media for large studies of behavior.](http://science.sciencemag.org/content/346/6213/1063.full)" Science, 346(6213), 1063-1064.

Seale, Clive, Sue Ziebland and Jonathan Charteris-Black. 2006. “Gender, cancer experience and internet use: A comparative keyword analysis of interviews and online cancer support groups.” _Social Science & Medicine_ 62(10):2577–2590.

Slapin, Jonathan B. and Sven-Oliver Proksch. 2008. “A Scaling Model for Estimating Time-Series Party Positions from Texts.” _American Journal of Political Science_ 52(3):705–722. doi: [`10.1111/j.1540-5907.2008.00338.x`](10.1111/j.1540-5907.2008.00338.x).

Steinert-Threlkeld, Z. 2018. "[Twitter as Data.](https://www.cambridge.org/core/elements/twitter-as-data/27B3DE20C22E12E162BFB173C5EB2592)" Cambridge University Press.

Tausczik, Y R and James W Pennebaker. 2010. “The Psychological Meaning of Words: LIWC and
Computerized Text Analysis Methods.” _Journal of Language and Social Psychology_ 29(1):24–54.

Young, L., and Soroka, S. 2012. "[Affective news: The automated coding of sentiment in political texts.](http://www.tandfonline.com/doi/abs/10.1080/10584609.2012.671234)" _Political Communication_ 29(2), 205-231.

Yu, B., S. Kaufmann and D. Diermeier. 2008. “Classifying Party Affiliation from Political Speech.” _Journal of Information Technology and Politics_ 5(1):33–48.

Zumel, Nina and John Mount. 2014. _Practical Data Science with R_. Manning Publications.

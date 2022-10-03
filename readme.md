<a id="TOP"> </a>
<div class="alert alert-box alert-info">
</div>

***

# FINAL REPORT

## by David Howell 

# District Attorney Actions Based on Arrests - Project Summary

## Project Goals

> - Document code, process (data acquistion, preparation, exploratory data analysis and statistical testing, modeling, and model evaluation), findings, and key takeaways.
> - Create modules that make the process repeateable and the report (notebook) easier to read and follow.
> - Ask exploratory questions of the data that will help to understand more about the attributes and drivers of DA Actions. 
> - Refine the work into a report, in the form of a jupyter notebook, that I will walk through in a 5 minute presentation to our data science team about our goals, the work done, findings, methodologies, and conclusions.
> - Be prepared to answer panel questions about my code, process, findings and key takeaways, and model.

## Project Description

### Business Goals

> - Utilize a random forest for exploration and model prediction
> - Find the key drivers DA Actions in the San Fransico Courts.
> - Deliver a report that the data science team can read through and replicate, understand what steps were taken, why and what the outcome was.
> - Make recommendations on what works or doesn't work in predicting future DA Actions.

### Deliverables

> - **Readme (.md)**
> - **A Python Module or Modules that automate the data acquisition and preparation process**
> - **Final Report (.ipynb)**
> - 5 min Recorded Presentation

## Data Dictionary

|**Target**|**Definition**
|:-------|:----------|
|**DA Action**|**Limited to Charges Filed or Discharged**|

|**Feature**|**Definition**|
|:-------|:----------|
|**arresting agency**                 |**SF Police, Sheriff, Highway Patrol, or Bay Area Rapid Transit**|
|**crime type**              |**Examined the top 7 crimes by occurence**|
|**booked case type**                     |**Felony or Misdemeanor**|
|**domestic violence case?**   |**was it a domestic violence case**|
|**arrest date**                |**What was the Date of arrest**| 
|**year**                  |**What Year was that in**|


## Initial Hypotheses
> We believe that by using these predictors we can create a model that will beat basline predictions. 

## Executive Summary - Key Findings and Recommendations
> 1. Utilizing three different random forest models we were able to increase the accuracy of our predicitons in a meaningful way (60% up to nearly 80%).

> 2. The newly created features via feature engineering did something to add value to the DA Actions Taken dataset in terms of model creation.

> 3. Our recommendations are that we maybe delve deeper into clustering techniques if given more time possibly brining in other features such as  and maybe looking into more time sensitive data such as arrest dates, etc.


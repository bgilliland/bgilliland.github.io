---
title: "Tableau: Coal Mining Machine Maintenance"
date: 2020-05-20
tags: [Tableau, Tableau Stories, Tableau Dashboards, Data Analysis, Power Insights]
excerpt: "Determine which machines need Maintenance based on capacity usage"
mathjax: "true"
---

## Background and Analysis: When does a coal mining machine need maitenence?
This project is from Kirill Eremenko's Advanced Tableau course on Udemy. It has fake data from an imaginary coal mining operation. The executives would like to know which of the machines need maintenance. This is an important question because many coal mining companies rely on these machines to be working efficiently 24/7 because they do not shut down their operations. If a machine breaks down this means millions of dollars in lost revenue. Preemptive maintenance can minimize lost revenue and thus is a crucial aspect for a data analyst to consider.

How can we be able to determine which machines may need maintenance? We look at the capacity at which they're running, by hour, over the course of a month. It has been decided that if a machine is operating at more than a 10% differential below its nominal capacity for 8 hours or more at any point during a given month then we decide to do maintenance on that machine.

Using custom Table Calculations in Tableau, we can determine %-differential in operating capacity using an 8-hr moving average. Thus, at each individual point we can see how the machine performed for the past 8 hours on average. 

<iframe src="https://public.tableau.com/views/CoalMiningMaintenenceAnalysis/Report?:showVizHome=no&:embed=true" width="100%" height="800"></iframe>

This Data Visualization is a Story made up of multiple dashboards. Each frame describes one of the machines, how it performed, how it is trending, and our advice as data experts on whether to perform maintenance on that machine.
---
title: "Excel & R: An Assessment of FGCU's Learning Assistant Program"
date: 2020-05-16
tags: [R, ggplot2, Excel]
excerpt: "Statistical Analysis of Learning Assistants"
mathjax: "true"
---

## Background 

Learning Assistants (LA's) are talented undergraduates who have recently taken a course and remember what it is like to learn the material. They help transform undergraduate courses to include small groups of students articulating, defending, and modifying their ideas about relevant problems or phenomena. Their main role is to support student learning in interactive classroom environments, working with small groups of students as they solve challenging conceptual or mathematical problems.

FGCU’s Learning Assistant program began in 2016 and spanned a wide range of STEM disciplines. In the last year, it has expanded to non-STEM classes as well. We are interested in measuring the effectiveness of the program and determining methods of improvement for the future based on DFW rates.


## Analysis

How can we get a grasp on how to quantify LA impact on student performance? Let's consider our data. We have data from Fall 2016 through Spring 2019 on a by-semester, and by-section granularity level.

```r
library(tidyverse)
library(dplyr)
library(xtable)
library(ggplot2)
library(knitr)

# Import CSV
dat = read.csv("LA_DFW_Data.csv")[,1:14] 
# Create new column for total DFW counts
dat$DFW = dat$DF.Number+dat$W.Number
# Filter columns by ones relevant to our analysis
dat = dat[,c(1,4,5,6,7,8,10,12,15)] 

attach(dat)
head(dat)
```

    ##     Course                  Course.Name      Term  LA   CRN Avg..DFW.RATE
    ## 1 EVR1001C Intro. Environmental Science Fall 2018  No 82604          0.07
    ## 2 EVR1001C Intro. Environmental Science Fall 2018  No 82605          0.10
    ## 3 EVR1001C Intro. Environmental Science Fall 2018  No 83202          0.08
    ## 4 EVR1001C Intro. Environmental Science Fall 2018 Yes 83350          0.08
    ## 5 COP2006   Introduction to Programming Fall 2016 Yes 80599          0.04
    ## 6 COP2006   Introduction to Programming Fall 2016  No 80600          0.09
    ##   Students DFW
    ## 1       72   5
    ## 2       69   7
    ## 3       75   6
    ## 4       38   3
    ## 5       24   1
    ## 6       32   3

As you can see by the head of our .csv, we know what subject the section was (columns 1 and 2), what term it was taught in, whether there was an LA or not, which section (CRN) it was (and therefore who taught it), what the DFW (D/F/Withdraw) rate was for that section, how many students were in it, and the DFW count, respectively.

## Paired T-test for DFW Rate, pairing by Section. Looking for an LA Effect.

We begin at the top-most and most granular level: comparaing sections without respect to instructor, term, or course. Simply, do LA's lower DFW Rates? We have plenty of data (51 sections with an LA, 239 sections without an LA) so we may proceed with parametric testing. Welch's Two-Sample, left-tailed t-test is appropriate, as determined by R automatically, since the sample sizes are unequal by a significant margin \($$\alpha = .05$$\).

Let's also take a look at the histogram for their differences, just to confirm it is approximately normally distributed, as well as the desnity plot of the two cases to get a sense of how the distribution of DFW rates compare between LA and Non-LA sections. 

```r
# Get DFW Rates for sections where an LA is/isn't present
LA_DFW<-dat$Avg..DFW.RATE[dat$LA=="Yes"]
NonLA_DFW<-dat$Avg..DFW.RATE[dat$LA=="No"]

# Get histogram for differences in DFW rates 
hist(LA_DFW-NonLA_DFW, xlab = "Difference between DFW Rates Accross Sections",main=paste("Histogram of the \nDifference between DFW Rates Accross Sections"))

# Perform parametric test for differences
t.test(LA_DFW,NonLA_DFW,alternative="less") 

# Plot densities for when an LA is/isn't present, accross sections
plot(density(dat$Avg..DFW.RATE[dat$LA=="Yes"]),xlim=c(0,.7),ylim=c(0,3.5),main = paste("Distribution of DFW Rates Across Sections \n with LA's or without LA's"),xlab = "DFW Rate") 
lines(density(dat$Avg..DFW.RATE[dat$LA=="No"]),col="red")
legend(x=.5,y=3,col = c("Black", "Red"),legend=c("No","Yes"),lwd=1) #Format curves and legends
```

    ## 
    ##  Welch Two Sample t-test
    ## 
    ## data:  LA_DFW and NonLA_DFW
    ## t = -0.38902, df = 70.485, p-value = 0.3492
    ## alternative hypothesis: true difference in means is less than 0
    ## 95 percent confidence interval:
    ##        -Inf 0.02501183
    ## sample estimates:
    ## mean of x mean of y 
    ## 0.2600000 0.2676151




<!-- ![](/Users/blakegilliland/Documents/GitHub/blakegilliland.github.io/images/LA-analysis_files/figure-gfm/unnamed-chunk-3-1.png) -->

![]({{ site.url }}{{ site.baseurl }}/images/LA-analysis_files/figure-gfm/unnamed-chunk-3-1.png)


<!-- ![](/Users/blakegilliland/Documents/GitHub/blakegilliland.github.io/images/LA-analysis_files/figure-gfm/unnamed-chunk-7-2.png) -->

![]({{ site.url }}{{ site.baseurl }}/images/LA-analysis_files/figure-gfm/unnamed-chunk-7-2.png)

With a p-value >.05 we may conclude that the average DFW Rate for sections with LA's are not significantly lower than those without LA's. This, as mentioned, does not account for multiple confounding variables such as course subject and instructor. So, let's try to do so. 

## Paired T-test for DFW Rate, pairing by Course. Looking for an LA Effect.

Let's aggregate our data so that we have DFW info by course and by LA presence. Then we can examine the shape of our data and perform a test of differences to see if LA's have a significantly positive impact on DFW rates.

```r
# Aggregate DFW count by LA and course as a sum
tab1 = aggregate(DFW,list(LA,Course.Name),sum)

# Aggregate number of students by LA and course as a sum
tab2 = aggregate(Students,list(LA,Course.Name),sum)

# Combine columns for LA and Course and DFW info by performing average on the DFW rate
tab3 = cbind(tab1,round(tab1[,3]/tab2[,3],2)) 

# Create table
colnames(tab3) = c("LA","Course","DFW Count","DFW Rate") # Title the columns
kable(tab3)
```

| LA  | Course                       | DFW Count | DFW Rate |
| :-- | :--------------------------- | --------: | -------: |
| No  | Calculus I                   |       124 |     0.33 |
| Yes | Calculus I                   |        39 |     0.28 |
| No  | Calculus II                  |        26 |     0.20 |
| Yes | Calculus II                  |        12 |     0.34 |
| No  | College Algebra              |       784 |     0.28 |
| Yes | College Algebra              |        88 |     0.31 |
| No  | College Physics w/Lab II     |         5 |     0.16 |
| Yes | College Physics w/Lab II     |         5 |     0.16 |
| No  | Elementary Calculus          |        76 |     0.35 |
| Yes | Elementary Calculus          |        50 |     0.47 |
| No  | Engineering Mechanics        |        14 |     0.27 |
| Yes | Engineering Mechanics        |         6 |     0.16 |
| No  | Finite Mathematics           |        23 |     0.11 |
| Yes | Finite Mathematics           |        18 |     0.17 |
| No  | Gen’l Biology w/Lab I        |       477 |     0.34 |
| Yes | Gen’l Biology w/Lab I        |        40 |     0.35 |
| No  | General Chemistry I          |       682 |     0.37 |
| Yes | General Chemistry I          |       178 |     0.39 |
| No  | General Chemistry II         |       138 |     0.33 |
| Yes | General Chemistry II         |       152 |     0.41 |
| No  | Intermediate Algebra         |       464 |     0.22 |
| Yes | Intermediate Algebra         |        62 |     0.29 |
| No  | Intro Earth Science          |        47 |     0.15 |
| Yes | Intro Earth Science          |        22 |     0.16 |
| No  | Intro to Computer Science    |        36 |     0.22 |
| Yes | Intro to Computer Science    |        31 |     0.17 |
| No  | Intro. Environmental Science |        18 |     0.08 |
| Yes | Intro. Environmental Science |         3 |     0.08 |
| No  | Introduction to Programming  |         6 |     0.12 |
| Yes | Introduction to Programming  |        21 |     0.13 |
| No  | Precalculus                  |       211 |     0.25 |
| Yes | Precalculus                  |        53 |     0.22 |
| No  | Social Science Statistics    |        12 |     0.18 |
| Yes | Social Science Statistics    |         5 |     0.16 |
| No  | Statistical Methods          |       674 |     0.25 |
| Yes | Statistical Methods          |        23 |     0.21 |

We need to check the shape of the data to see if it is normal enough to apply a parametric test. The Shapiro-Wilk Test for Normality will do, since our data set is relatively small. When interested in a test for differences, we must use the differences of the DFW rates for categories we are concerned about (LA/Non-LA) in the test for normality.

```r
# Get DFW rate by course when there is/isn't an LA
LA_DFW<-tab3$`DFW Rate`[tab3$LA=="Yes"] 
NonLA_DFW<-tab3$`DFW Rate`[tab3$LA=="No"]

# Perform test for normality on the difference between our DFW rates for LA and Non-LA courses
shapiro.test(LA_DFW-NonLA_DFW) 

# Create histogram for data used in normality test
hist(LA_DFW-NonLA_DFW,xlab = "Difference between DFW Rates by Course",main=paste("Histogram of the differences between \nDFW Rates by Courses with/without LA's")) 
```
    ## 
    ##  Shapiro-Wilk normality test
    ## 
    ## data:  LA_DFW - NonLA_DFW
    ## W = 0.97282, p-value = 0.8487
    
![]({{ site.url }}{{ site.baseurl }}/images/LA_Paper_files/figure-gfm/unnamed-chunk-2-1.png)

<!-- ![](/Users/blakegilliland/Documents/GitHub/blakegilliland.github.io/images/LA_Paper_files/figure-gfm/unnamed-chunk-2-1.png) -->   

As can be seen not only by the result of our test, but also the shape of the data, the data certainly is approximately normal and thus we may perform parametric tests, specifically a two-sample, left-tailed, paired t-test \($$\alpha = .05$$\).

```r
# Perform test for differences
t.test(LA_DFW,NonLA_DFW,paired=TRUE,alternative="less") 

# Plot density curves for LA/non-LA courses
plot(density(NonLA_DFW),xlim=c(0,.7),main = "Distribution of DFW Rates for Courses with LA's or without LA's",xlab = "DFW Rate") 
lines(density(LA_DFW),col="red")
legend(x=.5,y=3.5,col = c("Black", "Red"),legend=c("No","Yes"),lwd=1)
```

    ## 
    ##  Paired t-test
    ## 
    ## data:  LA_DFW and NonLA_DFW
    ## t = 0.93843, df = 17, p-value = 0.8194
    ## alternative hypothesis: true difference in means is less than 0
    ## 95 percent confidence interval:
    ##        -Inf 0.03963536
    ## sample estimates:
    ## mean of the differences 
    ##              0.01388889


![]({{ site.url }}{{ site.baseurl }}/images/LA-analysis_files/figure-gfm/unnamed-chunk-5-1.png)

<!-- ![](/Users/blakegilliland/Documents/GitHub/blakegilliland.github.io/images/LA-analysis_files/figure-gfm/unnamed-chunk-5-1.png) -->

We get a test statistic of t = 0.938 and a p-value that is much greater than our set \($$\alpha = .05$$\) level. Since our statistic is >0, this indicates that the average DFW rate accross courses actually goes up when LA's are present.

## Wilcoxon Signed-Rank Test for DFW Rate, pairing by Instructor. Looking for an LA Effect.

We now will consider a new dataset with similar information. Here we consider DFW Rates by instructor instead of by course as seen previously. We will assign instructors an ID in the "Instructors" column for anonymity.

```r
# Import dataset for DFW data paired by instructor
dat1 = read.csv("PairedData.csv")[,4:15]
# Get a column for DFW count
dat1$DFW = dat1$DF.Number+dat1$W.Number
attach(dat1)

# Perform similar aggregation as previous table but pairing by instructor instead of by course
tab1 = aggregate(DFW,list(LA,Instructors),sum) 
tab2 = aggregate(Students,list(LA,Instructors),sum)
tab3 = cbind(tab1,round(tab1[,3]/tab2[,3],2))
colnames(tab3) = c("LA","Instructors","DFW Count","DFW Rate") 
kable(tab3)
```

| LA  | Instructors | DFW Count | DFW Rate |
| :-- | ----------: | --------: | -------: |
| No  |           1 |        25 |     0.32 |
| Yes |           1 |        27 |     0.36 |
| No  |           2 |        19 |     0.26 |
| Yes |           2 |        21 |     0.30 |
| No  |           3 |        25 |     0.17 |
| Yes |           3 |        42 |     0.29 |
| No  |           4 |         2 |     0.06 |
| Yes |           4 |         3 |     0.09 |
| No  |           5 |        13 |     0.23 |
| Yes |           5 |        13 |     0.23 |
| No  |           6 |        29 |     0.33 |
| Yes |           6 |        25 |     0.28 |
| No  |           7 |        20 |     0.56 |
| Yes |           7 |        11 |     0.16 |
| No  |           8 |        42 |     0.49 |
| Yes |           8 |        45 |     0.48 |
| No  |           9 |         6 |     0.08 |
| Yes |           9 |         3 |     0.08 |
| No  |          10 |         5 |     0.16 |
| Yes |          10 |         5 |     0.16 |
| No  |          11 |        29 |     0.54 |
| Yes |          11 |        50 |     0.47 |
| No  |          12 |         5 |     0.16 |
| Yes |          12 |         5 |     0.16 |
| No  |          13 |        45 |     0.48 |
| Yes |          13 |        45 |     0.49 |
| No  |          14 |         6 |     0.19 |
| Yes |          14 |        14 |     0.20 |
| No  |          15 |        14 |     0.16 |
| Yes |          15 |        10 |     0.19 |


Similarly to our previous data exploration we need to examine the shape of the data. Specifically, we need to examine the shape of the data made up of the differences between matched pairs by instructor for when an LA is present and when one is not.

```r
# Get LA/Non-LA DFW rates by instructor
LA_DFW<-tab3$`DFW Rate`[tab3 == "Yes"]
NonLA_DFW<-tab3$`DFW Rate`[tab3$LA == "No"]

# Perform test for normality on difference between data
shapiro.test(LA_DFW - NonLA_DFW) 

# Show shape of data used in test for normality
hist(LA_DFW - NonLA_DFW,xlab = "Difference between DFW Rates by Instructor",main=paste("Histogram of the differences between \nDFW Rates by Instructors with/without LA's")) 
```

    ## 
    ##  Shapiro-Wilk normality test
    ## 
    ## data:  LA_DFW - NonLA_DFW
    ## W = 0.63002, p-value = 4.869e-05

![]({{ site.url }}{{ site.baseurl }}/images/LA-analysis_files/figure-gfm/unnamed-chunk-7-1.png)

<!-- ![](/Users/blakegilliland/Documents/GitHub/blakegilliland.github.io/images/LA-analysis_files/figure-gfm/unnamed-chunk-7-1.png) -->

Our p-value is approx. 0 and thus we reject the shape of the data being normal. As a result, we may not use parametric tests and we will revert to the nonparametric equivalent of the two-sample, left tailed, paired t-test: the Wilcoxon Signed-Rank Test \($$\alpha = .05$$\).

```r
# Perform nonparametric test
wilcox.test(LA_DFW,NonLA_DFW,alternative = "less",paired = T) 

# Plot densities similar to before but for new data
plot(density(Instruct_LA),xlim=c(0,.7),main = paste("Distribution of DFW Rates for Instructors \n that Taught Sections with LA's or without LA's"),xlab = "DFW Rate")
lines(density(Instruct_No_LA),col="red")
legend(x=.5,y=2.5,col = c("Black", "Red"),legend=c("No","Yes"),lwd=1) 
```

    ## 
    ##  Wilcoxon signed rank test with continuity correction
    ## 
    ## data:  LA_DFW and NonLA_DFW
    ## V = 36, p-value = 0.6225
    ## alternative hypothesis: true location shift is less than 0
    
![]({{ site.url }}{{ site.baseurl }}/images/LA_Paper_files/figure-gfm/unnamed-chunk-9-1.png)

<!-- ![](/Users/blakegilliland/Documents/GitHub/blakegilliland.github.io/images/LA_Paper_files/figure-gfm/unnamed-chunk-9-1.png) -->

We observe a p-value greater than our significance level and thus conclude that instructors do not observe a drop in DFW rates when an LA is present compared to when they are not present.


## Conclusion

It is evident from the data we have that LA's do not have the impact on students that they are meant to be having. However, it is our conjecture that there is reason to believe that there are underlying truths in the data that were not represented in the data due to lack of foresight when recording the data. 

For example, courses with a Learning Assistant had only one and no Instructional Assistants. However, courses with no Learning Assistants may have had an Instructional Assistant. So, in our Non-LA data we may have had courses who had an assistant in the classroom who, though not having the official role of a Learning Assistant, could be trained as one and is utilizing the techniques they have learned. This would cause a leftward shift in the densities of the DFW rates for sections without LA's.

One further question worth considering is if we can do better with our data? Are DFW rates the best indicator for Learning Assistant impact on student performance? Perhaps not. This too could account for lack of differences.
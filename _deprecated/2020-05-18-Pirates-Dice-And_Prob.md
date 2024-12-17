---
title: "Probability and Pirates Dice"
date: 2020-05-18
tags: [R, Probability Theory, Game Theory]
excerpt: "Determine probabilities and optimal strategies in dice games"
mathjax: "true"
---

## Background

I love playing games, particularly ones that require strategy based on probability or some form of mathematics. One of these games is called Pirate's Dice, based on Liar's Dice. You may be familiar with it if you have seen Pirates of Caribbean: Dead Man's Chest. In one scene, a number of characters partake in a betting game where they wage years of service to the Flying Dutchman.

## Gameplay

The rules are simple: You begin with 5 dice in your cup and only you know what dice you have. When it is your turn, you make a wager on how many of a kind are on the table between all players (ex. three 5's, or ten 3's ect.). The person immediately to your left decides on if they think your wager is correct. If they believe it is then they will raise the call. They may do so in one of the following ways: 

* Raise the face number and keep the volume of dice the same
* Change the face number to anything they'd like (or keep it as is) and raise the volume of dice

The process is repeated with the person to their immediate left. If, however, the person to your left does *not* believe your wager to be right, i.e. there is strictly fewer of a kind than what you've waged, then they will say so. At this point, every player reveals the dice in their cup and a count is made. The one who was wrong between the wager and the person who called them will lose the difference from what was true and what was waged from their cup. So, if you call five 3's and there is actually seven of them then the caller (the person to your left) will lose two dice from his cup.

**Here is the twist that makes this game amazing:** one of the sides of each of the dice is a wildcard, called a "Pirate Flag", or a just a "flag". *Whatever face of the dice was called on the last wager of that round, i.e. the wager that got called, is what the wildcard of each and every player will count as.* So, if the last wager was three 5's, for example, then every flag in play will count as a 5.

Something to remember is that it is called Liar's Dice for a reason! You can be as dishonest and untruthful as your heart desires. You may wager however any of whatever dice face you would like. Just don't get caught!

<!--![](/Users/blakegilliland/Documents/GitHub/blakegilliland.github.io/images/PiratesDiceCups.jpg)-->
![]({{ site.url }}{{ site.baseurl }}/images/PiratesDiceCups.jpg) 

This is a last man standing game. Whoever has dice last wins!


## Analyzing Probabilities

Clearly, there is a lot of room for interesting probabilistic questions. I wanted to focus on ones that I thought, if we could get the answers to them, that could give a decisive advantage in forming waging/calling strategies.

After playing this game many times, I realized that there are scenarios that put you in the drivers seat on any particular wage. If you have 3-of-a-kind or better, then you have a lot of leverage over a wage. When you have a lot of a certain kind then you're able to rely less on others giving your wage support which means a greater probability that you're right. Simultaneously, the person to your left who needs to decide on whether to call you or not does not know that you have such a strong hand and may want to call you since it seems unlikely that more than half of your dice are of the same kind.

Okay, so what is the probability of getting 3-of-a-kind or better? By how much does the flag increase the odds of rolling 3-of-a-kind or better?

```r
rf_3<-c()
rf_4<-c()
rf_5<-c()

rf_33<-c()
rf_43<-c()
rf_44<-c()
rf_53<-c()
rf_54<-c()
rf_55<-c()

library(pastecs)
library(psych)

# Simulate 5000 trials of 100 rolls of 5 dice and count how many of them have a 3/4/5-of-a-kind or more
# WITH flags (wildcards)
dice<-seq(1,6,1)

for (m in 3:5) { # let 'm' be the number of dice in a cup
  
  for (l in 1:5000) { # let 'l' be the current trial index
    a<-0
    for (k in 1:100) { # let 'k' be the index of the current roll in the l'th trial
      roll<-sample(1:6,size = m,replace = T) # roll m dice
      
      for (i in 1:length(dice)){ # let 'i' represent the face on our template dice to compare our roll to
        b<-0 # keep track of how many identical dice we have
        
        for (j in 1:length(roll)) { # let 'j' be the face of a dice in our sample roll
          if(roll[j]==dice[i]|roll[j]==1){ # account for a flag by letting a face value of 1 be considered a match
            b<-b+1 # incrememnt when we get matching dice
          }else{
            next # move onto the next die in our sample roll if not
          }
        }
        
        if(b>2){ # once done comparing every dice in our sample roll to the template die, count how many were 3 of a kind or not
          a<-a+1
          break
        }else{
          next
        }
      }
      
    }
    rf_3[l]<-a # save the total number of cases out of our 100 rolls when we got 3-of-a-kind or better
   
   
    # Repeat process as above but for 4-of-a-kind or more only when there's at least 4 dice in our cup (m>3)
    a<-0
    if(m>3){
    for (k in 1:100) {
      
        roll<-sample(1:6,size = m,replace = T)
        for (i in 1:length(dice)){
          b<-0
          for (j in 1:length(roll)) {
            if(roll[j]==dice[i]|roll[j]==1){
              b<-b+1
            }else{
              next
            }
          }
          if(b>3){
            a<-a+1
            break
          }else{
            next
          }
        }
      }
    }else{next}
    rf_4[l]<-a
    
    
    # Simulate 100 rolls and counts how many of them have a 5-of-a-kind
    a<-0
    if(m==5){
        
    for (k in 1:100) {
        roll<-sample(1:6,size = m,replace = T)
        for (i in 1:length(roll)){
          b<-0
          for (j in 1:length(roll)) {
            if(roll[j]==dice[i]|roll[j]==1){
              b<-b+1
            }else{
              next
            }
          }
          if(b>4){
            a<-a+1
            break
          }else{
            next
          }
        }
      }
    }else{next}
    rf_5[l]<-a
  }
  if(m==3){
    rf_33<-rf_3
  }else if (m==4){
    rf_43<-rf_3
    rf_44<-rf_4
  }else{
    rf_53<-rf_3
    rf_54<-rf_4
    rf_55<-rf_5
  }
}
hist(rf_33)
hist(rf_43)
hist(rf_44)
hist(rf_53)
hist(rf_54)
hist(rf_55)
}
```
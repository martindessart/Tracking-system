## 0) Libraries ----
library(lme4)
library(ggplot2)
library(dplyr)
rm(list=ls())
## 1) Open file ----
setwd("/Users/martin/Desktop/2)Data/2_Verified/Aedes")
filename="R2"
trial_nb <- 12 #number of trials -> 12 for group Expe & Ctrl.1, 1 for Ctrl.2
raw<-read.table(file=paste(filename,"_V.csv",sep=""), sep="",header=T)
raw %>% count(ID)
frame_max<-max(raw$frame) #max frames
## 2) Find stimulation response ranges ----
#By looking at the video, enter the second from which the first trial is applied
beg_1trial <- 127 #sec 
end_1trial <- beg_1trial+3 #sec 
adjust <- 22 #frames to adjust precisely the location of the 1st trial
Find_stimuli_ranges <- function(beg_1trial,end_1trial) {
  s1d<-(beg_1trial)*25 #Convert in frames
  s1f<-((end_1trial)*25)-1 #Convert in frames
  dist0 <- vector("numeric", trial_nb)
  dist1 <- vector("numeric", trial_nb)
  step<-1
  counter<-vector("numeric", trial_nb)
  for (i in 1:trial_nb) { #For each trial, calculate the frame number associated
    dist0[i]<-(s1d+(step*25)-adjust)
    dist1[i]<-(s1f+(step*25)-adjust)
    step<-step+120
    counter[i]<-i
  }
  df <- data.frame(counter,dist0,dist1) #extract ranges 
  colnames(df) <- c("Trial","frameD","frameF")
  return(df)
}
Sti_range<-Find_stimuli_ranges(beg_1trial,end_1trial)

## 3) Display trial responses ----
Display_response <-function(ranges,nb_Sti) {
  deb<-ranges[nb_Sti,2] #Find good ranges from the df
  fin<-ranges[nb_Sti,3] #Find good ranges from the df
  sti<-raw %>% 
    filter(frame>deb-500&frame<fin+500) #Extract data
  ggplot(sti)+ #Display data
    geom_line(aes(x=frame,y=-pos_y,colour=as.factor(ID)))+
    theme_light()+
    geom_rect(data=ranges,aes(xmin=deb, xmax=fin, 
                          ymin=(-400), ymax=(0)), fill='#0677d704',color=NA)
}
Display_response(Sti_range,1) # input = Df + trial number to plot
Display_response(Sti_range,5) # Half way
Display_response(Sti_range,10) # last trial of training
Display_response(Sti_range,11) # nothing for Group expe or Vibration for Ctrl. 1
Display_response(Sti_range,12) # test phase
#If the ranges are not well-fitted, increase or decrease frame number and repeat ## 2) and 3) 

#Special case for Ctrl.1 on Aedes: latency for the mechanical vibration was not controlled enough
# --> Manually shift ranges for the 11th and 12th trials:
Sti_range<-Sti_range %>% filter(Trial!=11) %>% 
  add_row(Trial=11,frameD=34260,frameF=(frameD+74)) %>% 
  arrange(Trial)
#Check
Display_response(Sti_range,11) # nothing for Group expe or Vibration for Ctrl. 1

Sti_range<-Sti_range %>% filter(Trial!=12) %>% 
  add_row(Trial=12,frameD=37430,frameF=(frameD+74)) %>% 
  arrange(Trial)
#Check
Display_response(Sti_range,12) # nothing for Group expe or Vibration for Ctrl. 1

## 4) Classify data ----
Sti_Data<- function(rg) {
  df_tot = data.frame()
  for (i in 1:trial_nb) {
    nam<-paste("st",i,sep="")
    interval<-
      nam1<-raw %>% 
      filter(frame>=rg[i,2] & frame< rg[i,3]) %>% 
      mutate(cat=i) %>% 
      mutate(gp="Sti") #Add variable that separate Sti and Iti
    
    df_tot<-rbind(df_tot,nam1)
  }
  return(df_tot)
}
Df_Sti<-Sti_Data(Sti_range)

Interval_Data<- function(rg) {
  df_tot = data.frame()
  for (i in 1:(trial_nb-3)) {
    nam1<-raw %>%
      filter(frame>rg[i,3] & frame<rg[(i+1),2]) %>% 
      mutate(cat=i) %>% 
      mutate(gp="ITI")
    df_tot<-rbind(df_tot,nam1)
  }
  return(df_tot)
}
Df_Iti<-Interval_Data(Sti_range)

### Check and combine ----
Df_Sti %>% count(cat)
Df_Iti %>% count(cat)
Df_Tot<-bind_rows(Df_Sti,Df_Iti)

setwd("/Users/martin/Desktop/2)Data/3_Classified/Aedes")
write.table(Df_Tot, file=paste(filename, "_C.csv", sep=""))

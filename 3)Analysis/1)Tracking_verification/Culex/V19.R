## Libraries ----
library(lme4)
library(ggplot2)
library(dplyr)
library(zoo)
rm(list=ls())
## Open file ----
setwd("/Users/martin/Desktop/2)Data/1_Raw/Culex")
filename="R19"
raw1<-read.table(file=paste(filename,"_A_tracked.csv",sep=""), sep=",",header=T)
raw2<-read.table(file=paste(filename,"_B_tracked.csv",sep=""), sep=",",header=T)

## Check individual identities on x axis  ----
# Since each individual is physically separated from the others, 
# we use the horizontal coordinate to classify them

classId1<- function(rr) {
  df_tot<-data.frame() 
  n_ind<-c("A","B","C","D","E","F","G","H") # 8 individuals HERE
  #calculate x horizontal distance
  dist<-(max(rr$pos_x)+30)-min(rr$pos_x) # total distance
  c<-dist/8 # x distance for each individual
  #filter and extract
  for(i in 0:7) { 
    namnam<-paste("st",n_ind[i],sep="") 
    namnam<-rr %>% 
      filter(pos_x>((i*c)) & pos_x<(((i+1)*c))) %>% 
      mutate(ID=i+1)
    
    df_tot<-rbind(df_tot,namnam)
  }
  return(df_tot)
}
classId2<- function(rr) {
  df_tot<-data.frame() 
  n_ind<-c("A","B","C") # 3 individuals HERE
  #calculate x horizontal distance
  dist<-(max(rr$pos_x)+30)-min(rr$pos_x) # total distance
  c<-dist/3 # x distance for each individual
  #filter and extract
  for(i in 0:2) { 
    namnam<-paste("st",n_ind[i],sep="") 
    namnam<-rr %>% 
      filter(pos_x>((i*c)) & pos_x<(((i+1)*c))) %>% 
      mutate(ID=i+1)
    
    df_tot<-rbind(df_tot,namnam)
  }
  return(df_tot)
}
IR1<-classId1(raw1)
IR2<-classId2(raw2)
IR1$ID<-as.factor(IR1$ID)
IR2b<-IR2 %>% filter(ID==3) %>% mutate(ID=9) %>% mutate(pos_x=pos_x+750)
IR2b$ID<-as.factor(IR2b$ID) 
IR<-bind_rows(IR1,IR2b)
IR<-IR %>%
  select(!id) %>% #delete old classification
  mutate(cat="ITI")
## Filter ----
# we create different variables to create the distance between two position
# and average data by rollmean function
IR2<-IR %>% 
  group_by(ID) %>% 
  mutate(sec=round(frame/10,digits=0)) %>% #Average by 10
  mutate(y2=rollmean(pos_y,10,na.pad=TRUE)) %>% #Moving average
  mutate(m=lag(y2)) %>% #shift column posy from one position
  mutate(dY=m-y2) %>% #create speed = difference from 2 successive positions
  mutate(absdY=abs(dY)) %>% #create absolute dt
  group_by(ID,sec) %>% 
  summarise(frame=mean(frame),
            pos_y=mean(pos_y),
            pos_x=mean(pos_x),
            dY=sum(dY,na.rm=T),
            absdY=sum(absdY,na.rm=T))

# we have also compared our data frames with the theorethical maximum number of frames
f1<-max(IR2$frame) #max frames
f2<-f1*nlevels(IR2$ID) #f1 x 10 individuals 
f3<-nrow(IR) #number of lines
f3/f2 #Check ratio

## Check visually that individuals are correctly identified and that data looks consistent ----
ggplot(IR2)+
  geom_density(aes(x=frame,colour=ID))
ggplot(IR2)+
  geom_line(aes(x=frame,y=pos_x,colour=ID))

## Save file ----

setwd("/Users/martin/Desktop/2)Data/2_Verified/Culex")
write.table(IR2, file=paste(filename, "_V.csv", sep=""))

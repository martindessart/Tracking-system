## Libraries ----
library(lme4)
library(ggplot2)
library(dplyr)
library(zoo)
rm(list=ls())
## Open file ----
setwd("/Users/martin/Desktop/2)Data/1_Raw/Culex")
filename="R11"
#2 TRACKING IN 3 PARTS
raw1<-read.table(file=paste(filename,"_A_tracked.csv",sep=""), sep=",",header=T)
raw2<-read.table(file=paste(filename,"_B_tracked.csv",sep=""), sep=",",header=T)
raw3<-read.table(file=paste(filename,"_C_tracked.csv",sep=""), sep=",",header=T)

## Check individual identities on x axis  ----
# Since each individual is physically separated from the others, 
# we use the horizontal coordinate to classify them

classId<- function(rr) {
  df_tot<-data.frame()
  n_ind<-c("A","B","C","D","E","F","G","H","I","J")
  #calculate distance
  dist<-(max(rr$pos_x)+30)-min(rr$pos_x)
  c<-dist/10
  #filter and extract
  for(i in 0:9) {
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
  n_ind<-c("A","B","C","D","E","F","G","H")
  #calculate distance
  dist<-(max(rr$pos_x)+30)-min(rr$pos_x)
  c<-dist/8
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
classId3<- function(rr) {
  df_tot<-data.frame()
  n_ind<-c("A","B","C","D","E")
  #calculate distance
  dist<-(max(rr$pos_x)+30)-min(rr$pos_x)
  c<-dist/5
  #filter and extract
  for(i in 0:4) {
    namnam<-paste("st",n_ind[i],sep="")
    namnam<-rr %>% 
      filter(pos_x>((i*c)) & pos_x<(((i+1)*c))) %>% 
      mutate(ID=i+1)
    
    df_tot<-rbind(df_tot,namnam)
  }
  return(df_tot)
}

r1<-raw1 %>% filter(frame>29500) %>% mutate(frame=frame-29500)
IR1<-classId2(r1)
IR1<-IR1 %>% mutate(ID=ID+2) %>% mutate(pos_x=pos_x+250)
IR1$ID<-as.factor(IR1$ID)
IR1<-IR1 %>%select(!id) %>% mutate(cat="ITI")
r2<-raw2 %>% filter(frame<30100) %>% mutate(frame=frame+645)
IR2<-classId(r2)
IR2$ID<-as.factor(IR2$ID)
IR2<-IR2 %>%filter(ID!=2) %>% select(!id) %>% mutate(cat="ITI")
r3<-raw3 %>% mutate(frame=frame+30100)
IR3<-classId3(r3)
IR3<-IR3 %>% mutate(ID=ID+5) %>% mutate(pos_x=pos_x+620)
IR3$ID<-as.factor(IR3$ID)
IR3<-IR3 %>%select(!id) %>% mutate(cat="ITI")
IR<-bind_rows(IR1,IR2,IR3) %>% 
  filter(ID==6|ID==7|ID==8|ID==9|ID==10) 

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
f2<-f1*nlevels(IR3$ID) #f1 x 10 individuals 
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

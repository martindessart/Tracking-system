## 0) Libraries ----
library(dplyr)
rm(list=ls())
set.seed(666) #Fix kernell
## 1) Open files and add metadata ----
setwd("/Users/martin/Desktop/2)Data/3_Classified/Anopheles")
R20<-read.table("R20_C.csv", sep="",header=T) %>% mutate(ct=2) %>% mutate(pck=4) %>% mutate(day=8) %>% mutate(expe=20)
R21<-read.table("R21_C.csv", sep="",header=T) %>% mutate(ct=1) %>% mutate(pck=4) %>% mutate(day=8) %>% mutate(expe=21)
R22<-read.table("R22_C.csv", sep="",header=T) %>% mutate(ct=2) %>% mutate(pck=4) %>% mutate(day=8) %>% mutate(expe=22)
R23<-read.table("R23_C.csv", sep="",header=T) %>% mutate(ct=1) %>% mutate(pck=4) %>% mutate(day=8) %>% mutate(expe=23)
R24<-read.table("R24_C.csv", sep="",header=T) %>% mutate(ct=2) %>% mutate(pck=4) %>% mutate(day=8) %>% mutate(expe=24)
R25<-read.table("R25_C.csv", sep="",header=T) %>% mutate(ct=3) %>% mutate(pck=4) %>% mutate(day=9) %>% mutate(expe=25)
R26<-read.table("R26_C.csv", sep="",header=T) %>% mutate(ct=3) %>% mutate(pck=4) %>% mutate(day=9) %>% mutate(expe=26)
R27<-read.table("R27_C.csv", sep="",header=T) %>% mutate(ct=1) %>% mutate(pck=5) %>% mutate(day=10) %>% mutate(expe=27)
R28<-read.table("R28_C.csv", sep="",header=T) %>% mutate(ct=3) %>% mutate(pck=5) %>% mutate(day=10) %>% mutate(expe=28)

## 2) Affect individuals per group ----
#GROUP EXPE ---
X23<-R23 %>% 
  mutate(ID=ID+10)
X27<-R27 %>% 
  mutate(ID=ID+20)
XG1<-bind_rows(R21,X23,X27)
XG1 %>% count(ID)

#GROUP CONTROL 1 ---
X22<-R22 %>% 
  mutate(ID=ID+10)
X24<-R24 %>% 
  mutate(ID=ID+20)
XG2<-bind_rows(R20,X22,X24)
XG2 %>% count(ID)

#GROUP CONTROL 2 ---
X26<-R26 %>% 
  mutate(ID=ID+10)
X28<-R28 %>% 
  mutate(ID=ID+20)
XG3<-bind_rows(R25,X26,X28)
XG3 %>% count(ID)

## Calculate number of diving events ---- 
countDiving<-function(arrive,ID){
  counter<-0 #Number of Top 
  count2<-0 #Number of down
  traj<-0 #Trajectory reminder
  df1 = data.frame()
  df_tot = data.frame()
  
  for (j in 1:ID) { #across table's rows
    sa<-arrive %>% filter(ID==j)
    if (nrow(sa)<1) {
      next
    } 
    else { 
      print(c("ID",j))
      sx<-arrive %>% filter(ID==j)
    }
    for (i in 1:nrow(sx)) { #across table's rows
      #if larvae going up
      if (traj<1) {
        #print("going up")
        
        #if larvae is near the top
        if(sx[i,4]>280){
          counter<-counter+1 #count 1 Top
          traj<-1 #change Trem
          #print("Up")
        }
      }
      
      #if larvae going down
      else { 
        #print("going down")
        #if larvae is near the bottom
        if(sx[i,4]<120){
          count2<-count2+1 #count 1 Down
          traj<-0 #change Trem
          #print("OK")
        }
      }
    }
    sx2<-sx %>% 
      mutate(plong=counter)
    print(c(counter,"Ok"))
    df_tot<-rbind(df_tot,sx2)
    #track<-j
    counter<-0
  }
  return(df_tot)
}
XGA<-countDiving(XG1,30)
XGB<-countDiving(XG2,30)
XGC<-countDiving(XG3,30)

Df_Tot<-bind_rows(XGA,XGB,XGC)
Df_Tot$cat<-as.factor(Df_Tot$cat)
Df_Tot$ID<-as.factor(Df_Tot$ID)
Df_Tot$gp<-as.factor(Df_Tot$gp)
Df_Tot$ct<-as.factor(Df_Tot$ct)

## Convert pixels in mm ----
# Calculate mean vertical length 
a1<-max(R21$pos_y)-min(R21$pos_y)
a2<-max(R22$pos_y)-min(R22$pos_y)
a3<-max(R23$pos_y)-min(R23$pos_y)
a4<-max(R24$pos_y)-min(R24$pos_y)
a5<-max(R25$pos_y)-min(R25$pos_y)
a6<-max(R26$pos_y)-min(R26$pos_y)
a7<-max(R27$pos_y)-min(R27$pos_y)
a8<-max(R28$pos_y)-min(R28$pos_y)
a9<-max(R20$pos_y)-min(R20$pos_y)

AT<-(a1+a2+a3+a4+a5+a6+a7+a8+a9)/9

# Convert pixels in mm : 
# Length cuvette = 45mm
# Pixel number = AT

Df_Tot2<-Df_Tot %>% 
  mutate(dY=dY*45/AT)
str(Df_Tot)

## Save file ----
setwd("/Users/martin/Desktop/2)Data/4_Grouped")
write.table(Df_Tot2,"Anopheles.csv")



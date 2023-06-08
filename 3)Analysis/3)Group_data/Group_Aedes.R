## 0) Libraries ----
library(dplyr)
rm(list=ls())
set.seed(666) #Fix kernell
## 1) Open files and add metadata ----
setwd("/Users/martin/Desktop/2)Data/3_Classified/Aedes")
R1<-read.table("R1_C.csv", sep="",header=T) %>% mutate(ct=1) %>% mutate(pck=1) %>% mutate(day=8) %>% mutate(expe=1)
R2<-read.table("R2_C.csv", sep="",header=T) %>% mutate(ct=2) %>% mutate(pck=1) %>% mutate(day=9) %>% mutate(expe=2)
R3<-read.table("R3_C.csv", sep="",header=T) %>% mutate(ct=3) %>% mutate(pck=1) %>% mutate(day=9) %>% mutate(expe=3)
R4<-read.table("R4_C.csv", sep="",header=T) %>% mutate(ct=2) %>% mutate(pck=1) %>% mutate(day=9) %>% mutate(expe=4)
R5<-read.table("R5_C.csv", sep="",header=T) %>% mutate(ct=3) %>% mutate(pck=1) %>% mutate(day=9) %>% mutate(expe=5)
R6<-read.table("R6_C.csv", sep="",header=T) %>% mutate(ct=1) %>% mutate(pck=1) %>% mutate(day=10) %>% mutate(expe=6)
R7<-read.table("R7_C.csv", sep="",header=T) %>% mutate(ct=2) %>% mutate(pck=1) %>% mutate(day=10) %>% mutate(expe=7)
R8<-read.table("R8_C.csv", sep="",header=T) %>% mutate(ct=3) %>% mutate(pck=1) %>% mutate(day=10) %>% mutate(expe=8)
R9<-read.table("R9_C.csv", sep="",header=T) %>% mutate(ct=1) %>% mutate(pck=1) %>% mutate(day=11) %>% mutate(expe=9)

## 2) Affect individuals per group ----
#GROUP EXPE ---
X6<-R6 %>% 
  mutate(ID=ID+10)
X9<-R9 %>% 
  mutate(ID=ID+20)
XG1<-bind_rows(R1,X6,X9)
XG1 %>% count(ID)

#GROUP CONTROL 1 ---
X4<-R4 %>% 
  mutate(ID=ID+10)
X7<-R7 %>% 
  mutate(ID=ID+20)
XG2<-bind_rows(R2,X4,X7)
XG2 %>% count(ID)

#GROUP CONTROL 2 ---
X5<-R5 %>% 
  mutate(ID=ID+10)
X8<-R8 %>% 
  mutate(ID=ID+20)
XG3<-bind_rows(R3,X5,X8)
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
a1<-max(R1$pos_y)-min(R1$pos_y)
a2<-max(R2$pos_y)-min(R2$pos_y)
a3<-max(R3$pos_y)-min(R3$pos_y)
a4<-max(R4$pos_y)-min(R4$pos_y)
a5<-max(R5$pos_y)-min(R5$pos_y)
a6<-max(R6$pos_y)-min(R6$pos_y)
a7<-max(R7$pos_y)-min(R7$pos_y)
a8<-max(R8$pos_y)-min(R8$pos_y)
a9<-max(R9$pos_y)-min(R9$pos_y)

AT<-(a1+a2+a3+a4+a5+a6+a7+a8+a9)/9

# Convert pixels in mm : 
# Length cuvette = 45mm
# Pixel number = AT

Df_Tot2<-Df_Tot %>% 
  mutate(dY=dY*45/AT)
str(Df_Tot)

## Save file ----
setwd("/Users/martin/Desktop/2)Data/4_Grouped")
write.table(Df_Tot2,"Aedes.csv")



## 0) Libraries ----
library(dplyr)
rm(list=ls())
set.seed(666) #Fix kernell
## 1) Open files and add metadata ----
setwd("/Users/martin/Desktop/2)Data/3_Classified/Culex")
R10<-read.table("R10_C.csv", sep="",header=T) %>% mutate(ct=2) %>% mutate(pck=2) %>% mutate(day=5) %>% mutate(expe=10)
R11<-read.table("R11_C.csv", sep="",header=T) %>% mutate(ct=2) %>% mutate(pck=2) %>% mutate(day=5) %>% mutate(expe=11)
R12<-read.table("R12_C.csv", sep="",header=T) %>% mutate(ct=2) %>% mutate(pck=2) %>% mutate(day=5) %>% mutate(expe=12)
R13<-read.table("R13_C.csv", sep="",header=T) %>% mutate(ct=1) %>% mutate(pck=2) %>% mutate(day=6) %>% mutate(expe=13)
R14<-read.table("R14_C.csv", sep="",header=T) %>% mutate(ct=2) %>% mutate(pck=3) %>% mutate(day=7) %>% mutate(expe=14)
R15<-read.table("R15_C.csv", sep="",header=T) %>% mutate(ct=1) %>% mutate(pck=3) %>% mutate(day=7) %>% mutate(expe=15)
R16<-read.table("R16_C.csv", sep="",header=T) %>% mutate(ct=1) %>% mutate(pck=3) %>% mutate(day=7) %>% mutate(expe=16)
R17<-read.table("R17_C.csv", sep="",header=T) %>% mutate(ct=3) %>% mutate(pck=3) %>% mutate(day=7) %>% mutate(expe=17)
R18<-read.table("R18_C.csv", sep="",header=T) %>% mutate(ct=3) %>% mutate(pck=3) %>% mutate(day=7) %>% mutate(expe=18)
R19<-read.table("R19_C.csv", sep="",header=T) %>% mutate(ct=3) %>% mutate(pck=3) %>% mutate(day=7) %>% mutate(expe=19)

## 2) Affect individuals per group ----
#GROUP EXPE ---
X15<-R15 %>% 
  mutate(ID=ID+10)
X16<-R16 %>% 
  mutate(ID=ID+20)
XG1<-bind_rows(R13,X15,X16)
XG1 %>% count(ID)

#GROUP CONTROL 1 ---
X14 %>% count(ID)
R14 %>% count(ID)
X11<-R11 %>% 
  mutate(ID=ID+5)
X12<-R12 %>% 
  mutate(ID=ID+15)
X14<-R14 %>% 
  mutate(ID=ID+25)
XG2<-bind_rows(R10,X11,X12,X14)
XG2 %>% count(ID)

#GROUP CONTROL 2 ---
X18<-R18 %>% 
  mutate(ID=ID+10)
X19<-R19 %>% 
  mutate(ID=ID+20)
XG3<-bind_rows(R17,X18,X19)
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
XGB<-countDiving(XG2,40)
XGC<-countDiving(XG3,30)

Df_Tot<-bind_rows(XGA,XGB,XGC)
Df_Tot$cat<-as.factor(Df_Tot$cat)
Df_Tot$ID<-as.factor(Df_Tot$ID)
Df_Tot$gp<-as.factor(Df_Tot$gp)
Df_Tot$ct<-as.factor(Df_Tot$ct)

## Convert pixels in mm ----
# Calculate mean vertical length 
a1<-max(R11$pos_y)-min(R11$pos_y)
a2<-max(R12$pos_y)-min(R12$pos_y)
a3<-max(R13$pos_y)-min(R13$pos_y)
a4<-max(R14$pos_y)-min(R14$pos_y)
a5<-max(R15$pos_y)-min(R15$pos_y)
a6<-max(R16$pos_y)-min(R16$pos_y)
a7<-max(R17$pos_y)-min(R17$pos_y)
a8<-max(R18$pos_y)-min(R18$pos_y)
a9<-max(R19$pos_y)-min(R19$pos_y)
a10<-max(R10$pos_y)-min(R10$pos_y)

AT<-(a1+a2+a3+a4+a5+a6+a7+a8+a9+a10)/10

# Convert pixels in mm : 
# Length cuvette = 45mm
# Pixel number = AT

Df_Tot2<-Df_Tot %>% 
  mutate(dY=dY*45/AT)
str(Df_Tot)




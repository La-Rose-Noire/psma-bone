library(foreign)
library(survival)
library(rms)
newbancer <- read.csv('newbancer.csv')
newbancer <- as.data.frame(newbancer)
head(newbancer)

newbancer$subtype<- factor(newbancer$subtype,order=TRUE) #设置为等级变量
options(contrasts=c("contr.treatment", "contr.treatment")) #指定等级变量的参照水平


dd<-datadist(newbancer)
options(datadist='dd')

str(newbancer)

newbancer$T <- as.numeric(newbancer$T)
newbancer$N3 <- as.numeric(newbancer$N3)
newbancer$ER <- as.numeric(newbancer$ER)
newbancer$PR <- as.numeric(newbancer$PR)
newbancer$SUVmax <- as.numeric(newbancer$SUVmax)
newbancer$TMTV <- as.numeric(newbancer$TMTV)
newbancer$SUVL <- as.numeric(newbancer$SUVL)
newbancer$SUVS <- as.numeric(newbancer$SUVS)
newbancer$SUVM <- as.numeric(newbancer$SUVM)
newbancer$SLR <- as.numeric(newbancer$SLR)
newbancer$BLR <- as.numeric(newbancer$BLR)
newbancer$RFS <- as.numeric(newbancer$RFS)
newbancer$time <- as.numeric(newbancer$time)
newbancer$score <- as.numeric(newbancer$score)
newbancer$risk <- as.numeric(newbancer$risk)

##做多因素Cox回归分析、作列线图
Model4<-coxph(Surv(time,RFS)~N3+ER+PR+TMTV+SLR,x=T,y=T,data=newbancer)
Model4
summary(Model4)
library(survminer) #载入所需R包
ggforest(Model4, #直接用前面多因素cox回归分析的结果
         main = "Hazard ratio",
         cpositions = c(0.02,-0.15, 0.25), #前三列的位置，第二列是样品数，设了个负值，相当于隐藏了
         fontsize = 0.8, #字体大小
         refLabel = "reference", 
         noDigits = 3) 


Model4<-cph(Surv(newbancer$time,newbancer$RFS==1)~N3+ER+PR+TMTV+SUVS+SLR,x=T,y=T,data=newbancer,surv=T)
Model4
summary(Model4)
Model3<-cph(Surv(newbancer$time,newbancer$RFS==1)~N3+ER+PR+TMTV+SLR,x=T,y=T,data=newbancer,surv=T)
Model3
summary(Model3)
Model2<-cph(Surv(newbancer$time,newbancer$RFS==1)~N3+ER+PR+TMTV+SUVS,x=T,y=T,data=newbancer,surv=T)
Model2
summary(Model2)
Model1<-cph(Surv(newbancer$time,newbancer$RFS==1)~N3+ER+PR+TMTV,x=T,y=T,data=newbancer,surv=T)
Model1
summary(Model1)

surv <- Survival(Model4)
surv1 <- function(x)surv(1*36,lp=x)
surv2 <- function(x)surv(1*60,lp=x)
nom1<-nomogram(Model4,fun=list(surv1,surv2),lp = F,
               funlabel=c('36-Month recurrence probability',
                          '60-Month recurrence probability'),
               maxscale=100,
               fun.at=c('0.9','0.85','0.75','0.65','0.55','0.40','0.25','0.15','0.05'))
plot(nom1)

library(ggplot2)

##绘制校准曲线
library(pec)
calPlot(list("Model1" = Model1,"Model2" = Model2,"Model3" = Model3,"Model4" = Model4),
                    data=newbancer,
                    legend.x=0.5,legend.y=0.3,legend.cex=0.8,
                    splitMethod = "BootCV",
                    B=1000)

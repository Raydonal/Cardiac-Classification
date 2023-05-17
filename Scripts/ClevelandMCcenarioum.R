rm(list=ls())

##########################################
#              PACOTES
##########################################
library(MASS)
library(car)
library(xtable)
library(latex2exp)
library(FSelector)
library(FSelectorRcpp)
library(RWeka)
library(RWekajars)
library(e1071)
library(fastAdaboost)
library(neuralnet)
library(GenAlgo)
library(ClassDiscovery)
library(RSNNS)
library(randomForest)

###### DADOS
dados=read.table("Clevelanddata.txt",na.strings = -9,header=T,dec=".",sep = "\t")
attach(dados)

##########################################
#         RECODIFICANDO O BANCO
##########################################

sex=factor(sex,levels =c(1,0),labels =  c("Male","Female"))
painloc=factor(painloc,levels = c(1,0),labels = c("substernal","otherwise"))
painexer=factor(painexer,levels=c(1,0),labels=c("provoked","otherwise")) 
relrest=factor(relrest,levels = c(1,0),labels = c("relieved","otherwise"))
cp=factor(cp,levels=c(1,2,3,4),labels=c("typical","atypical","nonanginal","asymptomatic")) 
htn=factor(htn,levels =c(1,0),labels =  c("yes","No"))
smoke=factor(smoke,levels =c(1,0),labels =  c("Yes","No"))
fbs=factor(fbs,levels =c(1,0),labels =  c("Yes","No"))
dm=factor(dm,levels =c(1,0),labels =  c("Yes","No"))
famhist=factor(famhist,levels =c(1,0),labels =  c("Yes","No"))
restcg=factor(restcg,levels =c(0,1,2),labels =  c("normal","anormal","hypertrophy"))
dig=factor(dig,levels =c(1,0),labels =  c("Yes","No"))
prop=factor(prop,levels =c(1,0),labels =  c("Yes","No"))
nitr=factor(nitr,levels =c(1,0),labels =  c("Yes","No"))
pro=factor(pro,levels =c(1,0),labels =  c("Yes","No"))
diuretic=factor(diuretic,levels =c(1,0),labels =  c("Yes","No"))
exang=factor(exang,levels =c(1,0),labels =  c("Yes","No"))
xhypo=factor(xhypo,levels =c(1,0),labels =  c("Yes","No"))
slope=factor(slope,levels =c(1,2,3),labels =  c("up","flat","down"))

##########################################
#              VARIAVEIS
##########################################

PS=trestbps
PD=trestbpd
FC=thalrest
#Perio=(1/FC)*60
Perio=1/(FC/60)

##########################################
#              PARAMETROS
##########################################

PAM=(PS-PD)/(log(PS)-log(PD))
IPPA=(PS-PD)/PD
RC=1/(FC*log(PS/PD))
IPPARC=IPPA/RC
HM=((1000/(FC/60))^2)/((PS-PAM)^3)
ALPHA=1/2-(1/(2*Perio))*sqrt(abs(Perio^2-4*(((PS-PD)*0.001)^2)))
ALPHA2=log(1/ALPHA)

PAM=scale(PAM)
IPPA=scale(IPPA)
RC=scale(RC)
IPPARC=scale(IPPARC)
HM=scale(HM)
ALPHA=scale(ALPHA)
ALPHA2=scale(ALPHA2)


X=data.frame(PAM,IPPA,RC,IPPARC,HM,ALPHA,ALPHA2)
Y=factor(ifelse(num==0,0,1))

newdata = data.frame(cbind(age,sex,cp,trestbps,htn,chol,cigs,years,fbs,dm,famhist,restecg,ekgmo,ekgday,ekgyr,dig,prop,nitr,pro,diuretic,proto,thaldur,thaltime,met,thalach,thalrest,tpeakbps,tpeakbpd,dummy,trestbpd,exang,xhypo,oldpeak,slope,rldv5e,ca,thal,cmo,cday,cyr,lmt,ladprox,laddist,cxmain,om1,rcaprox,rcadist,lvx1,lvx2,lvx3,lvx4,lvf,X,Y))
B=100
models=6
metodos=5


acurNBtrei=matrix(,B,models);colnames(acurNBtrei)=c("A","B","C","D","E","F")
acurDTtrei=matrix(,B,models);colnames(acurDTtrei)=c("A","B","C","D","E","F")
acurSVMtrei=matrix(,B,models);colnames(acurSVMtrei)=c("A","B","C","D","E","F")
acurLRtrei=matrix(,B,models);colnames(acurLRtrei)=c("A","B","C","D","E","F")
acurADBtrei=matrix(,B,models);colnames(acurADBtrei)=c("A","B","C","D","E","F")

acurNBtest=matrix(,B,models);colnames(acurNBtest)=c("A","B","C","D","E","F")
acurDTtest=matrix(,B,models);colnames(acurDTtest)=c("A","B","C","D","E","F")
acurSVMtest=matrix(,B,models);colnames(acurSVMtest)=c("A","B","C","D","E","F")
acurLRtest=matrix(,B,models);colnames(acurLRtest)=c("A","B","C","D","E","F")
acurADBtest=matrix(,B,models);colnames(acurADBtest)=c("A","B","C","D","E","F")

tempoNB=matrix(,B,models);colnames(tempoNB)=c("A","B","C","D","E","F")
tempoDT=matrix(,B,models);colnames(tempoDT)=c("A","B","C","D","E","F")
tempoSVM=matrix(,B,models);colnames(tempoSVM)=c("A","B","C","D","E","F")
tempoLR=matrix(,B,models);colnames(tempoLR)=c("A","B","C","D","E","F")
tempoADB=matrix(,B,models);colnames(tempoADB)=c("A","B","C","D","E","F")

matconftrei=matrix(0,2*metodos,2*models);colnames(matconftrei)=rep(c(0,1),models);rownames(matconftrei)=rep(c(0,1),metodos)
matconftest=matrix(0,2*metodos,2*models);colnames(matconftest)=rep(c(0,1),models);rownames(matconftest)=rep(c(0,1),metodos)
k=0 #contador

# 1 - Y~sex+htn
# 2 - Y~sex+age+htn+PAM+IPPA+HM+ALPHA
# 3 - Y~sex+age+PAM+IPPARC
# 4 - Y~sex+age+PAM
# 5 - Y~sex+age+IPPA+IPPARC+ALPHA
# 6 - Y~sex+age

set.seed(26071993)

for (i in 1:B) {
  
treinamento=sample(nrow(newdata),round(nrow(newdata)*0.7))
dadostrei=newdata[treinamento,]
dadostest=newdata[-treinamento,]

##########################################
#             CLASSIFICACAO
#              NAIVE BAYES
##########################################

tempo0NB1=proc.time()[3]
modNB1=naiveBayes(Y~sex+htn,data = dadostrei)
YestNB1trei=predict(modNB1,newdata = dadostrei);YestNB1test=predict(modNB1,newdata = dadostest)
tempoNB[i,1]=proc.time()[3]-tempo0NB1

tempo0NB2=proc.time()[3]
modNB2=naiveBayes(Y~sex+age+htn+PAM+IPPA+HM+ALPHA,data = dadostrei);YestNB2=predict(modNB2,newdata = dadostest)
YestNB2trei=predict(modNB2,newdata = dadostrei);YestNB2test=predict(modNB2,newdata = dadostest)
tempoNB[i,2]=proc.time()[3]-tempo0NB2

tempo0NB3=proc.time()[3]
modNB3=naiveBayes(Y~sex+age+PAM+IPPARC,data = dadostrei) #### MELHOR
YestNB3trei=predict(modNB3,newdata = dadostrei);YestNB3test=predict(modNB3,newdata = dadostest)
tempoNB[i,3]=proc.time()[3]-tempo0NB3

tempo0NB4=proc.time()[3]
modNB4=naiveBayes(Y~sex+age+PAM,data = dadostrei)
YestNB4trei=predict(modNB4,newdata = dadostrei);YestNB4test=predict(modNB4,newdata = dadostest)
tempoNB[i,4]=proc.time()[3]-tempo0NB4

tempo0NB5=proc.time()[3]
modNB5=naiveBayes(Y~sex+age+IPPA+IPPARC+ALPHA,data = dadostrei)
YestNB5trei=predict(modNB5,newdata = dadostrei);YestNB5test=predict(modNB5,newdata = dadostest)
tempoNB[i,5]=proc.time()[3]-tempo0NB5

tempo0NB6=proc.time()[3]
modNB6=naiveBayes(Y~sex+age,data = dadostrei)
YestNB6trei=predict(modNB6,newdata = dadostrei);YestNB6test=predict(modNB6,newdata = dadostest)
tempoNB[i,6]=proc.time()[3]-tempo0NB6

##########################################
#             CLASSIFICACAO
#           ARVORE DE DECISAO
##########################################

tempo0DT1=proc.time()[3]
DadostestRF1=na.omit(data.frame(dadostest$Y,dadostest$sex,dadostest$htn))
RF1=randomForest(Y~sex+htn,data = dadostrei,xtest=DadostestRF1[,-1],ytest=DadostestRF1$dadostest.Y,na.action = na.omit)
tempoDT[i,1]=proc.time()[3]-tempo0DT1

tempo0DT2=proc.time()[3]
DadostestRF2=na.omit(data.frame(dadostest$Y,dadostest$sex,dadostest$age,dadostest$htn,dadostest$PAM,dadostest$IPPA,dadostest$HM,dadostest$ALPHA))
RF2=randomForest(Y~sex+age+htn+PAM+IPPA+HM+ALPHA,data = dadostrei,xtest=DadostestRF2[,-1],ytest=DadostestRF2$dadostest.Y,na.action = na.omit)
tempoDT[i,2]=proc.time()[3]-tempo0DT2

tempo0DT3=proc.time()[3]
DadostestRF3=na.omit(data.frame(dadostest$Y,dadostest$sex,dadostest$age,dadostest$PAM,dadostest$IPPARC))
RF3=randomForest(Y~sex+age+PAM+IPPARC,data = dadostrei,xtest=DadostestRF3[,-1],ytest=DadostestRF3$dadostest.Y,na.action = na.omit)
tempoDT[i,3]=proc.time()[3]-tempo0DT3

tempo0DT4=proc.time()[3]
DadostestRF4=na.omit(data.frame(dadostest$Y,dadostest$sex,dadostest$age,dadostest$PAM))
RF4=randomForest(Y~sex+age+PAM,data = dadostrei,xtest=DadostestRF4[,-1],ytest=DadostestRF4$dadostest.Y,na.action = na.omit)
tempoDT[i,4]=proc.time()[3]-tempo0DT4

tempo0DT5=proc.time()[3]
DadostestRF5=na.omit(data.frame(dadostest$Y,dadostest$sex,dadostest$age,dadostest$IPPA,dadostest$IPPARC,dadostest$ALPHA))
RF5=randomForest(Y~sex+age+IPPA+IPPARC+ALPHA,data = dadostrei,xtest=DadostestRF5[,-1],ytest=DadostestRF5$dadostest.Y,na.action = na.omit)
tempoDT[i,5]=proc.time()[3]-tempo0DT5

tempo0DT6=proc.time()[3]
DadostestRF6=na.omit(data.frame(dadostest$Y,dadostest$sex,dadostest$age))
RF6=randomForest(Y~sex+age,data = dadostrei,xtest=DadostestRF6[,-1],ytest=DadostestRF6$dadostest.Y,na.action = na.omit)
tempoDT[i,6]=proc.time()[3]-tempo0DT6

##########################################
#             CLASSIFICACAO
#     MAQUINA DE VETORES DE SUPORTE
##########################################

tempo0SVM1=proc.time()[3]
modSVM1=svm(Y~sex+htn,data=dadostrei,scale=F,kernel="radial",cost=1000,epsilon=1.0e-12, sigma=5)
YestSVM1test=predict(modSVM1,dadostest);YestSVM1trei=modSVM1$fitted
tempoSVM[i,1]=proc.time()[3]-tempo0SVM1

tempo0SVM2=proc.time()[3]
modSVM2=svm(Y~sex+age+htn+PAM+IPPA+HM+ALPHA,data=dadostrei,scale=F,kernel="poly",cost=100,epsilon=1.0e-12)
YestSVM2test=predict(modSVM2,dadostest);YestSVM2trei=modSVM2$fitted
tempoSVM[i,2]=proc.time()[3]-tempo0SVM2

tempo0SVM3=proc.time()[3]
modSVM3=svm(Y~sex+age+PAM+IPPARC,scale=F,data=dadostrei,kernel="poly",cost=100,epsilon=1.0e-12)
YestSVM3test=predict(modSVM3,dadostest);YestSVM3trei=modSVM3$fitted
tempoSVM[i,3]=proc.time()[3]-tempo0SVM3

tempo0SVM4=proc.time()[3]
modSVM4=svm(Y~sex+age+PAM,data=dadostrei,scale=F,kernel="poly",cost=100,epsilon=1.0e-12)
YestSVM4test=predict(modSVM4,dadostest);YestSVM4trei=modSVM4$fitted
tempoSVM[i,4]=proc.time()[3]-tempo0SVM4

tempo0SVM5=proc.time()[3]
modSVM5=svm(Y~sex+age+IPPA+IPPARC+ALPHA,data=dadostrei,scale=F,kernel="poly",cost=100,epsilon=1.0e-12)
YestSVM5test=predict(modSVM5,dadostest);YestSVM5trei=modSVM5$fitted
tempoSVM[i,5]=proc.time()[3]-tempo0SVM5

tempo0SVM6=proc.time()[3]
modSVM6=svm(Y~sex+age,scale=F,data=dadostrei,kernel="poly",cost=100,epsilon=1.0e-12)
YestSVM6test=predict(modSVM6,dadostest);YestSVM6trei=modSVM6$fitted
tempoSVM[i,6]=proc.time()[3]-tempo0SVM6

##########################################
#             CLASSIFICACAO
#     REGRESSAO LINEAR - LOGISTICA
##########################################

tempo0LR1=proc.time()[3]
modLR1=glm(Y~sex+htn,data=dadostrei,family = binomial(link = "logit"),na.action = na.omit)
YestLR1test=ifelse(predict.glm(modLR1,newdata = dadostest,type = "response")<0.5,0,1);YestLR1trei=ifelse(predict.glm(modLR1,newdata = dadostrei,type = "response")<0.5,0,1)
tempoLR[i,1]=proc.time()[3]-tempo0LR1

tempo0LR2=proc.time()[3]
modLR2=glm(Y~sex+age+htn+PAM+IPPA+HM+ALPHA,data=dadostrei,family = binomial(link = "logit"))
YestLR2test=ifelse(predict.glm(modLR2,newdata = dadostest,type = "response")<0.5,0,1);YestLR2trei=ifelse(predict.glm(modLR2,newdata = dadostrei,type = "response")<0.5,0,1)
tempoLR[i,2]=proc.time()[3]-tempo0LR2

tempo0LR3=proc.time()[3]
modLR3=glm(Y~sex+age+PAM+IPPARC,data=dadostrei,family = binomial(link = "logit"))
YestLR3test=ifelse(predict.glm(modLR3,newdata = dadostest,type = "response")<0.5,0,1);YestLR3trei=ifelse(predict.glm(modLR3,newdata = dadostrei,type = "response")<0.5,0,1)
tempoLR[i,3]=proc.time()[3]-tempo0LR3

tempo0LR4=proc.time()[3]
modLR4=glm(Y~sex+age+PAM,data=dadostrei,family = binomial(link = "logit"))
YestLR4test=ifelse(predict.glm(modLR4,newdata = dadostest,type = "response")<0.5,0,1);YestLR4trei=ifelse(predict.glm(modLR4,newdata = dadostrei,type = "response")<0.5,0,1)
tempoLR[i,4]=proc.time()[3]-tempo0LR4

tempo0LR5=proc.time()[3]
modLR5=glm(Y~sex+age+IPPA+IPPARC+ALPHA,data=dadostrei,family = binomial(link = "logit"))
YestLR5test=ifelse(predict.glm(modLR5,newdata = dadostest,type = "response")<0.5,0,1);YestLR5trei=ifelse(predict.glm(modLR5,newdata = dadostrei,type = "response")<0.5,0,1)
tempoLR[i,5]=proc.time()[3]-tempo0LR5

tempo0LR6=proc.time()[3]
modLR6=glm(Y~sex+age,family=binomial(link="logit"),data=dadostrei)
YestLR6test=ifelse(predict.glm(modLR6,newdata = dadostest,type = "response")<0.5,0,1);YestLR6trei=ifelse(predict.glm(modLR6,newdata = dadostrei,type = "response")<0.5,0,1)
tempoLR[i,6]=proc.time()[3]-tempo0LR6

##########################################
#             CLASSIFICACAO
#               ADABOOST
##########################################

tempo0ADB1=proc.time()[3]
modADB1=adaboost(Y~sex+htn,data=dadostrei,10)
YestADB1test=predict(modADB1,newdata=dadostest)$class;YestADB1trei=predict(modADB1,newdata=dadostrei)$class
tempoADB[i,1]=proc.time()[3]-tempo0ADB1

tempo0ADB2=proc.time()[3]
modADB2=adaboost(Y~sex+age+htn+PAM+IPPA+HM+ALPHA,data=dadostrei,10)
YestADB2test=predict(modADB2,newdata=dadostest)$class;YestADB2trei=predict(modADB2,newdata=dadostrei)$class
tempoADB[i,2]=proc.time()[3]-tempo0ADB2

tempo0ADB3=proc.time()[3]
modADB3=adaboost(Y~sex+age+PAM+IPPARC,data=dadostrei,10)
YestADB3test=predict(modADB3,newdata=dadostest)$class;YestADB3trei=predict(modADB3,newdata=dadostrei)$class
tempoADB[i,3]=proc.time()[3]-tempo0ADB3

tempo0ADB4=proc.time()[3]
modADB4=adaboost(Y~sex+age+PAM,data=dadostrei,10)
YestADB4test=predict(modADB4,newdata=dadostest)$class;YestADB4trei=predict(modADB4,newdata=dadostrei)$class
tempoADB[i,4]=proc.time()[3]-tempo0ADB4

tempo0ADB5=proc.time()[3]
modADB5=adaboost(Y~sex+age+IPPA+IPPARC+ALPHA,data=dadostrei,10)
YestADB5test=predict(modADB5,newdata=dadostest)$class;YestADB5trei=predict(modADB5,newdata=dadostrei)$class
tempoADB[i,5]=proc.time()[3]-tempo0ADB5

tempo0ADB6=proc.time()[3]
modADB6=adaboost(Y~sex+age,data=dadostrei,10)
YestADB6test=predict(modADB6,newdata=dadostest)$class;YestADB6trei=predict(modADB6,newdata=dadostrei)$class
tempoADB[i,6]=proc.time()[3]-tempo0ADB6


#############################
#         ACURACIAS         #
#############################

acurNBtrei[i,]=c((length(which(dadostrei$Y==0&YestNB1trei==0))+length(which(dadostrei$Y==1&YestNB1trei==1)))/length(dadostrei$Y)*100,(length(which(dadostrei$Y==0&YestNB2trei==0))+length(which(dadostrei$Y==1&YestNB2trei==1)))/length(dadostrei$Y)*100,(length(which(dadostrei$Y==0&YestNB3trei==0))+length(which(dadostrei$Y==1&YestNB3trei==1)))/length(dadostrei$Y)*100,(length(which(dadostrei$Y==0&YestNB4trei==0))+length(which(dadostrei$Y==1&YestNB4trei==1)))/length(dadostrei$Y)*100,(length(which(dadostrei$Y==0&YestNB5trei==0))+length(which(dadostrei$Y==1&YestNB5trei==1)))/length(dadostrei$Y)*100,(length(which(dadostrei$Y==0&YestNB6trei==0))+length(which(dadostrei$Y==1&YestNB6trei==1)))/length(dadostrei$Y)*100)
acurNBtest[i,]=c((length(which(dadostest$Y==0&YestNB1test==0))+length(which(dadostest$Y==1&YestNB1test==1)))/length(dadostest$Y)*100,(length(which(dadostest$Y==0&YestNB2test==0))+length(which(dadostest$Y==1&YestNB2test==1)))/length(dadostest$Y)*100,(length(which(dadostest$Y==0&YestNB3test==0))+length(which(dadostest$Y==1&YestNB3test==1)))/length(dadostest$Y)*100,(length(which(dadostest$Y==0&YestNB4test==0))+length(which(dadostest$Y==1&YestNB4test==1)))/length(dadostest$Y)*100,(length(which(dadostest$Y==0&YestNB5test==0))+length(which(dadostest$Y==1&YestNB5test==1)))/length(dadostest$Y)*100,(length(which(dadostest$Y==0&YestNB6test==0))+length(which(dadostest$Y==1&YestNB6test==1)))/length(dadostest$Y)*100)

acurDTtrei[i,]=c((RF1$confusion[1,1]+RF1$confusion[2,2])/sum(RF1$confusion[,-3])*100,(RF2$confusion[1,1]+RF2$confusion[2,2])/sum(RF2$confusion[,-3])*100,(RF3$confusion[1,1]+RF3$confusion[2,2])/sum(RF3$confusion[,-3])*100,(RF4$confusion[1,1]+RF4$confusion[2,2])/sum(RF4$confusion[,-3])*100,(RF5$confusion[1,1]+RF5$confusion[2,2])/sum(RF5$confusion[,-3])*100,(RF6$confusion[1,1]+RF6$confusion[2,2])/sum(RF6$confusion[,-3])*100)
acurDTtest[i,]=c((RF1$test$confusion[1,1]+RF1$test$confusion[2,2])/sum(RF1$test$confusion[,-3])*100,(RF2$test$confusion[1,1]+RF2$test$confusion[2,2])/sum(RF2$test$confusion[,-3])*100,(RF3$test$confusion[1,1]+RF3$test$confusion[2,2])/sum(RF3$test$confusion[,-3])*100,(RF4$test$confusion[1,1]+RF4$test$confusion[2,2])/sum(RF4$test$confusion[,-3])*100,(RF5$test$confusion[1,1]+RF5$test$confusion[2,2])/sum(RF5$test$confusion[,-3])*100,(RF6$test$confusion[1,1]+RF6$test$confusion[2,2])/sum(RF6$test$confusion[,-3])*100)

acurSVMtrei[i,]=c((length(which(Y[as.numeric(names(YestSVM1trei))]==0&YestSVM1trei==0))+length(which(Y[as.numeric(names(YestSVM1trei))]==1&YestSVM1trei==1)))/length(YestSVM1trei)*100,(length(which(Y[as.numeric(names(YestSVM2trei))]==0&YestSVM2trei==0))+length(which(Y[as.numeric(names(YestSVM2trei))]==1&YestSVM2trei==1)))/length(YestSVM2trei)*100,(length(which(Y[as.numeric(names(YestSVM3trei))]==0&YestSVM3trei==0))+length(which(Y[as.numeric(names(YestSVM3trei))]==1&YestSVM3trei==1)))/length(YestSVM3trei)*100,(length(which(Y[as.numeric(names(YestSVM4trei))]==0&YestSVM4trei==0))+length(which(Y[as.numeric(names(YestSVM4trei))]==1&YestSVM4trei==1)))/length(YestSVM4trei)*100,(length(which(Y[as.numeric(names(YestSVM5trei))]==0&YestSVM5trei==0))+length(which(Y[as.numeric(names(YestSVM5trei))]==1&YestSVM5trei==1)))/length(YestSVM5trei)*100,(length(which(Y[as.numeric(names(YestSVM6trei))]==0&YestSVM6trei==0))+length(which(Y[as.numeric(names(YestSVM6trei))]==1&YestSVM6trei==1)))/length(YestSVM6trei)*100)
acurSVMtest[i,]=c((length(which(Y[as.numeric(names(YestSVM1test))]==0&YestSVM1test==0))+length(which(Y[as.numeric(names(YestSVM1test))]==1&YestSVM1test==1)))/length(YestSVM1test)*100,(length(which(Y[as.numeric(names(YestSVM2test))]==0&YestSVM2test==0))+length(which(Y[as.numeric(names(YestSVM2test))]==1&YestSVM2test==1)))/length(YestSVM2test)*100,(length(which(Y[as.numeric(names(YestSVM3test))]==0&YestSVM3test==0))+length(which(Y[as.numeric(names(YestSVM3test))]==1&YestSVM3test==1)))/length(YestSVM3test)*100,(length(which(Y[as.numeric(names(YestSVM4test))]==0&YestSVM4test==0))+length(which(Y[as.numeric(names(YestSVM4test))]==1&YestSVM4test==1)))/length(YestSVM4test)*100,(length(which(Y[as.numeric(names(YestSVM5test))]==0&YestSVM5test==0))+length(which(Y[as.numeric(names(YestSVM5test))]==1&YestSVM5test==1)))/length(YestSVM5test)*100,(length(which(Y[as.numeric(names(YestSVM6test))]==0&YestSVM6test==0))+length(which(Y[as.numeric(names(YestSVM6test))]==1&YestSVM6test==1)))/length(YestSVM6test)*100)

acurLRtrei[i,]=c((length(which(Y[as.numeric(names(na.omit(YestLR1trei)))]==0&na.omit(YestLR1trei)==0))+length(which(Y[as.numeric(names(na.omit(YestLR1trei)))]==1&na.omit(YestLR1trei)==1)))/length(na.omit(YestLR1trei))*100,(length(which(Y[as.numeric(names(na.omit(YestLR2trei)))]==0&na.omit(YestLR2trei)==0))+length(which(Y[as.numeric(names(na.omit(YestLR2trei)))]==1&na.omit(YestLR2trei)==1)))/length(na.omit(YestLR2trei))*100,(length(which(Y[as.numeric(names(na.omit(YestLR3trei)))]==0&na.omit(YestLR3trei)==0))+length(which(Y[as.numeric(names(na.omit(YestLR3trei)))]==1&na.omit(YestLR3trei)==1)))/length(na.omit(YestLR3trei))*100,(length(which(Y[as.numeric(names(na.omit(YestLR4trei)))]==0&na.omit(YestLR4trei)==0))+length(which(Y[as.numeric(names(na.omit(YestLR4trei)))]==1&na.omit(YestLR4trei)==1)))/length(na.omit(YestLR4trei))*100,(length(which(Y[as.numeric(names(na.omit(YestLR5trei)))]==0&na.omit(YestLR5trei)==0))+length(which(Y[as.numeric(names(na.omit(YestLR5trei)))]==1&na.omit(YestLR5trei)==1)))/length(na.omit(YestLR5trei))*100,(length(which(Y[as.numeric(names(na.omit(YestLR6trei)))]==0&na.omit(YestLR6trei)==0))+length(which(Y[as.numeric(names(na.omit(YestLR6trei)))]==1&na.omit(YestLR6trei)==1)))/length(na.omit(YestLR6trei))*100)
acurLRtest[i,]=c((length(which(Y[as.numeric(names(na.omit(YestLR1test)))]==0&na.omit(YestLR1test)==0))+length(which(Y[as.numeric(names(na.omit(YestLR1test)))]==1&na.omit(YestLR1test)==1)))/length(na.omit(YestLR1test))*100,(length(which(Y[as.numeric(names(na.omit(YestLR2test)))]==0&na.omit(YestLR2test)==0))+length(which(Y[as.numeric(names(na.omit(YestLR2test)))]==1&na.omit(YestLR2test)==1)))/length(na.omit(YestLR2test))*100,(length(which(Y[as.numeric(names(na.omit(YestLR3test)))]==0&na.omit(YestLR3test)==0))+length(which(Y[as.numeric(names(na.omit(YestLR3test)))]==1&na.omit(YestLR3test)==1)))/length(na.omit(YestLR3test))*100,(length(which(Y[as.numeric(names(na.omit(YestLR4test)))]==0&na.omit(YestLR4test)==0))+length(which(Y[as.numeric(names(na.omit(YestLR4test)))]==1&na.omit(YestLR4test)==1)))/length(na.omit(YestLR4test))*100,(length(which(Y[as.numeric(names(na.omit(YestLR5test)))]==0&na.omit(YestLR5test)==0))+length(which(Y[as.numeric(names(na.omit(YestLR5test)))]==1&na.omit(YestLR5test)==1)))/length(na.omit(YestLR5test))*100,(length(which(Y[as.numeric(names(na.omit(YestLR6test)))]==0&na.omit(YestLR6test)==0))+length(which(Y[as.numeric(names(na.omit(YestLR6test)))]==1&na.omit(YestLR6test)==1)))/length(na.omit(YestLR6test))*100)

acurADBtrei[i,]=c((length(which(dadostrei$Y==0&YestADB1trei==0))+length(which(dadostrei$Y==1&YestADB1trei==1)))/length(dadostrei$Y)*100,(length(which(dadostrei$Y==0&YestADB2trei==0))+length(which(dadostrei$Y==1&YestADB2trei==1)))/length(dadostrei$Y)*100,(length(which(dadostrei$Y==0&YestADB3trei==0))+length(which(dadostrei$Y==1&YestADB3trei==1)))/length(dadostrei$Y)*100,(length(which(dadostrei$Y==0&YestADB4trei==0))+length(which(dadostrei$Y==1&YestADB4trei==1)))/length(dadostrei$Y)*100,(length(which(dadostrei$Y==0&YestADB5trei==0))+length(which(dadostrei$Y==1&YestADB5trei==1)))/length(dadostrei$Y)*100,(length(which(dadostrei$Y==0&YestADB6trei==0))+length(which(dadostrei$Y==1&YestADB6trei==1)))/length(dadostrei$Y)*100)
acurADBtest[i,]=c((length(which(dadostest$Y==0&YestADB1test==0))+length(which(dadostest$Y==1&YestADB1test==1)))/length(dadostest$Y)*100,(length(which(dadostest$Y==0&YestADB2test==0))+length(which(dadostest$Y==1&YestADB2test==1)))/length(dadostest$Y)*100,(length(which(dadostest$Y==0&YestADB3test==0))+length(which(dadostest$Y==1&YestADB3test==1)))/length(dadostest$Y)*100,(length(which(dadostest$Y==0&YestADB4test==0))+length(which(dadostest$Y==1&YestADB4test==1)))/length(dadostest$Y)*100,(length(which(dadostest$Y==0&YestADB5test==0))+length(which(dadostest$Y==1&YestADB5test==1)))/length(dadostest$Y)*100,(length(which(dadostest$Y==0&YestADB6test==0))+length(which(dadostest$Y==1&YestADB6test==1)))/length(dadostest$Y)*100)


matconftrei=matconftrei+matrix(c(
  length(which(dadostrei$Y==0&YestNB1trei==0)),length(which(dadostrei$Y==1&YestNB1trei==0)),RF1$confusion[1,1],RF1$confusion[2,1],length(which(Y[as.numeric(names(YestSVM1trei))]==0&YestSVM1trei==0)),length(which(Y[as.numeric(names(YestSVM1trei))]==1&YestSVM1trei==0)),length(which(Y[as.numeric(names(na.omit(YestLR1trei)))]==0&na.omit(YestLR1trei)==0)),length(which(Y[as.numeric(names(na.omit(YestLR1trei)))]==1&na.omit(YestLR1trei)==0)),length(which(dadostrei$Y==0&YestADB1trei==0)),length(which(dadostrei$Y==1&YestADB1trei==0)),
  length(which(dadostrei$Y==0&YestNB1trei==1)),length(which(dadostrei$Y==1&YestNB1trei==1)),RF1$confusion[1,2],RF1$confusion[2,2],length(which(Y[as.numeric(names(YestSVM1trei))]==0&YestSVM1trei==1)),length(which(Y[as.numeric(names(YestSVM1trei))]==1&YestSVM1trei==1)),length(which(Y[as.numeric(names(na.omit(YestLR1trei)))]==0&na.omit(YestLR1trei)==1)),length(which(Y[as.numeric(names(na.omit(YestLR1trei)))]==1&na.omit(YestLR1trei)==1)),length(which(dadostrei$Y==0&YestADB1trei==1)),length(which(dadostrei$Y==1&YestADB1trei==1)),
  length(which(dadostrei$Y==0&YestNB2trei==0)),length(which(dadostrei$Y==1&YestNB2trei==0)),RF2$confusion[1,1],RF2$confusion[2,1],length(which(Y[as.numeric(names(YestSVM2trei))]==0&YestSVM2trei==0)),length(which(Y[as.numeric(names(YestSVM2trei))]==1&YestSVM2trei==0)),length(which(Y[as.numeric(names(na.omit(YestLR2trei)))]==0&na.omit(YestLR2trei)==0)),length(which(Y[as.numeric(names(na.omit(YestLR2trei)))]==1&na.omit(YestLR2trei)==0)),length(which(dadostrei$Y==0&YestADB2trei==0)),length(which(dadostrei$Y==1&YestADB2trei==0)),
  length(which(dadostrei$Y==0&YestNB2trei==1)),length(which(dadostrei$Y==1&YestNB2trei==1)),RF2$confusion[1,2],RF2$confusion[2,2],length(which(Y[as.numeric(names(YestSVM2trei))]==0&YestSVM2trei==1)),length(which(Y[as.numeric(names(YestSVM2trei))]==1&YestSVM2trei==1)),length(which(Y[as.numeric(names(na.omit(YestLR2trei)))]==0&na.omit(YestLR2trei)==1)),length(which(Y[as.numeric(names(na.omit(YestLR2trei)))]==1&na.omit(YestLR2trei)==1)),length(which(dadostrei$Y==0&YestADB2trei==1)),length(which(dadostrei$Y==1&YestADB2trei==1)),
  length(which(dadostrei$Y==0&YestNB3trei==0)),length(which(dadostrei$Y==1&YestNB3trei==0)),RF3$confusion[1,1],RF3$confusion[2,1],length(which(Y[as.numeric(names(YestSVM3trei))]==0&YestSVM3trei==0)),length(which(Y[as.numeric(names(YestSVM3trei))]==1&YestSVM3trei==0)),length(which(Y[as.numeric(names(na.omit(YestLR3trei)))]==0&na.omit(YestLR3trei)==0)),length(which(Y[as.numeric(names(na.omit(YestLR3trei)))]==1&na.omit(YestLR3trei)==0)),length(which(dadostrei$Y==0&YestADB3trei==0)),length(which(dadostrei$Y==1&YestADB3trei==0)),
  length(which(dadostrei$Y==0&YestNB3trei==1)),length(which(dadostrei$Y==1&YestNB3trei==1)),RF3$confusion[1,2],RF3$confusion[2,2],length(which(Y[as.numeric(names(YestSVM3trei))]==0&YestSVM3trei==1)),length(which(Y[as.numeric(names(YestSVM3trei))]==1&YestSVM3trei==1)),length(which(Y[as.numeric(names(na.omit(YestLR3trei)))]==0&na.omit(YestLR3trei)==1)),length(which(Y[as.numeric(names(na.omit(YestLR3trei)))]==1&na.omit(YestLR3trei)==1)),length(which(dadostrei$Y==0&YestADB3trei==1)),length(which(dadostrei$Y==1&YestADB3trei==1)),
  length(which(dadostrei$Y==0&YestNB4trei==0)),length(which(dadostrei$Y==1&YestNB4trei==0)),RF4$confusion[1,1],RF4$confusion[2,1],length(which(Y[as.numeric(names(YestSVM4trei))]==0&YestSVM4trei==0)),length(which(Y[as.numeric(names(YestSVM4trei))]==1&YestSVM4trei==0)),length(which(Y[as.numeric(names(na.omit(YestLR4trei)))]==0&na.omit(YestLR4trei)==0)),length(which(Y[as.numeric(names(na.omit(YestLR4trei)))]==1&na.omit(YestLR4trei)==0)),length(which(dadostrei$Y==0&YestADB4trei==0)),length(which(dadostrei$Y==1&YestADB4trei==0)),
  length(which(dadostrei$Y==0&YestNB4trei==1)),length(which(dadostrei$Y==1&YestNB4trei==1)),RF4$confusion[1,2],RF4$confusion[2,2],length(which(Y[as.numeric(names(YestSVM4trei))]==0&YestSVM4trei==1)),length(which(Y[as.numeric(names(YestSVM4trei))]==1&YestSVM4trei==1)),length(which(Y[as.numeric(names(na.omit(YestLR4trei)))]==0&na.omit(YestLR4trei)==1)),length(which(Y[as.numeric(names(na.omit(YestLR4trei)))]==1&na.omit(YestLR4trei)==1)),length(which(dadostrei$Y==0&YestADB4trei==1)),length(which(dadostrei$Y==1&YestADB4trei==1)),
  length(which(dadostrei$Y==0&YestNB5trei==0)),length(which(dadostrei$Y==1&YestNB5trei==0)),RF5$confusion[1,1],RF5$confusion[2,1],length(which(Y[as.numeric(names(YestSVM5trei))]==0&YestSVM5trei==0)),length(which(Y[as.numeric(names(YestSVM5trei))]==1&YestSVM5trei==0)),length(which(Y[as.numeric(names(na.omit(YestLR5trei)))]==0&na.omit(YestLR5trei)==0)),length(which(Y[as.numeric(names(na.omit(YestLR5trei)))]==1&na.omit(YestLR5trei)==0)),length(which(dadostrei$Y==0&YestADB5trei==0)),length(which(dadostrei$Y==1&YestADB5trei==0)),
  length(which(dadostrei$Y==0&YestNB5trei==1)),length(which(dadostrei$Y==1&YestNB5trei==1)),RF5$confusion[1,2],RF5$confusion[2,2],length(which(Y[as.numeric(names(YestSVM5trei))]==0&YestSVM5trei==1)),length(which(Y[as.numeric(names(YestSVM5trei))]==1&YestSVM5trei==1)),length(which(Y[as.numeric(names(na.omit(YestLR5trei)))]==0&na.omit(YestLR5trei)==1)),length(which(Y[as.numeric(names(na.omit(YestLR5trei)))]==1&na.omit(YestLR5trei)==1)),length(which(dadostrei$Y==0&YestADB5trei==1)),length(which(dadostrei$Y==1&YestADB5trei==1)),
  length(which(dadostrei$Y==0&YestNB6trei==0)),length(which(dadostrei$Y==1&YestNB6trei==0)),RF6$confusion[1,1],RF6$confusion[2,1],length(which(Y[as.numeric(names(YestSVM6trei))]==0&YestSVM6trei==0)),length(which(Y[as.numeric(names(YestSVM6trei))]==1&YestSVM6trei==0)),length(which(Y[as.numeric(names(na.omit(YestLR6trei)))]==0&na.omit(YestLR6trei)==0)),length(which(Y[as.numeric(names(na.omit(YestLR6trei)))]==1&na.omit(YestLR6trei)==0)),length(which(dadostrei$Y==0&YestADB6trei==0)),length(which(dadostrei$Y==1&YestADB6trei==0)),
  length(which(dadostrei$Y==0&YestNB6trei==1)),length(which(dadostrei$Y==1&YestNB6trei==1)),RF6$confusion[1,2],RF6$confusion[2,2],length(which(Y[as.numeric(names(YestSVM6trei))]==0&YestSVM6trei==1)),length(which(Y[as.numeric(names(YestSVM6trei))]==1&YestSVM6trei==1)),length(which(Y[as.numeric(names(na.omit(YestLR6trei)))]==0&na.omit(YestLR6trei)==1)),length(which(Y[as.numeric(names(na.omit(YestLR6trei)))]==1&na.omit(YestLR6trei)==1)),length(which(dadostrei$Y==0&YestADB6trei==1)),length(which(dadostrei$Y==1&YestADB6trei==1))
),2*metodos,2*models)/B


matconftest=matconftest+matrix(c(
  length(which(dadostest$Y==0&YestNB1test==0)),length(which(dadostest$Y==1&YestNB1test==0)),RF1$test$confusion[1,1],RF1$test$confusion[2,1],length(which(Y[as.numeric(names(YestSVM1test))]==0&YestSVM1test==0)),length(which(Y[as.numeric(names(YestSVM1test))]==1&YestSVM1test==0)),length(which(Y[as.numeric(names(na.omit(YestLR1test)))]==0&na.omit(YestLR1test)==0)),length(which(Y[as.numeric(names(na.omit(YestLR1test)))]==1&na.omit(YestLR1test)==0)),length(which(dadostest$Y==0&YestADB1test==0)),length(which(dadostest$Y==1&YestADB1test==0)),
  length(which(dadostest$Y==0&YestNB1test==1)),length(which(dadostest$Y==1&YestNB1test==1)),RF1$test$confusion[1,2],RF1$test$confusion[2,2],length(which(Y[as.numeric(names(YestSVM1test))]==0&YestSVM1test==1)),length(which(Y[as.numeric(names(YestSVM1test))]==1&YestSVM1test==1)),length(which(Y[as.numeric(names(na.omit(YestLR1test)))]==0&na.omit(YestLR1test)==1)),length(which(Y[as.numeric(names(na.omit(YestLR1test)))]==1&na.omit(YestLR1test)==1)),length(which(dadostest$Y==0&YestADB1test==1)),length(which(dadostest$Y==1&YestADB1test==1)),
  length(which(dadostest$Y==0&YestNB2test==0)),length(which(dadostest$Y==1&YestNB2test==0)),RF2$test$confusion[1,1],RF2$test$confusion[2,1],length(which(Y[as.numeric(names(YestSVM2test))]==0&YestSVM2test==0)),length(which(Y[as.numeric(names(YestSVM2test))]==1&YestSVM2test==0)),length(which(Y[as.numeric(names(na.omit(YestLR2test)))]==0&na.omit(YestLR2test)==0)),length(which(Y[as.numeric(names(na.omit(YestLR2test)))]==1&na.omit(YestLR2test)==0)),length(which(dadostest$Y==0&YestADB2test==0)),length(which(dadostest$Y==1&YestADB2test==0)),
  length(which(dadostest$Y==0&YestNB2test==1)),length(which(dadostest$Y==1&YestNB2test==1)),RF2$test$confusion[1,2],RF2$test$confusion[2,2],length(which(Y[as.numeric(names(YestSVM2test))]==0&YestSVM2test==1)),length(which(Y[as.numeric(names(YestSVM2test))]==1&YestSVM2test==1)),length(which(Y[as.numeric(names(na.omit(YestLR2test)))]==0&na.omit(YestLR2test)==1)),length(which(Y[as.numeric(names(na.omit(YestLR2test)))]==1&na.omit(YestLR2test)==1)),length(which(dadostest$Y==0&YestADB2test==1)),length(which(dadostest$Y==1&YestADB2test==1)),
  length(which(dadostest$Y==0&YestNB3test==0)),length(which(dadostest$Y==1&YestNB3test==0)),RF3$test$confusion[1,1],RF3$test$confusion[2,1],length(which(Y[as.numeric(names(YestSVM3test))]==0&YestSVM3test==0)),length(which(Y[as.numeric(names(YestSVM3test))]==1&YestSVM3test==0)),length(which(Y[as.numeric(names(na.omit(YestLR3test)))]==0&na.omit(YestLR3test)==0)),length(which(Y[as.numeric(names(na.omit(YestLR3test)))]==1&na.omit(YestLR3test)==0)),length(which(dadostest$Y==0&YestADB3test==0)),length(which(dadostest$Y==1&YestADB3test==0)),
  length(which(dadostest$Y==0&YestNB3test==1)),length(which(dadostest$Y==1&YestNB3test==1)),RF3$test$confusion[1,2],RF3$test$confusion[2,2],length(which(Y[as.numeric(names(YestSVM3test))]==0&YestSVM3test==1)),length(which(Y[as.numeric(names(YestSVM3test))]==1&YestSVM3test==1)),length(which(Y[as.numeric(names(na.omit(YestLR3test)))]==0&na.omit(YestLR3test)==1)),length(which(Y[as.numeric(names(na.omit(YestLR3test)))]==1&na.omit(YestLR3test)==1)),length(which(dadostest$Y==0&YestADB3test==1)),length(which(dadostest$Y==1&YestADB3test==1)),
  length(which(dadostest$Y==0&YestNB4test==0)),length(which(dadostest$Y==1&YestNB4test==0)),RF4$test$confusion[1,1],RF4$test$confusion[2,1],length(which(Y[as.numeric(names(YestSVM4test))]==0&YestSVM4test==0)),length(which(Y[as.numeric(names(YestSVM4test))]==1&YestSVM4test==0)),length(which(Y[as.numeric(names(na.omit(YestLR4test)))]==0&na.omit(YestLR4test)==0)),length(which(Y[as.numeric(names(na.omit(YestLR4test)))]==1&na.omit(YestLR4test)==0)),length(which(dadostest$Y==0&YestADB4test==0)),length(which(dadostest$Y==1&YestADB4test==0)),
  length(which(dadostest$Y==0&YestNB4test==1)),length(which(dadostest$Y==1&YestNB4test==1)),RF4$test$confusion[1,2],RF4$test$confusion[2,2],length(which(Y[as.numeric(names(YestSVM4test))]==0&YestSVM4test==1)),length(which(Y[as.numeric(names(YestSVM4test))]==1&YestSVM4test==1)),length(which(Y[as.numeric(names(na.omit(YestLR4test)))]==0&na.omit(YestLR4test)==1)),length(which(Y[as.numeric(names(na.omit(YestLR4test)))]==1&na.omit(YestLR4test)==1)),length(which(dadostest$Y==0&YestADB4test==1)),length(which(dadostest$Y==1&YestADB4test==1)),
  length(which(dadostest$Y==0&YestNB5test==0)),length(which(dadostest$Y==1&YestNB5test==0)),RF5$test$confusion[1,1],RF5$test$confusion[2,1],length(which(Y[as.numeric(names(YestSVM5test))]==0&YestSVM5test==0)),length(which(Y[as.numeric(names(YestSVM5test))]==1&YestSVM5test==0)),length(which(Y[as.numeric(names(na.omit(YestLR5test)))]==0&na.omit(YestLR5test)==0)),length(which(Y[as.numeric(names(na.omit(YestLR5test)))]==1&na.omit(YestLR5test)==0)),length(which(dadostest$Y==0&YestADB5test==0)),length(which(dadostest$Y==1&YestADB5test==0)),
  length(which(dadostest$Y==0&YestNB5test==1)),length(which(dadostest$Y==1&YestNB5test==1)),RF5$test$confusion[1,2],RF5$test$confusion[2,2],length(which(Y[as.numeric(names(YestSVM5test))]==0&YestSVM5test==1)),length(which(Y[as.numeric(names(YestSVM5test))]==1&YestSVM5test==1)),length(which(Y[as.numeric(names(na.omit(YestLR5test)))]==0&na.omit(YestLR5test)==1)),length(which(Y[as.numeric(names(na.omit(YestLR5test)))]==1&na.omit(YestLR5test)==1)),length(which(dadostest$Y==0&YestADB5test==1)),length(which(dadostest$Y==1&YestADB5test==1)),
  length(which(dadostest$Y==0&YestNB6test==0)),length(which(dadostest$Y==1&YestNB6test==0)),RF6$test$confusion[1,1],RF6$test$confusion[2,1],length(which(Y[as.numeric(names(YestSVM6test))]==0&YestSVM6test==0)),length(which(Y[as.numeric(names(YestSVM6test))]==1&YestSVM6test==0)),length(which(Y[as.numeric(names(na.omit(YestLR6test)))]==0&na.omit(YestLR6test)==0)),length(which(Y[as.numeric(names(na.omit(YestLR6test)))]==1&na.omit(YestLR6test)==0)),length(which(dadostest$Y==0&YestADB6test==0)),length(which(dadostest$Y==1&YestADB6test==0)),
  length(which(dadostest$Y==0&YestNB6test==1)),length(which(dadostest$Y==1&YestNB6test==1)),RF6$test$confusion[1,2],RF6$test$confusion[2,2],length(which(Y[as.numeric(names(YestSVM6test))]==0&YestSVM6test==1)),length(which(Y[as.numeric(names(YestSVM6test))]==1&YestSVM6test==1)),length(which(Y[as.numeric(names(na.omit(YestLR6test)))]==0&na.omit(YestLR6test)==1)),length(which(Y[as.numeric(names(na.omit(YestLR6test)))]==1&na.omit(YestLR6test)==1)),length(which(dadostest$Y==0&YestADB6test==1)),length(which(dadostest$Y==1&YestADB6test==1))
),2*metodos,2*models)/B


k=k+1; print(k)
}


miacuraciastrei=rbind(apply(na.omit(acurNBtrei),2,mean),apply(na.omit(acurDTtrei),2,mean),apply(na.omit(acurSVMtrei),2,mean),apply(na.omit(acurLRtrei),2,mean),apply(na.omit(acurADBtrei),2,mean))
rownames(miacuraciastrei)=c("NB","DT","SVM","LR","Adaboost")

miacuraciastest=rbind(apply(na.omit(acurNBtest),2,mean),apply(na.omit(acurDTtest),2,mean),apply(na.omit(acurSVMtest),2,mean),apply(na.omit(acurLRtest),2,mean),apply(na.omit(acurADBtest),2,mean))
rownames(miacuraciastest)=c("NB","DT","SVM","LR","Adaboost")


sdacuraciastrei=rbind(apply(na.omit(acurNBtrei),2,sd),apply(na.omit(acurDTtrei),2,sd),apply(na.omit(acurSVMtrei),2,sd),apply(na.omit(acurLRtrei),2,sd),apply(na.omit(acurADBtrei),2,sd))
rownames(sdacuraciastrei)=c("NB","DT","SVM","LR","Adaboost")

sdacuraciastest=rbind(apply(na.omit(acurNBtest),2,sd),apply(na.omit(acurDTtest),2,sd),apply(na.omit(acurSVMtest),2,sd),apply(na.omit(acurLRtest),2,sd),apply(na.omit(acurADBtest),2,sd))
rownames(sdacuraciastest)=c("NB","DT","SVM","LR","Adaboost")

mitempo=rbind(apply(na.omit(tempoNB),2,mean),apply(na.omit(tempoDT),2,mean),apply(na.omit(tempoSVM),2,mean),apply(na.omit(tempoLR),2,mean),apply(na.omit(tempoADB),2,mean))
rownames(mitempo)=c("NB","DT","SVM","LR","Adaboost")

sdtempo=rbind(apply(na.omit(tempoNB),2,sd),apply(na.omit(tempoDT),2,sd),apply(na.omit(tempoSVM),2,sd),apply(na.omit(tempoLR),2,sd),apply(na.omit(tempoADB),2,sd))
rownames(sdtempo)=c("NB","DT","SVM","LR","Adaboost")

sdmatconftrei=matrix(0,2*metodos,2*models);colnames(sdmatconftrei)=rep(c(0,1),models);rownames(sdmatconftrei)=rep(c(0,1),metodos)
sdmatconftest=matrix(0,2*metodos,2*models);colnames(sdmatconftest)=rep(c(0,1),models);rownames(sdmatconftest)=rep(c(0,1),metodos)
j=0 #contador

set.seed(26071993)

for (i in 1:B) {
  
  treinamento=sample(nrow(newdata),round(nrow(newdata)*0.7))
  dadostrei=newdata[treinamento,]
  dadostest=newdata[-treinamento,]
  
  ##########################################
  #             CLASSIFICACAO
  #              NAIVE BAYES
  ##########################################
  
  modNB1=naiveBayes(Y~sex+htn,data = dadostrei)
  YestNB1trei=predict(modNB1,newdata = dadostrei);YestNB1test=predict(modNB1,newdata = dadostest)
  
  modNB2=naiveBayes(Y~sex+age+htn+PAM+IPPA+HM+ALPHA,data = dadostrei);YestNB2=predict(modNB2,newdata = dadostest)
  YestNB2trei=predict(modNB2,newdata = dadostrei);YestNB2test=predict(modNB2,newdata = dadostest)
  
  modNB3=naiveBayes(Y~sex+age+PAM+IPPARC,data = dadostrei) #### MELHOR
  YestNB3trei=predict(modNB3,newdata = dadostrei);YestNB3test=predict(modNB3,newdata = dadostest)
  
  modNB4=naiveBayes(Y~sex+age+PAM,data = dadostrei)
  YestNB4trei=predict(modNB4,newdata = dadostrei);YestNB4test=predict(modNB4,newdata = dadostest)
  
  modNB5=naiveBayes(Y~sex+age+IPPA+IPPARC+ALPHA,data = dadostrei)
  YestNB5trei=predict(modNB5,newdata = dadostrei);YestNB5test=predict(modNB5,newdata = dadostest)
  
  modNB6=naiveBayes(Y~sex+age,data = dadostrei)
  YestNB6trei=predict(modNB6,newdata = dadostrei);YestNB6test=predict(modNB6,newdata = dadostest)
  
  ##########################################
  #             CLASSIFICACAO
  #           ARVORE DE DECISAO
  ##########################################
  
  DadostestRF1=na.omit(data.frame(dadostest$Y,dadostest$sex,dadostest$htn))
  RF1=randomForest(Y~sex+htn,data = dadostrei,xtest=DadostestRF1[,-1],ytest=DadostestRF1$dadostest.Y,na.action = na.omit)
  
  DadostestRF2=na.omit(data.frame(dadostest$Y,dadostest$sex,dadostest$age,dadostest$htn,dadostest$PAM,dadostest$IPPA,dadostest$HM,dadostest$ALPHA))
  RF2=randomForest(Y~sex+age+htn+PAM+IPPA+HM+ALPHA,data = dadostrei,xtest=DadostestRF2[,-1],ytest=DadostestRF2$dadostest.Y,na.action = na.omit)
  
  DadostestRF3=na.omit(data.frame(dadostest$Y,dadostest$sex,dadostest$age,dadostest$PAM,dadostest$IPPARC))
  RF3=randomForest(Y~sex+age+PAM+IPPARC,data = dadostrei,xtest=DadostestRF3[,-1],ytest=DadostestRF3$dadostest.Y,na.action = na.omit)
  
  DadostestRF4=na.omit(data.frame(dadostest$Y,dadostest$sex,dadostest$age,dadostest$PAM))
  RF4=randomForest(Y~sex+age+PAM,data = dadostrei,xtest=DadostestRF4[,-1],ytest=DadostestRF4$dadostest.Y,na.action = na.omit)
  
  DadostestRF5=na.omit(data.frame(dadostest$Y,dadostest$sex,dadostest$age,dadostest$IPPA,dadostest$IPPARC,dadostest$ALPHA))
  RF5=randomForest(Y~sex+age+IPPA+IPPARC+ALPHA,data = dadostrei,xtest=DadostestRF5[,-1],ytest=DadostestRF5$dadostest.Y,na.action = na.omit)
  
  DadostestRF6=na.omit(data.frame(dadostest$Y,dadostest$sex,dadostest$age))
  RF6=randomForest(Y~sex+age,data = dadostrei,xtest=DadostestRF6[,-1],ytest=DadostestRF6$dadostest.Y,na.action = na.omit)
  
  ##########################################
  #             CLASSIFICACAO
  #     MAQUINA DE VETORES DE SUPORTE
  ##########################################
  
  modSVM1=svm(Y~sex+htn,data=dadostrei,scale=F,kernel="radial",cost=1000,epsilon=1.0e-12, sigma=5)
  YestSVM1test=predict(modSVM1,dadostest);YestSVM1trei=modSVM1$fitted
  
  modSVM2=svm(Y~sex+age+htn+PAM+IPPA+HM+ALPHA,data=dadostrei,scale=F,kernel="poly",cost=100,epsilon=1.0e-12)
  YestSVM2test=predict(modSVM2,dadostest);YestSVM2trei=modSVM2$fitted
  
  modSVM3=svm(Y~sex+age+PAM+IPPARC,scale=F,data=dadostrei,kernel="poly",cost=100,epsilon=1.0e-12)
  YestSVM3test=predict(modSVM3,dadostest);YestSVM3trei=modSVM3$fitted
  
  modSVM4=svm(Y~sex+age+PAM,data=dadostrei,scale=F,kernel="poly",cost=100,epsilon=1.0e-12)
  YestSVM4test=predict(modSVM4,dadostest);YestSVM4trei=modSVM4$fitted
  
  modSVM5=svm(Y~sex+age+IPPA+IPPARC+ALPHA,data=dadostrei,scale=F,kernel="poly",cost=100,epsilon=1.0e-12)
  YestSVM5test=predict(modSVM5,dadostest);YestSVM5trei=modSVM5$fitted
  
  modSVM6=svm(Y~sex+age,scale=F,data=dadostrei,kernel="poly",cost=100,epsilon=1.0e-12)
  YestSVM6test=predict(modSVM6,dadostest);YestSVM6trei=modSVM6$fitted
  
  ##########################################
  #             CLASSIFICACAO
  #     REGRESSAO LINEAR - LOGISTICA
  ##########################################
  
  modLR1=glm(Y~sex+htn,data=dadostrei,family = binomial(link = "logit"),na.action = na.omit)
  YestLR1test=ifelse(predict.glm(modLR1,newdata = dadostest,type = "response")<0.5,0,1);YestLR1trei=ifelse(predict.glm(modLR1,newdata = dadostrei,type = "response")<0.5,0,1)
  
  modLR2=glm(Y~sex+age+htn+PAM+IPPA+HM+ALPHA,data=dadostrei,family = binomial(link = "logit"))
  YestLR2test=ifelse(predict.glm(modLR2,newdata = dadostest,type = "response")<0.5,0,1);YestLR2trei=ifelse(predict.glm(modLR2,newdata = dadostrei,type = "response")<0.5,0,1)
  
  modLR3=glm(Y~sex+age+PAM+IPPARC,data=dadostrei,family = binomial(link = "logit"))
  YestLR3test=ifelse(predict.glm(modLR3,newdata = dadostest,type = "response")<0.5,0,1);YestLR3trei=ifelse(predict.glm(modLR3,newdata = dadostrei,type = "response")<0.5,0,1)
  
  modLR4=glm(Y~sex+age+PAM,data=dadostrei,family = binomial(link = "logit"))
  YestLR4test=ifelse(predict.glm(modLR4,newdata = dadostest,type = "response")<0.5,0,1);YestLR4trei=ifelse(predict.glm(modLR4,newdata = dadostrei,type = "response")<0.5,0,1)
  
  modLR5=glm(Y~sex+age+IPPA+IPPARC+ALPHA,data=dadostrei,family = binomial(link = "logit"))
  YestLR5test=ifelse(predict.glm(modLR5,newdata = dadostest,type = "response")<0.5,0,1);YestLR5trei=ifelse(predict.glm(modLR5,newdata = dadostrei,type = "response")<0.5,0,1)
  
  modLR6=glm(Y~sex+age,family=binomial(link="logit"),data=dadostrei)
  YestLR6test=ifelse(predict.glm(modLR6,newdata = dadostest,type = "response")<0.5,0,1);YestLR6trei=ifelse(predict.glm(modLR6,newdata = dadostrei,type = "response")<0.5,0,1)
  
  ##########################################
  #             CLASSIFICACAO
  #               ADABOOST
  ##########################################
  
  modADB1=adaboost(Y~sex+htn,data=dadostrei,10)
  YestADB1test=predict(modADB1,newdata=dadostest)$class;YestADB1trei=predict(modADB1,newdata=dadostrei)$class
  
  modADB2=adaboost(Y~sex+age+htn+PAM+IPPA+HM+ALPHA,data=dadostrei,10)
  YestADB2test=predict(modADB2,newdata=dadostest)$class;YestADB2trei=predict(modADB2,newdata=dadostrei)$class
  
  modADB3=adaboost(Y~sex+age+PAM+IPPARC,data=dadostrei,10)
  YestADB3test=predict(modADB3,newdata=dadostest)$class;YestADB3trei=predict(modADB3,newdata=dadostrei)$class
  
  modADB4=adaboost(Y~sex+age+PAM,data=dadostrei,10)
  YestADB4test=predict(modADB4,newdata=dadostest)$class;YestADB4trei=predict(modADB4,newdata=dadostrei)$class
  
  modADB5=adaboost(Y~sex+age+IPPA+IPPARC+ALPHA,data=dadostrei,10)
  YestADB5test=predict(modADB5,newdata=dadostest)$class;YestADB5trei=predict(modADB5,newdata=dadostrei)$class
  
  modADB6=adaboost(Y~sex+age,data=dadostrei,10)
  YestADB6test=predict(modADB6,newdata=dadostest)$class;YestADB6trei=predict(modADB6,newdata=dadostrei)$class
  
  
  #############################
  #         ACURACIAS         #
  #############################
  
  acurNBtrei[i,]=c((length(which(dadostrei$Y==0&YestNB1trei==0))+length(which(dadostrei$Y==1&YestNB1trei==1)))/length(dadostrei$Y)*100,(length(which(dadostrei$Y==0&YestNB2trei==0))+length(which(dadostrei$Y==1&YestNB2trei==1)))/length(dadostrei$Y)*100,(length(which(dadostrei$Y==0&YestNB3trei==0))+length(which(dadostrei$Y==1&YestNB3trei==1)))/length(dadostrei$Y)*100,(length(which(dadostrei$Y==0&YestNB4trei==0))+length(which(dadostrei$Y==1&YestNB4trei==1)))/length(dadostrei$Y)*100,(length(which(dadostrei$Y==0&YestNB5trei==0))+length(which(dadostrei$Y==1&YestNB5trei==1)))/length(dadostrei$Y)*100,(length(which(dadostrei$Y==0&YestNB6trei==0))+length(which(dadostrei$Y==1&YestNB6trei==1)))/length(dadostrei$Y)*100)
  acurNBtest[i,]=c((length(which(dadostest$Y==0&YestNB1test==0))+length(which(dadostest$Y==1&YestNB1test==1)))/length(dadostest$Y)*100,(length(which(dadostest$Y==0&YestNB2test==0))+length(which(dadostest$Y==1&YestNB2test==1)))/length(dadostest$Y)*100,(length(which(dadostest$Y==0&YestNB3test==0))+length(which(dadostest$Y==1&YestNB3test==1)))/length(dadostest$Y)*100,(length(which(dadostest$Y==0&YestNB4test==0))+length(which(dadostest$Y==1&YestNB4test==1)))/length(dadostest$Y)*100,(length(which(dadostest$Y==0&YestNB5test==0))+length(which(dadostest$Y==1&YestNB5test==1)))/length(dadostest$Y)*100,(length(which(dadostest$Y==0&YestNB6test==0))+length(which(dadostest$Y==1&YestNB6test==1)))/length(dadostest$Y)*100)
  
  acurDTtrei[i,]=c((RF1$confusion[1,1]+RF1$confusion[2,2])/sum(RF1$confusion[,-3])*100,(RF2$confusion[1,1]+RF2$confusion[2,2])/sum(RF2$confusion[,-3])*100,(RF3$confusion[1,1]+RF3$confusion[2,2])/sum(RF3$confusion[,-3])*100,(RF4$confusion[1,1]+RF4$confusion[2,2])/sum(RF4$confusion[,-3])*100,(RF5$confusion[1,1]+RF5$confusion[2,2])/sum(RF5$confusion[,-3])*100,(RF6$confusion[1,1]+RF6$confusion[2,2])/sum(RF6$confusion[,-3])*100)
  acurDTtest[i,]=c((RF1$test$confusion[1,1]+RF1$test$confusion[2,2])/sum(RF1$test$confusion[,-3])*100,(RF2$test$confusion[1,1]+RF2$test$confusion[2,2])/sum(RF2$test$confusion[,-3])*100,(RF3$test$confusion[1,1]+RF3$test$confusion[2,2])/sum(RF3$test$confusion[,-3])*100,(RF4$test$confusion[1,1]+RF4$test$confusion[2,2])/sum(RF4$test$confusion[,-3])*100,(RF5$test$confusion[1,1]+RF5$test$confusion[2,2])/sum(RF5$test$confusion[,-3])*100,(RF6$test$confusion[1,1]+RF6$test$confusion[2,2])/sum(RF6$test$confusion[,-3])*100)
  
  acurSVMtrei[i,]=c((length(which(Y[as.numeric(names(YestSVM1trei))]==0&YestSVM1trei==0))+length(which(Y[as.numeric(names(YestSVM1trei))]==1&YestSVM1trei==1)))/length(YestSVM1trei)*100,(length(which(Y[as.numeric(names(YestSVM2trei))]==0&YestSVM2trei==0))+length(which(Y[as.numeric(names(YestSVM2trei))]==1&YestSVM2trei==1)))/length(YestSVM2trei)*100,(length(which(Y[as.numeric(names(YestSVM3trei))]==0&YestSVM3trei==0))+length(which(Y[as.numeric(names(YestSVM3trei))]==1&YestSVM3trei==1)))/length(YestSVM3trei)*100,(length(which(Y[as.numeric(names(YestSVM4trei))]==0&YestSVM4trei==0))+length(which(Y[as.numeric(names(YestSVM4trei))]==1&YestSVM4trei==1)))/length(YestSVM4trei)*100,(length(which(Y[as.numeric(names(YestSVM5trei))]==0&YestSVM5trei==0))+length(which(Y[as.numeric(names(YestSVM5trei))]==1&YestSVM5trei==1)))/length(YestSVM5trei)*100,(length(which(Y[as.numeric(names(YestSVM6trei))]==0&YestSVM6trei==0))+length(which(Y[as.numeric(names(YestSVM6trei))]==1&YestSVM6trei==1)))/length(YestSVM6trei)*100)
  acurSVMtest[i,]=c((length(which(Y[as.numeric(names(YestSVM1test))]==0&YestSVM1test==0))+length(which(Y[as.numeric(names(YestSVM1test))]==1&YestSVM1test==1)))/length(YestSVM1test)*100,(length(which(Y[as.numeric(names(YestSVM2test))]==0&YestSVM2test==0))+length(which(Y[as.numeric(names(YestSVM2test))]==1&YestSVM2test==1)))/length(YestSVM2test)*100,(length(which(Y[as.numeric(names(YestSVM3test))]==0&YestSVM3test==0))+length(which(Y[as.numeric(names(YestSVM3test))]==1&YestSVM3test==1)))/length(YestSVM3test)*100,(length(which(Y[as.numeric(names(YestSVM4test))]==0&YestSVM4test==0))+length(which(Y[as.numeric(names(YestSVM4test))]==1&YestSVM4test==1)))/length(YestSVM4test)*100,(length(which(Y[as.numeric(names(YestSVM5test))]==0&YestSVM5test==0))+length(which(Y[as.numeric(names(YestSVM5test))]==1&YestSVM5test==1)))/length(YestSVM5test)*100,(length(which(Y[as.numeric(names(YestSVM6test))]==0&YestSVM6test==0))+length(which(Y[as.numeric(names(YestSVM6test))]==1&YestSVM6test==1)))/length(YestSVM6test)*100)
  
  acurLRtrei[i,]=c((length(which(Y[as.numeric(names(na.omit(YestLR1trei)))]==0&na.omit(YestLR1trei)==0))+length(which(Y[as.numeric(names(na.omit(YestLR1trei)))]==1&na.omit(YestLR1trei)==1)))/length(na.omit(YestLR1trei))*100,(length(which(Y[as.numeric(names(na.omit(YestLR2trei)))]==0&na.omit(YestLR2trei)==0))+length(which(Y[as.numeric(names(na.omit(YestLR2trei)))]==1&na.omit(YestLR2trei)==1)))/length(na.omit(YestLR2trei))*100,(length(which(Y[as.numeric(names(na.omit(YestLR3trei)))]==0&na.omit(YestLR3trei)==0))+length(which(Y[as.numeric(names(na.omit(YestLR3trei)))]==1&na.omit(YestLR3trei)==1)))/length(na.omit(YestLR3trei))*100,(length(which(Y[as.numeric(names(na.omit(YestLR4trei)))]==0&na.omit(YestLR4trei)==0))+length(which(Y[as.numeric(names(na.omit(YestLR4trei)))]==1&na.omit(YestLR4trei)==1)))/length(na.omit(YestLR4trei))*100,(length(which(Y[as.numeric(names(na.omit(YestLR5trei)))]==0&na.omit(YestLR5trei)==0))+length(which(Y[as.numeric(names(na.omit(YestLR5trei)))]==1&na.omit(YestLR5trei)==1)))/length(na.omit(YestLR5trei))*100,(length(which(Y[as.numeric(names(na.omit(YestLR6trei)))]==0&na.omit(YestLR6trei)==0))+length(which(Y[as.numeric(names(na.omit(YestLR6trei)))]==1&na.omit(YestLR6trei)==1)))/length(na.omit(YestLR6trei))*100)
  acurLRtest[i,]=c((length(which(Y[as.numeric(names(na.omit(YestLR1test)))]==0&na.omit(YestLR1test)==0))+length(which(Y[as.numeric(names(na.omit(YestLR1test)))]==1&na.omit(YestLR1test)==1)))/length(na.omit(YestLR1test))*100,(length(which(Y[as.numeric(names(na.omit(YestLR2test)))]==0&na.omit(YestLR2test)==0))+length(which(Y[as.numeric(names(na.omit(YestLR2test)))]==1&na.omit(YestLR2test)==1)))/length(na.omit(YestLR2test))*100,(length(which(Y[as.numeric(names(na.omit(YestLR3test)))]==0&na.omit(YestLR3test)==0))+length(which(Y[as.numeric(names(na.omit(YestLR3test)))]==1&na.omit(YestLR3test)==1)))/length(na.omit(YestLR3test))*100,(length(which(Y[as.numeric(names(na.omit(YestLR4test)))]==0&na.omit(YestLR4test)==0))+length(which(Y[as.numeric(names(na.omit(YestLR4test)))]==1&na.omit(YestLR4test)==1)))/length(na.omit(YestLR4test))*100,(length(which(Y[as.numeric(names(na.omit(YestLR5test)))]==0&na.omit(YestLR5test)==0))+length(which(Y[as.numeric(names(na.omit(YestLR5test)))]==1&na.omit(YestLR5test)==1)))/length(na.omit(YestLR5test))*100,(length(which(Y[as.numeric(names(na.omit(YestLR6test)))]==0&na.omit(YestLR6test)==0))+length(which(Y[as.numeric(names(na.omit(YestLR6test)))]==1&na.omit(YestLR6test)==1)))/length(na.omit(YestLR6test))*100)
  
  acurADBtrei[i,]=c((length(which(dadostrei$Y==0&YestADB1trei==0))+length(which(dadostrei$Y==1&YestADB1trei==1)))/length(dadostrei$Y)*100,(length(which(dadostrei$Y==0&YestADB2trei==0))+length(which(dadostrei$Y==1&YestADB2trei==1)))/length(dadostrei$Y)*100,(length(which(dadostrei$Y==0&YestADB3trei==0))+length(which(dadostrei$Y==1&YestADB3trei==1)))/length(dadostrei$Y)*100,(length(which(dadostrei$Y==0&YestADB4trei==0))+length(which(dadostrei$Y==1&YestADB4trei==1)))/length(dadostrei$Y)*100,(length(which(dadostrei$Y==0&YestADB5trei==0))+length(which(dadostrei$Y==1&YestADB5trei==1)))/length(dadostrei$Y)*100,(length(which(dadostrei$Y==0&YestADB6trei==0))+length(which(dadostrei$Y==1&YestADB6trei==1)))/length(dadostrei$Y)*100)
  acurADBtest[i,]=c((length(which(dadostest$Y==0&YestADB1test==0))+length(which(dadostest$Y==1&YestADB1test==1)))/length(dadostest$Y)*100,(length(which(dadostest$Y==0&YestADB2test==0))+length(which(dadostest$Y==1&YestADB2test==1)))/length(dadostest$Y)*100,(length(which(dadostest$Y==0&YestADB3test==0))+length(which(dadostest$Y==1&YestADB3test==1)))/length(dadostest$Y)*100,(length(which(dadostest$Y==0&YestADB4test==0))+length(which(dadostest$Y==1&YestADB4test==1)))/length(dadostest$Y)*100,(length(which(dadostest$Y==0&YestADB5test==0))+length(which(dadostest$Y==1&YestADB5test==1)))/length(dadostest$Y)*100,(length(which(dadostest$Y==0&YestADB6test==0))+length(which(dadostest$Y==1&YestADB6test==1)))/length(dadostest$Y)*100)
  
  sdmatconftrei=sdmatconftrei+((matrix(c(
    length(which(dadostrei$Y==0&YestNB1trei==0)),length(which(dadostrei$Y==1&YestNB1trei==0)),RF1$confusion[1,1],RF1$confusion[2,1],length(which(Y[as.numeric(names(YestSVM1trei))]==0&YestSVM1trei==0)),length(which(Y[as.numeric(names(YestSVM1trei))]==1&YestSVM1trei==0)),length(which(Y[as.numeric(names(na.omit(YestLR1trei)))]==0&na.omit(YestLR1trei)==0)),length(which(Y[as.numeric(names(na.omit(YestLR1trei)))]==1&na.omit(YestLR1trei)==0)),length(which(dadostrei$Y==0&YestADB1trei==0)),length(which(dadostrei$Y==1&YestADB1trei==0)),
    length(which(dadostrei$Y==0&YestNB1trei==1)),length(which(dadostrei$Y==1&YestNB1trei==1)),RF1$confusion[1,2],RF1$confusion[2,2],length(which(Y[as.numeric(names(YestSVM1trei))]==0&YestSVM1trei==1)),length(which(Y[as.numeric(names(YestSVM1trei))]==1&YestSVM1trei==1)),length(which(Y[as.numeric(names(na.omit(YestLR1trei)))]==0&na.omit(YestLR1trei)==1)),length(which(Y[as.numeric(names(na.omit(YestLR1trei)))]==1&na.omit(YestLR1trei)==1)),length(which(dadostrei$Y==0&YestADB1trei==1)),length(which(dadostrei$Y==1&YestADB1trei==1)),
    length(which(dadostrei$Y==0&YestNB2trei==0)),length(which(dadostrei$Y==1&YestNB2trei==0)),RF2$confusion[1,1],RF2$confusion[2,1],length(which(Y[as.numeric(names(YestSVM2trei))]==0&YestSVM2trei==0)),length(which(Y[as.numeric(names(YestSVM2trei))]==1&YestSVM2trei==0)),length(which(Y[as.numeric(names(na.omit(YestLR2trei)))]==0&na.omit(YestLR2trei)==0)),length(which(Y[as.numeric(names(na.omit(YestLR2trei)))]==1&na.omit(YestLR2trei)==0)),length(which(dadostrei$Y==0&YestADB2trei==0)),length(which(dadostrei$Y==1&YestADB2trei==0)),
    length(which(dadostrei$Y==0&YestNB2trei==1)),length(which(dadostrei$Y==1&YestNB2trei==1)),RF2$confusion[1,2],RF2$confusion[2,2],length(which(Y[as.numeric(names(YestSVM2trei))]==0&YestSVM2trei==1)),length(which(Y[as.numeric(names(YestSVM2trei))]==1&YestSVM2trei==1)),length(which(Y[as.numeric(names(na.omit(YestLR2trei)))]==0&na.omit(YestLR2trei)==1)),length(which(Y[as.numeric(names(na.omit(YestLR2trei)))]==1&na.omit(YestLR2trei)==1)),length(which(dadostrei$Y==0&YestADB2trei==1)),length(which(dadostrei$Y==1&YestADB2trei==1)),
    length(which(dadostrei$Y==0&YestNB3trei==0)),length(which(dadostrei$Y==1&YestNB3trei==0)),RF3$confusion[1,1],RF3$confusion[2,1],length(which(Y[as.numeric(names(YestSVM3trei))]==0&YestSVM3trei==0)),length(which(Y[as.numeric(names(YestSVM3trei))]==1&YestSVM3trei==0)),length(which(Y[as.numeric(names(na.omit(YestLR3trei)))]==0&na.omit(YestLR3trei)==0)),length(which(Y[as.numeric(names(na.omit(YestLR3trei)))]==1&na.omit(YestLR3trei)==0)),length(which(dadostrei$Y==0&YestADB3trei==0)),length(which(dadostrei$Y==1&YestADB3trei==0)),
    length(which(dadostrei$Y==0&YestNB3trei==1)),length(which(dadostrei$Y==1&YestNB3trei==1)),RF3$confusion[1,2],RF3$confusion[2,2],length(which(Y[as.numeric(names(YestSVM3trei))]==0&YestSVM3trei==1)),length(which(Y[as.numeric(names(YestSVM3trei))]==1&YestSVM3trei==1)),length(which(Y[as.numeric(names(na.omit(YestLR3trei)))]==0&na.omit(YestLR3trei)==1)),length(which(Y[as.numeric(names(na.omit(YestLR3trei)))]==1&na.omit(YestLR3trei)==1)),length(which(dadostrei$Y==0&YestADB3trei==1)),length(which(dadostrei$Y==1&YestADB3trei==1)),
    length(which(dadostrei$Y==0&YestNB4trei==0)),length(which(dadostrei$Y==1&YestNB4trei==0)),RF4$confusion[1,1],RF4$confusion[2,1],length(which(Y[as.numeric(names(YestSVM4trei))]==0&YestSVM4trei==0)),length(which(Y[as.numeric(names(YestSVM4trei))]==1&YestSVM4trei==0)),length(which(Y[as.numeric(names(na.omit(YestLR4trei)))]==0&na.omit(YestLR4trei)==0)),length(which(Y[as.numeric(names(na.omit(YestLR4trei)))]==1&na.omit(YestLR4trei)==0)),length(which(dadostrei$Y==0&YestADB4trei==0)),length(which(dadostrei$Y==1&YestADB4trei==0)),
    length(which(dadostrei$Y==0&YestNB4trei==1)),length(which(dadostrei$Y==1&YestNB4trei==1)),RF4$confusion[1,2],RF4$confusion[2,2],length(which(Y[as.numeric(names(YestSVM4trei))]==0&YestSVM4trei==1)),length(which(Y[as.numeric(names(YestSVM4trei))]==1&YestSVM4trei==1)),length(which(Y[as.numeric(names(na.omit(YestLR4trei)))]==0&na.omit(YestLR4trei)==1)),length(which(Y[as.numeric(names(na.omit(YestLR4trei)))]==1&na.omit(YestLR4trei)==1)),length(which(dadostrei$Y==0&YestADB4trei==1)),length(which(dadostrei$Y==1&YestADB4trei==1)),
    length(which(dadostrei$Y==0&YestNB5trei==0)),length(which(dadostrei$Y==1&YestNB5trei==0)),RF5$confusion[1,1],RF5$confusion[2,1],length(which(Y[as.numeric(names(YestSVM5trei))]==0&YestSVM5trei==0)),length(which(Y[as.numeric(names(YestSVM5trei))]==1&YestSVM5trei==0)),length(which(Y[as.numeric(names(na.omit(YestLR5trei)))]==0&na.omit(YestLR5trei)==0)),length(which(Y[as.numeric(names(na.omit(YestLR5trei)))]==1&na.omit(YestLR5trei)==0)),length(which(dadostrei$Y==0&YestADB5trei==0)),length(which(dadostrei$Y==1&YestADB5trei==0)),
    length(which(dadostrei$Y==0&YestNB5trei==1)),length(which(dadostrei$Y==1&YestNB5trei==1)),RF5$confusion[1,2],RF5$confusion[2,2],length(which(Y[as.numeric(names(YestSVM5trei))]==0&YestSVM5trei==1)),length(which(Y[as.numeric(names(YestSVM5trei))]==1&YestSVM5trei==1)),length(which(Y[as.numeric(names(na.omit(YestLR5trei)))]==0&na.omit(YestLR5trei)==1)),length(which(Y[as.numeric(names(na.omit(YestLR5trei)))]==1&na.omit(YestLR5trei)==1)),length(which(dadostrei$Y==0&YestADB5trei==1)),length(which(dadostrei$Y==1&YestADB5trei==1)),
    length(which(dadostrei$Y==0&YestNB6trei==0)),length(which(dadostrei$Y==1&YestNB6trei==0)),RF6$confusion[1,1],RF6$confusion[2,1],length(which(Y[as.numeric(names(YestSVM6trei))]==0&YestSVM6trei==0)),length(which(Y[as.numeric(names(YestSVM6trei))]==1&YestSVM6trei==0)),length(which(Y[as.numeric(names(na.omit(YestLR6trei)))]==0&na.omit(YestLR6trei)==0)),length(which(Y[as.numeric(names(na.omit(YestLR6trei)))]==1&na.omit(YestLR6trei)==0)),length(which(dadostrei$Y==0&YestADB6trei==0)),length(which(dadostrei$Y==1&YestADB6trei==0)),
    length(which(dadostrei$Y==0&YestNB6trei==1)),length(which(dadostrei$Y==1&YestNB6trei==1)),RF6$confusion[1,2],RF6$confusion[2,2],length(which(Y[as.numeric(names(YestSVM6trei))]==0&YestSVM6trei==1)),length(which(Y[as.numeric(names(YestSVM6trei))]==1&YestSVM6trei==1)),length(which(Y[as.numeric(names(na.omit(YestLR6trei)))]==0&na.omit(YestLR6trei)==1)),length(which(Y[as.numeric(names(na.omit(YestLR6trei)))]==1&na.omit(YestLR6trei)==1)),length(which(dadostrei$Y==0&YestADB6trei==1)),length(which(dadostrei$Y==1&YestADB6trei==1))
  ),2*metodos,2*models)-matconftrei)^2)/(B-1)
  
  
  sdmatconftest=sdmatconftest+((matrix(c(
    length(which(dadostest$Y==0&YestNB1test==0)),length(which(dadostest$Y==1&YestNB1test==0)),RF1$test$confusion[1,1],RF1$test$confusion[2,1],length(which(Y[as.numeric(names(YestSVM1test))]==0&YestSVM1test==0)),length(which(Y[as.numeric(names(YestSVM1test))]==1&YestSVM1test==0)),length(which(Y[as.numeric(names(na.omit(YestLR1test)))]==0&na.omit(YestLR1test)==0)),length(which(Y[as.numeric(names(na.omit(YestLR1test)))]==1&na.omit(YestLR1test)==0)),length(which(dadostest$Y==0&YestADB1test==0)),length(which(dadostest$Y==1&YestADB1test==0)),
    length(which(dadostest$Y==0&YestNB1test==1)),length(which(dadostest$Y==1&YestNB1test==1)),RF1$test$confusion[1,2],RF1$test$confusion[2,2],length(which(Y[as.numeric(names(YestSVM1test))]==0&YestSVM1test==1)),length(which(Y[as.numeric(names(YestSVM1test))]==1&YestSVM1test==1)),length(which(Y[as.numeric(names(na.omit(YestLR1test)))]==0&na.omit(YestLR1test)==1)),length(which(Y[as.numeric(names(na.omit(YestLR1test)))]==1&na.omit(YestLR1test)==1)),length(which(dadostest$Y==0&YestADB1test==1)),length(which(dadostest$Y==1&YestADB1test==1)),
    length(which(dadostest$Y==0&YestNB2test==0)),length(which(dadostest$Y==1&YestNB2test==0)),RF2$test$confusion[1,1],RF2$test$confusion[2,1],length(which(Y[as.numeric(names(YestSVM2test))]==0&YestSVM2test==0)),length(which(Y[as.numeric(names(YestSVM2test))]==1&YestSVM2test==0)),length(which(Y[as.numeric(names(na.omit(YestLR2test)))]==0&na.omit(YestLR2test)==0)),length(which(Y[as.numeric(names(na.omit(YestLR2test)))]==1&na.omit(YestLR2test)==0)),length(which(dadostest$Y==0&YestADB2test==0)),length(which(dadostest$Y==1&YestADB2test==0)),
    length(which(dadostest$Y==0&YestNB2test==1)),length(which(dadostest$Y==1&YestNB2test==1)),RF2$test$confusion[1,2],RF2$test$confusion[2,2],length(which(Y[as.numeric(names(YestSVM2test))]==0&YestSVM2test==1)),length(which(Y[as.numeric(names(YestSVM2test))]==1&YestSVM2test==1)),length(which(Y[as.numeric(names(na.omit(YestLR2test)))]==0&na.omit(YestLR2test)==1)),length(which(Y[as.numeric(names(na.omit(YestLR2test)))]==1&na.omit(YestLR2test)==1)),length(which(dadostest$Y==0&YestADB2test==1)),length(which(dadostest$Y==1&YestADB2test==1)),
    length(which(dadostest$Y==0&YestNB3test==0)),length(which(dadostest$Y==1&YestNB3test==0)),RF3$test$confusion[1,1],RF3$test$confusion[2,1],length(which(Y[as.numeric(names(YestSVM3test))]==0&YestSVM3test==0)),length(which(Y[as.numeric(names(YestSVM3test))]==1&YestSVM3test==0)),length(which(Y[as.numeric(names(na.omit(YestLR3test)))]==0&na.omit(YestLR3test)==0)),length(which(Y[as.numeric(names(na.omit(YestLR3test)))]==1&na.omit(YestLR3test)==0)),length(which(dadostest$Y==0&YestADB3test==0)),length(which(dadostest$Y==1&YestADB3test==0)),
    length(which(dadostest$Y==0&YestNB3test==1)),length(which(dadostest$Y==1&YestNB3test==1)),RF3$test$confusion[1,2],RF3$test$confusion[2,2],length(which(Y[as.numeric(names(YestSVM3test))]==0&YestSVM3test==1)),length(which(Y[as.numeric(names(YestSVM3test))]==1&YestSVM3test==1)),length(which(Y[as.numeric(names(na.omit(YestLR3test)))]==0&na.omit(YestLR3test)==1)),length(which(Y[as.numeric(names(na.omit(YestLR3test)))]==1&na.omit(YestLR3test)==1)),length(which(dadostest$Y==0&YestADB3test==1)),length(which(dadostest$Y==1&YestADB3test==1)),
    length(which(dadostest$Y==0&YestNB4test==0)),length(which(dadostest$Y==1&YestNB4test==0)),RF4$test$confusion[1,1],RF4$test$confusion[2,1],length(which(Y[as.numeric(names(YestSVM4test))]==0&YestSVM4test==0)),length(which(Y[as.numeric(names(YestSVM4test))]==1&YestSVM4test==0)),length(which(Y[as.numeric(names(na.omit(YestLR4test)))]==0&na.omit(YestLR4test)==0)),length(which(Y[as.numeric(names(na.omit(YestLR4test)))]==1&na.omit(YestLR4test)==0)),length(which(dadostest$Y==0&YestADB4test==0)),length(which(dadostest$Y==1&YestADB4test==0)),
    length(which(dadostest$Y==0&YestNB4test==1)),length(which(dadostest$Y==1&YestNB4test==1)),RF4$test$confusion[1,2],RF4$test$confusion[2,2],length(which(Y[as.numeric(names(YestSVM4test))]==0&YestSVM4test==1)),length(which(Y[as.numeric(names(YestSVM4test))]==1&YestSVM4test==1)),length(which(Y[as.numeric(names(na.omit(YestLR4test)))]==0&na.omit(YestLR4test)==1)),length(which(Y[as.numeric(names(na.omit(YestLR4test)))]==1&na.omit(YestLR4test)==1)),length(which(dadostest$Y==0&YestADB4test==1)),length(which(dadostest$Y==1&YestADB4test==1)),
    length(which(dadostest$Y==0&YestNB5test==0)),length(which(dadostest$Y==1&YestNB5test==0)),RF5$test$confusion[1,1],RF5$test$confusion[2,1],length(which(Y[as.numeric(names(YestSVM5test))]==0&YestSVM5test==0)),length(which(Y[as.numeric(names(YestSVM5test))]==1&YestSVM5test==0)),length(which(Y[as.numeric(names(na.omit(YestLR5test)))]==0&na.omit(YestLR5test)==0)),length(which(Y[as.numeric(names(na.omit(YestLR5test)))]==1&na.omit(YestLR5test)==0)),length(which(dadostest$Y==0&YestADB5test==0)),length(which(dadostest$Y==1&YestADB5test==0)),
    length(which(dadostest$Y==0&YestNB5test==1)),length(which(dadostest$Y==1&YestNB5test==1)),RF5$test$confusion[1,2],RF5$test$confusion[2,2],length(which(Y[as.numeric(names(YestSVM5test))]==0&YestSVM5test==1)),length(which(Y[as.numeric(names(YestSVM5test))]==1&YestSVM5test==1)),length(which(Y[as.numeric(names(na.omit(YestLR5test)))]==0&na.omit(YestLR5test)==1)),length(which(Y[as.numeric(names(na.omit(YestLR5test)))]==1&na.omit(YestLR5test)==1)),length(which(dadostest$Y==0&YestADB5test==1)),length(which(dadostest$Y==1&YestADB5test==1)),
    length(which(dadostest$Y==0&YestNB6test==0)),length(which(dadostest$Y==1&YestNB6test==0)),RF6$test$confusion[1,1],RF6$test$confusion[2,1],length(which(Y[as.numeric(names(YestSVM6test))]==0&YestSVM6test==0)),length(which(Y[as.numeric(names(YestSVM6test))]==1&YestSVM6test==0)),length(which(Y[as.numeric(names(na.omit(YestLR6test)))]==0&na.omit(YestLR6test)==0)),length(which(Y[as.numeric(names(na.omit(YestLR6test)))]==1&na.omit(YestLR6test)==0)),length(which(dadostest$Y==0&YestADB6test==0)),length(which(dadostest$Y==1&YestADB6test==0)),
    length(which(dadostest$Y==0&YestNB6test==1)),length(which(dadostest$Y==1&YestNB6test==1)),RF6$test$confusion[1,2],RF6$test$confusion[2,2],length(which(Y[as.numeric(names(YestSVM6test))]==0&YestSVM6test==1)),length(which(Y[as.numeric(names(YestSVM6test))]==1&YestSVM6test==1)),length(which(Y[as.numeric(names(na.omit(YestLR6test)))]==0&na.omit(YestLR6test)==1)),length(which(Y[as.numeric(names(na.omit(YestLR6test)))]==1&na.omit(YestLR6test)==1)),length(which(dadostest$Y==0&YestADB6test==1)),length(which(dadostest$Y==1&YestADB6test==1))
  ),2*metodos,2*models)-matconftest)^2)/(B-1)
  
  
  j=j+1; print(j)
}
matdesvtest=round(sqrt(sdmatconftest),2)
matdesvtrei=round(sqrt(sdmatconftrei),2)

mattreifinal=cbind(matconftrei[,1],matdesvtrei[,1],matconftrei[,2],matdesvtrei[,2],matconftrei[,3],matdesvtrei[,3],matconftrei[,4],matdesvtrei[,4],matconftrei[,5],matdesvtrei[,5],matconftrei[,6],matdesvtrei[,6],matconftrei[,7],matdesvtrei[,7],matconftrei[,8],matdesvtrei[,8],matconftrei[,9],matdesvtrei[,9],matconftrei[,10],matdesvtrei[,10],matconftrei[,11],matdesvtrei[,11],matconftrei[,12],matdesvtrei[,12])
mattestfinal=cbind(matconftest[,1],matdesvtest[,1],matconftest[,2],matdesvtest[,2],matconftest[,3],matdesvtest[,3],matconftest[,4],matdesvtest[,4],matconftest[,5],matdesvtest[,5],matconftest[,6],matdesvtest[,6],matconftest[,7],matdesvtest[,7],matconftest[,8],matdesvtest[,8],matconftest[,9],matdesvtest[,9],matconftest[,10],matdesvtest[,10],matconftest[,11],matdesvtest[,11],matconftest[,12],matdesvtest[,12])

xtable(miacuraciastrei) #ACURACIAS TREINAMENTO
xtable(miacuraciastest) #ACURACIAS TESTE
xtable(sdacuraciastrei) #DESVIOS PADROES DAS ACURACIAS TREINAMENTO
xtable(sdacuraciastest) #DESVIOS PADROES DAS ACURACIAS TESTE
xtable(mattreifinal)    #MATRIZES DE CONFUSOES TREINAMENTO
xtable(mattestfinal)    #MATRIZES DE CONFUSOES TESTE


round(miacuraciastrei,2) #ACURACIAS TREINAMENTO
round(miacuraciastest,2) #ACURACIAS TESTE
round(sdacuraciastrei,2) #DESVIOS PADROES DAS ACURACIAS TREINAMENTO
round(sdacuraciastest,2) #DESVIOS PADROES DAS ACURACIAS TESTE

round(mitempo,2) #MEDIAS TEMPOS
round(sdtempo,2) #DESVIOS PADROES TEMPOS

tempotabela=cbind(mitempo[,1],sdtempo[,1],mitempo[,2],sdtempo[,2],mitempo[,3],sdtempo[,3],
                  mitempo[,4],sdtempo[,4],mitempo[,5],sdtempo[,5],mitempo[,6],sdtempo[,6])
row.names(tempotabela)=c("NB","DT","SVM","LR","Adaboost")
xtable(tempotabela)


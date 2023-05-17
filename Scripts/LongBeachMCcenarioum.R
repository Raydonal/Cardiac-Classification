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
dados=read.table("LongBeachdata.txt",na.strings = -9,header=T,dec=".",sep = "\t")
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

newdata = data.frame(cbind(age,sex,htn,X,Y))

B=100
models=4
metodos=5


acurNBtrei=matrix(,B,models);colnames(acurNBtrei)=c("A","B","C","D")
acurDTtrei=matrix(,B,models);colnames(acurDTtrei)=c("A","B","C","D")
acurSVMtrei=matrix(,B,models);colnames(acurSVMtrei)=c("A","B","C","D")
acurLRtrei=matrix(,B,models);colnames(acurLRtrei)=c("A","B","C","D")
acurADBtrei=matrix(,B,models);colnames(acurADBtrei)=c("A","B","C","D")

acurNBtest=matrix(,B,models);colnames(acurNBtest)=c("A","B","C","D")
acurDTtest=matrix(,B,models);colnames(acurDTtest)=c("A","B","C","D")
acurSVMtest=matrix(,B,models);colnames(acurSVMtest)=c("A","B","C","D")
acurLRtest=matrix(,B,models);colnames(acurLRtest)=c("A","B","C","D")
acurADBtest=matrix(,B,models);colnames(acurADBtest)=c("A","B","C","D")

matconftrei=matrix(0,2*metodos,2*models);colnames(matconftrei)=rep(c(0,1),models);rownames(matconftrei)=rep(c(0,1),metodos)
matconftest=matrix(0,2*metodos,2*models);colnames(matconftest)=rep(c(0,1),models);rownames(matconftest)=rep(c(0,1),metodos)
k=0 #contador

# 1 - Y~sex+htn
# 2 - Y~sex+age+htn+PAM+IPPA+HM+ALPHA
# 3 - Y~sex+age+htn+RC+ALPHA2
# 4 - Y~sex+age+htn+IPPARC+ALPHA2

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

modNB3=naiveBayes(Y~sex+age+htn+RC+ALPHA2,data = dadostrei) #### MELHOR
YestNB3trei=predict(modNB3,newdata = dadostrei);YestNB3test=predict(modNB3,newdata = dadostest)

modNB4=naiveBayes(Y~sex+age+htn+IPPARC+ALPHA2,data = dadostrei)
YestNB4trei=predict(modNB4,newdata = dadostrei);YestNB4test=predict(modNB4,newdata = dadostest)

##########################################
#             CLASSIFICACAO
#           ARVORE DE DECISAO
##########################################

DadostestRF1=na.omit(data.frame(dadostest$Y,dadostest$sex,dadostest$htn))
RF1=randomForest(Y~sex+htn,data = dadostrei,xtest=DadostestRF1[,-1],ytest=DadostestRF1$dadostest.Y,na.action = na.omit)

DadostestRF2=na.omit(data.frame(dadostest$Y,dadostest$sex,dadostest$age,dadostest$htn,dadostest$PAM,dadostest$IPPA,dadostest$HM,dadostest$ALPHA))
RF2=randomForest(Y~sex+age+htn+PAM+IPPA+HM+ALPHA,data = dadostrei,xtest=DadostestRF2[,-1],ytest=DadostestRF2$dadostest.Y,na.action = na.omit)

DadostestRF3=na.omit(data.frame(dadostest$Y,dadostest$sex,dadostest$age,dadostest$htn,dadostest$RC,dadostest$ALPHA2))
RF3=randomForest(Y~sex+age+htn+RC+ALPHA2,data = dadostrei,xtest=DadostestRF3[,-1],ytest=DadostestRF3$dadostest.Y,na.action = na.omit)

DadostestRF4=na.omit(data.frame(dadostest$Y,dadostest$sex,dadostest$age,dadostest$htn,dadostest$IPPARC,dadostest$ALPHA2))
RF4=randomForest(Y~sex+age+htn+IPPARC+ALPHA2,data = dadostrei,xtest=DadostestRF4[,-1],ytest=DadostestRF4$dadostest.Y,na.action = na.omit)

##########################################
#             CLASSIFICACAO
#     MAQUINA DE VETORES DE SUPORTE
##########################################

modSVM1=svm(Y~sex+htn,data=dadostrei,scale=F,kernel="poly",cost=100,epsilon=1.0e-12)
YestSVM1test=predict(modSVM1,dadostest);YestSVM1trei=modSVM1$fitted

modSVM2=svm(Y~sex+age+htn+PAM+IPPA+HM+ALPHA,data=dadostrei,scale=F,kernel="poly",cost=100,epsilon=1.0e-12)
YestSVM2test=predict(modSVM2,dadostest);YestSVM2trei=modSVM2$fitted

modSVM3=svm(Y~sex+age+htn+RC+ALPHA2,scale=F,data=dadostrei,kernel="poly",cost=100,epsilon=1.0e-12)
YestSVM3test=predict(modSVM3,dadostest);YestSVM3trei=modSVM3$fitted

modSVM4=svm(Y~sex+age+htn+IPPARC+ALPHA2,data=dadostrei,scale=F,kernel="poly",cost=100,epsilon=1.0e-12)
YestSVM4test=predict(modSVM4,dadostest);YestSVM4trei=modSVM4$fitted

##########################################
#             CLASSIFICACAO
#     REGRESSAO LINEAR - LOGISTICA
##########################################

modLR1=glm(Y~sex+htn,data=dadostrei,family = binomial(link = "logit"),na.action = na.omit)
YestLR1test=ifelse(predict.glm(modLR1,newdata = dadostest,type = "response")<0.5,0,1);YestLR1trei=ifelse(predict.glm(modLR1,newdata = dadostrei,type = "response")<0.5,0,1)

modLR2=glm(Y~sex+age+htn+PAM+IPPA+HM+ALPHA,data=dadostrei,family = binomial(link = "logit"))
YestLR2test=ifelse(predict.glm(modLR2,newdata = dadostest,type = "response")<0.5,0,1);YestLR2trei=ifelse(predict.glm(modLR2,newdata = dadostrei,type = "response")<0.5,0,1)

modLR3=glm(Y~sex+age+htn+RC+ALPHA2,data=dadostrei,family = binomial(link = "logit"))
YestLR3test=ifelse(predict.glm(modLR3,newdata = dadostest,type = "response")<0.5,0,1);YestLR3trei=ifelse(predict.glm(modLR3,newdata = dadostrei,type = "response")<0.5,0,1)

modLR4=glm(Y~sex+age+htn+IPPARC+ALPHA2,data=dadostrei,family = binomial(link = "logit"))
YestLR4test=ifelse(predict.glm(modLR4,newdata = dadostest,type = "response")<0.5,0,1);YestLR4trei=ifelse(predict.glm(modLR4,newdata = dadostrei,type = "response")<0.5,0,1)

##########################################
#             CLASSIFICACAO
#               ADABOOST
##########################################

modADB1=adaboost(Y~sex+htn,data=dadostrei,10)
YestADB1test=predict(modADB1,newdata=dadostest)$class;YestADB1trei=predict(modADB1,newdata=dadostrei)$class

modADB2=adaboost(Y~sex+age+htn+PAM+IPPA+HM+ALPHA,data=dadostrei,10)
YestADB2test=predict(modADB2,newdata=dadostest)$class;YestADB2trei=predict(modADB2,newdata=dadostrei)$class

modADB3=adaboost(Y~sex+age+htn+RC+ALPHA2,data=dadostrei,10)
YestADB3test=predict(modADB3,newdata=dadostest)$class;YestADB3trei=predict(modADB3,newdata=dadostrei)$class

modADB4=adaboost(Y~sex+age+htn+IPPARC+ALPHA2,data=dadostrei,10)
YestADB4test=predict(modADB4,newdata=dadostest)$class;YestADB4trei=predict(modADB4,newdata=dadostrei)$class


#############################
#         ACURACIAS         #
#############################

acurNBtrei[i,]=c((length(which(dadostrei$Y==0&YestNB1trei==0))+length(which(dadostrei$Y==1&YestNB1trei==1)))/length(dadostrei$Y)*100,(length(which(dadostrei$Y==0&YestNB2trei==0))+length(which(dadostrei$Y==1&YestNB2trei==1)))/length(dadostrei$Y)*100,(length(which(dadostrei$Y==0&YestNB3trei==0))+length(which(dadostrei$Y==1&YestNB3trei==1)))/length(dadostrei$Y)*100,(length(which(dadostrei$Y==0&YestNB4trei==0))+length(which(dadostrei$Y==1&YestNB4trei==1)))/length(dadostrei$Y)*100)
acurNBtest[i,]=c((length(which(dadostest$Y==0&YestNB1test==0))+length(which(dadostest$Y==1&YestNB1test==1)))/length(dadostest$Y)*100,(length(which(dadostest$Y==0&YestNB2test==0))+length(which(dadostest$Y==1&YestNB2test==1)))/length(dadostest$Y)*100,(length(which(dadostest$Y==0&YestNB3test==0))+length(which(dadostest$Y==1&YestNB3test==1)))/length(dadostest$Y)*100,(length(which(dadostest$Y==0&YestNB4test==0))+length(which(dadostest$Y==1&YestNB4test==1)))/length(dadostest$Y)*100)

acurDTtrei[i,]=c((RF1$confusion[1,1]+RF1$confusion[2,2])/sum(RF1$confusion[,-3])*100,(RF2$confusion[1,1]+RF2$confusion[2,2])/sum(RF2$confusion[,-3])*100,(RF3$confusion[1,1]+RF3$confusion[2,2])/sum(RF3$confusion[,-3])*100,(RF4$confusion[1,1]+RF4$confusion[2,2])/sum(RF4$confusion[,-3])*100)
acurDTtest[i,]=c((RF1$test$confusion[1,1]+RF1$test$confusion[2,2])/sum(RF1$test$confusion[,-3])*100,(RF2$test$confusion[1,1]+RF2$test$confusion[2,2])/sum(RF2$test$confusion[,-3])*100,(RF3$test$confusion[1,1]+RF3$test$confusion[2,2])/sum(RF3$test$confusion[,-3])*100,(RF4$test$confusion[1,1]+RF4$test$confusion[2,2])/sum(RF4$test$confusion[,-3])*100)

acurSVMtrei[i,]=c((length(which(Y[as.numeric(names(YestSVM1trei))]==0&YestSVM1trei==0))+length(which(Y[as.numeric(names(YestSVM1trei))]==1&YestSVM1trei==1)))/length(YestSVM1trei)*100,(length(which(Y[as.numeric(names(YestSVM2trei))]==0&YestSVM2trei==0))+length(which(Y[as.numeric(names(YestSVM2trei))]==1&YestSVM2trei==1)))/length(YestSVM2trei)*100,(length(which(Y[as.numeric(names(YestSVM3trei))]==0&YestSVM3trei==0))+length(which(Y[as.numeric(names(YestSVM3trei))]==1&YestSVM3trei==1)))/length(YestSVM3trei)*100,(length(which(Y[as.numeric(names(YestSVM4trei))]==0&YestSVM4trei==0))+length(which(Y[as.numeric(names(YestSVM4trei))]==1&YestSVM4trei==1)))/length(YestSVM4trei)*100)
acurSVMtest[i,]=c((length(which(Y[as.numeric(names(YestSVM1test))]==0&YestSVM1test==0))+length(which(Y[as.numeric(names(YestSVM1test))]==1&YestSVM1test==1)))/length(YestSVM1test)*100,(length(which(Y[as.numeric(names(YestSVM2test))]==0&YestSVM2test==0))+length(which(Y[as.numeric(names(YestSVM2test))]==1&YestSVM2test==1)))/length(YestSVM2test)*100,(length(which(Y[as.numeric(names(YestSVM3test))]==0&YestSVM3test==0))+length(which(Y[as.numeric(names(YestSVM3test))]==1&YestSVM3test==1)))/length(YestSVM3test)*100,(length(which(Y[as.numeric(names(YestSVM4test))]==0&YestSVM4test==0))+length(which(Y[as.numeric(names(YestSVM4test))]==1&YestSVM4test==1)))/length(YestSVM4test)*100)

acurLRtrei[i,]=c((length(which(Y[as.numeric(names(na.omit(YestLR1trei)))]==0&na.omit(YestLR1trei)==0))+length(which(Y[as.numeric(names(na.omit(YestLR1trei)))]==1&na.omit(YestLR1trei)==1)))/length(na.omit(YestLR1trei))*100,(length(which(Y[as.numeric(names(na.omit(YestLR2trei)))]==0&na.omit(YestLR2trei)==0))+length(which(Y[as.numeric(names(na.omit(YestLR2trei)))]==1&na.omit(YestLR2trei)==1)))/length(na.omit(YestLR2trei))*100,(length(which(Y[as.numeric(names(na.omit(YestLR3trei)))]==0&na.omit(YestLR3trei)==0))+length(which(Y[as.numeric(names(na.omit(YestLR3trei)))]==1&na.omit(YestLR3trei)==1)))/length(na.omit(YestLR3trei))*100,(length(which(Y[as.numeric(names(na.omit(YestLR4trei)))]==0&na.omit(YestLR4trei)==0))+length(which(Y[as.numeric(names(na.omit(YestLR4trei)))]==1&na.omit(YestLR4trei)==1)))/length(na.omit(YestLR4trei))*100)
acurLRtest[i,]=c((length(which(Y[as.numeric(names(na.omit(YestLR1test)))]==0&na.omit(YestLR1test)==0))+length(which(Y[as.numeric(names(na.omit(YestLR1test)))]==1&na.omit(YestLR1test)==1)))/length(na.omit(YestLR1test))*100,(length(which(Y[as.numeric(names(na.omit(YestLR2test)))]==0&na.omit(YestLR2test)==0))+length(which(Y[as.numeric(names(na.omit(YestLR2test)))]==1&na.omit(YestLR2test)==1)))/length(na.omit(YestLR2test))*100,(length(which(Y[as.numeric(names(na.omit(YestLR3test)))]==0&na.omit(YestLR3test)==0))+length(which(Y[as.numeric(names(na.omit(YestLR3test)))]==1&na.omit(YestLR3test)==1)))/length(na.omit(YestLR3test))*100,(length(which(Y[as.numeric(names(na.omit(YestLR4test)))]==0&na.omit(YestLR4test)==0))+length(which(Y[as.numeric(names(na.omit(YestLR4test)))]==1&na.omit(YestLR4test)==1)))/length(na.omit(YestLR4test))*100)

acurADBtrei[i,]=c((length(which(dadostrei$Y==0&YestADB1trei==0))+length(which(dadostrei$Y==1&YestADB1trei==1)))/length(dadostrei$Y)*100,(length(which(dadostrei$Y==0&YestADB2trei==0))+length(which(dadostrei$Y==1&YestADB2trei==1)))/length(dadostrei$Y)*100,(length(which(dadostrei$Y==0&YestADB3trei==0))+length(which(dadostrei$Y==1&YestADB3trei==1)))/length(dadostrei$Y)*100,(length(which(dadostrei$Y==0&YestADB4trei==0))+length(which(dadostrei$Y==1&YestADB4trei==1)))/length(dadostrei$Y)*100)
acurADBtest[i,]=c((length(which(dadostest$Y==0&YestADB1test==0))+length(which(dadostest$Y==1&YestADB1test==1)))/length(dadostest$Y)*100,(length(which(dadostest$Y==0&YestADB2test==0))+length(which(dadostest$Y==1&YestADB2test==1)))/length(dadostest$Y)*100,(length(which(dadostest$Y==0&YestADB3test==0))+length(which(dadostest$Y==1&YestADB3test==1)))/length(dadostest$Y)*100,(length(which(dadostest$Y==0&YestADB4test==0))+length(which(dadostest$Y==1&YestADB4test==1)))/length(dadostest$Y)*100)


matconftrei=matconftrei+matrix(c(
length(which(dadostrei$Y==0&YestNB1trei==0)),length(which(dadostrei$Y==1&YestNB1trei==0)),RF1$confusion[1,1],RF1$confusion[2,1],length(which(Y[as.numeric(names(YestSVM1trei))]==0&YestSVM1trei==0)),length(which(Y[as.numeric(names(YestSVM1trei))]==1&YestSVM1trei==0)),length(which(Y[as.numeric(names(na.omit(YestLR1trei)))]==0&na.omit(YestLR1trei)==0)),length(which(Y[as.numeric(names(na.omit(YestLR1trei)))]==1&na.omit(YestLR1trei)==0)),length(which(dadostrei$Y==0&YestADB1trei==0)),length(which(dadostrei$Y==1&YestADB1trei==0)),
length(which(dadostrei$Y==0&YestNB1trei==1)),length(which(dadostrei$Y==1&YestNB1trei==1)),RF1$confusion[1,2],RF1$confusion[2,2],length(which(Y[as.numeric(names(YestSVM1trei))]==0&YestSVM1trei==1)),length(which(Y[as.numeric(names(YestSVM1trei))]==1&YestSVM1trei==1)),length(which(Y[as.numeric(names(na.omit(YestLR1trei)))]==0&na.omit(YestLR1trei)==1)),length(which(Y[as.numeric(names(na.omit(YestLR1trei)))]==1&na.omit(YestLR1trei)==1)),length(which(dadostrei$Y==0&YestADB1trei==1)),length(which(dadostrei$Y==1&YestADB1trei==1)),
length(which(dadostrei$Y==0&YestNB2trei==0)),length(which(dadostrei$Y==1&YestNB2trei==0)),RF2$confusion[1,1],RF2$confusion[2,1],length(which(Y[as.numeric(names(YestSVM2trei))]==0&YestSVM2trei==0)),length(which(Y[as.numeric(names(YestSVM2trei))]==1&YestSVM2trei==0)),length(which(Y[as.numeric(names(na.omit(YestLR2trei)))]==0&na.omit(YestLR2trei)==0)),length(which(Y[as.numeric(names(na.omit(YestLR2trei)))]==1&na.omit(YestLR2trei)==0)),length(which(dadostrei$Y==0&YestADB2trei==0)),length(which(dadostrei$Y==1&YestADB2trei==0)),
length(which(dadostrei$Y==0&YestNB2trei==1)),length(which(dadostrei$Y==1&YestNB2trei==1)),RF2$confusion[1,2],RF2$confusion[2,2],length(which(Y[as.numeric(names(YestSVM2trei))]==0&YestSVM2trei==1)),length(which(Y[as.numeric(names(YestSVM2trei))]==1&YestSVM2trei==1)),length(which(Y[as.numeric(names(na.omit(YestLR2trei)))]==0&na.omit(YestLR2trei)==1)),length(which(Y[as.numeric(names(na.omit(YestLR2trei)))]==1&na.omit(YestLR2trei)==1)),length(which(dadostrei$Y==0&YestADB2trei==1)),length(which(dadostrei$Y==1&YestADB2trei==1)),
length(which(dadostrei$Y==0&YestNB3trei==0)),length(which(dadostrei$Y==1&YestNB3trei==0)),RF3$confusion[1,1],RF3$confusion[2,1],length(which(Y[as.numeric(names(YestSVM3trei))]==0&YestSVM3trei==0)),length(which(Y[as.numeric(names(YestSVM3trei))]==1&YestSVM3trei==0)),length(which(Y[as.numeric(names(na.omit(YestLR3trei)))]==0&na.omit(YestLR3trei)==0)),length(which(Y[as.numeric(names(na.omit(YestLR3trei)))]==1&na.omit(YestLR3trei)==0)),length(which(dadostrei$Y==0&YestADB3trei==0)),length(which(dadostrei$Y==1&YestADB3trei==0)),
length(which(dadostrei$Y==0&YestNB3trei==1)),length(which(dadostrei$Y==1&YestNB3trei==1)),RF3$confusion[1,2],RF3$confusion[2,2],length(which(Y[as.numeric(names(YestSVM3trei))]==0&YestSVM3trei==1)),length(which(Y[as.numeric(names(YestSVM3trei))]==1&YestSVM3trei==1)),length(which(Y[as.numeric(names(na.omit(YestLR3trei)))]==0&na.omit(YestLR3trei)==1)),length(which(Y[as.numeric(names(na.omit(YestLR3trei)))]==1&na.omit(YestLR3trei)==1)),length(which(dadostrei$Y==0&YestADB3trei==1)),length(which(dadostrei$Y==1&YestADB3trei==1)),
length(which(dadostrei$Y==0&YestNB4trei==0)),length(which(dadostrei$Y==1&YestNB4trei==0)),RF4$confusion[1,1],RF4$confusion[2,1],length(which(Y[as.numeric(names(YestSVM4trei))]==0&YestSVM4trei==0)),length(which(Y[as.numeric(names(YestSVM4trei))]==1&YestSVM4trei==0)),length(which(Y[as.numeric(names(na.omit(YestLR4trei)))]==0&na.omit(YestLR4trei)==0)),length(which(Y[as.numeric(names(na.omit(YestLR4trei)))]==1&na.omit(YestLR4trei)==0)),length(which(dadostrei$Y==0&YestADB4trei==0)),length(which(dadostrei$Y==1&YestADB4trei==0)),
length(which(dadostrei$Y==0&YestNB4trei==1)),length(which(dadostrei$Y==1&YestNB4trei==1)),RF4$confusion[1,2],RF4$confusion[2,2],length(which(Y[as.numeric(names(YestSVM4trei))]==0&YestSVM4trei==1)),length(which(Y[as.numeric(names(YestSVM4trei))]==1&YestSVM4trei==1)),length(which(Y[as.numeric(names(na.omit(YestLR4trei)))]==0&na.omit(YestLR4trei)==1)),length(which(Y[as.numeric(names(na.omit(YestLR4trei)))]==1&na.omit(YestLR4trei)==1)),length(which(dadostrei$Y==0&YestADB4trei==1)),length(which(dadostrei$Y==1&YestADB4trei==1))
),2*metodos,2*models)/B


matconftest=matconftest+matrix(c(
  length(which(dadostest$Y==0&YestNB1test==0)),length(which(dadostest$Y==1&YestNB1test==0)),RF1$test$confusion[1,1],RF1$test$confusion[2,1],length(which(Y[as.numeric(names(YestSVM1test))]==0&YestSVM1test==0)),length(which(Y[as.numeric(names(YestSVM1test))]==1&YestSVM1test==0)),length(which(Y[as.numeric(names(na.omit(YestLR1test)))]==0&na.omit(YestLR1test)==0)),length(which(Y[as.numeric(names(na.omit(YestLR1test)))]==1&na.omit(YestLR1test)==0)),length(which(dadostest$Y==0&YestADB1test==0)),length(which(dadostest$Y==1&YestADB1test==0)),
  length(which(dadostest$Y==0&YestNB1test==1)),length(which(dadostest$Y==1&YestNB1test==1)),RF1$test$confusion[1,2],RF1$test$confusion[2,2],length(which(Y[as.numeric(names(YestSVM1test))]==0&YestSVM1test==1)),length(which(Y[as.numeric(names(YestSVM1test))]==1&YestSVM1test==1)),length(which(Y[as.numeric(names(na.omit(YestLR1test)))]==0&na.omit(YestLR1test)==1)),length(which(Y[as.numeric(names(na.omit(YestLR1test)))]==1&na.omit(YestLR1test)==1)),length(which(dadostest$Y==0&YestADB1test==1)),length(which(dadostest$Y==1&YestADB1test==1)),
  length(which(dadostest$Y==0&YestNB2test==0)),length(which(dadostest$Y==1&YestNB2test==0)),RF2$test$confusion[1,1],RF2$test$confusion[2,1],length(which(Y[as.numeric(names(YestSVM2test))]==0&YestSVM2test==0)),length(which(Y[as.numeric(names(YestSVM2test))]==1&YestSVM2test==0)),length(which(Y[as.numeric(names(na.omit(YestLR2test)))]==0&na.omit(YestLR2test)==0)),length(which(Y[as.numeric(names(na.omit(YestLR2test)))]==1&na.omit(YestLR2test)==0)),length(which(dadostest$Y==0&YestADB2test==0)),length(which(dadostest$Y==1&YestADB2test==0)),
  length(which(dadostest$Y==0&YestNB2test==1)),length(which(dadostest$Y==1&YestNB2test==1)),RF2$test$confusion[1,2],RF2$test$confusion[2,2],length(which(Y[as.numeric(names(YestSVM2test))]==0&YestSVM2test==1)),length(which(Y[as.numeric(names(YestSVM2test))]==1&YestSVM2test==1)),length(which(Y[as.numeric(names(na.omit(YestLR2test)))]==0&na.omit(YestLR2test)==1)),length(which(Y[as.numeric(names(na.omit(YestLR2test)))]==1&na.omit(YestLR2test)==1)),length(which(dadostest$Y==0&YestADB2test==1)),length(which(dadostest$Y==1&YestADB2test==1)),
  length(which(dadostest$Y==0&YestNB3test==0)),length(which(dadostest$Y==1&YestNB3test==0)),RF3$test$confusion[1,1],RF3$test$confusion[2,1],length(which(Y[as.numeric(names(YestSVM3test))]==0&YestSVM3test==0)),length(which(Y[as.numeric(names(YestSVM3test))]==1&YestSVM3test==0)),length(which(Y[as.numeric(names(na.omit(YestLR3test)))]==0&na.omit(YestLR3test)==0)),length(which(Y[as.numeric(names(na.omit(YestLR3test)))]==1&na.omit(YestLR3test)==0)),length(which(dadostest$Y==0&YestADB3test==0)),length(which(dadostest$Y==1&YestADB3test==0)),
  length(which(dadostest$Y==0&YestNB3test==1)),length(which(dadostest$Y==1&YestNB3test==1)),RF3$test$confusion[1,2],RF3$test$confusion[2,2],length(which(Y[as.numeric(names(YestSVM3test))]==0&YestSVM3test==1)),length(which(Y[as.numeric(names(YestSVM3test))]==1&YestSVM3test==1)),length(which(Y[as.numeric(names(na.omit(YestLR3test)))]==0&na.omit(YestLR3test)==1)),length(which(Y[as.numeric(names(na.omit(YestLR3test)))]==1&na.omit(YestLR3test)==1)),length(which(dadostest$Y==0&YestADB3test==1)),length(which(dadostest$Y==1&YestADB3test==1)),
  length(which(dadostest$Y==0&YestNB4test==0)),length(which(dadostest$Y==1&YestNB4test==0)),RF4$test$confusion[1,1],RF4$test$confusion[2,1],length(which(Y[as.numeric(names(YestSVM4test))]==0&YestSVM4test==0)),length(which(Y[as.numeric(names(YestSVM4test))]==1&YestSVM4test==0)),length(which(Y[as.numeric(names(na.omit(YestLR4test)))]==0&na.omit(YestLR4test)==0)),length(which(Y[as.numeric(names(na.omit(YestLR4test)))]==1&na.omit(YestLR4test)==0)),length(which(dadostest$Y==0&YestADB4test==0)),length(which(dadostest$Y==1&YestADB4test==0)),
  length(which(dadostest$Y==0&YestNB4test==1)),length(which(dadostest$Y==1&YestNB4test==1)),RF4$test$confusion[1,2],RF4$test$confusion[2,2],length(which(Y[as.numeric(names(YestSVM4test))]==0&YestSVM4test==1)),length(which(Y[as.numeric(names(YestSVM4test))]==1&YestSVM4test==1)),length(which(Y[as.numeric(names(na.omit(YestLR4test)))]==0&na.omit(YestLR4test)==1)),length(which(Y[as.numeric(names(na.omit(YestLR4test)))]==1&na.omit(YestLR4test)==1)),length(which(dadostest$Y==0&YestADB4test==1)),length(which(dadostest$Y==1&YestADB4test==1))
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
  
  modNB3=naiveBayes(Y~sex+age+htn+RC+ALPHA2,data = dadostrei) #### MELHOR
  YestNB3trei=predict(modNB3,newdata = dadostrei);YestNB3test=predict(modNB3,newdata = dadostest)
  
  modNB4=naiveBayes(Y~sex+age+htn+IPPARC+ALPHA2,data = dadostrei)
  YestNB4trei=predict(modNB4,newdata = dadostrei);YestNB4test=predict(modNB4,newdata = dadostest)
  
  ##########################################
  #             CLASSIFICACAO
  #           ARVORE DE DECISAO
  ##########################################
  
  DadostestRF1=na.omit(data.frame(dadostest$Y,dadostest$sex,dadostest$htn))
  RF1=randomForest(Y~sex+htn,data = dadostrei,xtest=DadostestRF1[,-1],ytest=DadostestRF1$dadostest.Y,na.action = na.omit)
  
  DadostestRF2=na.omit(data.frame(dadostest$Y,dadostest$sex,dadostest$age,dadostest$htn,dadostest$PAM,dadostest$IPPA,dadostest$HM,dadostest$ALPHA))
  RF2=randomForest(Y~sex+age+htn+PAM+IPPA+HM+ALPHA,data = dadostrei,xtest=DadostestRF2[,-1],ytest=DadostestRF2$dadostest.Y,na.action = na.omit)
  
  DadostestRF3=na.omit(data.frame(dadostest$Y,dadostest$sex,dadostest$age,dadostest$htn,dadostest$RC,dadostest$ALPHA2))
  RF3=randomForest(Y~sex+age+htn+RC+ALPHA2,data = dadostrei,xtest=DadostestRF3[,-1],ytest=DadostestRF3$dadostest.Y,na.action = na.omit)
  
  DadostestRF4=na.omit(data.frame(dadostest$Y,dadostest$sex,dadostest$age,dadostest$htn,dadostest$IPPARC,dadostest$ALPHA2))
  RF4=randomForest(Y~sex+age+htn+IPPARC+ALPHA2,data = dadostrei,xtest=DadostestRF4[,-1],ytest=DadostestRF4$dadostest.Y,na.action = na.omit)
  
  ##########################################
  #             CLASSIFICACAO
  #     MAQUINA DE VETORES DE SUPORTE
  ##########################################
  
  modSVM1=svm(Y~sex+htn,data=dadostrei,scale=F,kernel="poly",cost=100,epsilon=1.0e-12)
  YestSVM1test=predict(modSVM1,dadostest);YestSVM1trei=modSVM1$fitted
  
  modSVM2=svm(Y~sex+age+htn+PAM+IPPA+HM+ALPHA,data=dadostrei,scale=F,kernel="poly",cost=100,epsilon=1.0e-12)
  YestSVM2test=predict(modSVM2,dadostest);YestSVM2trei=modSVM2$fitted
  
  modSVM3=svm(Y~sex+age+htn+RC+ALPHA2,scale=F,data=dadostrei,kernel="poly",cost=100,epsilon=1.0e-12)
  YestSVM3test=predict(modSVM3,dadostest);YestSVM3trei=modSVM3$fitted
  
  modSVM4=svm(Y~sex+age+htn+IPPARC+ALPHA2,data=dadostrei,scale=F,kernel="poly",cost=100,epsilon=1.0e-12)
  YestSVM4test=predict(modSVM4,dadostest);YestSVM4trei=modSVM4$fitted
  
  ##########################################
  #             CLASSIFICACAO
  #     REGRESSAO LINEAR - LOGISTICA
  ##########################################
  
  modLR1=glm(Y~sex+htn,data=dadostrei,family = binomial(link = "logit"),na.action = na.omit)
  YestLR1test=ifelse(predict.glm(modLR1,newdata = dadostest,type = "response")<0.5,0,1);YestLR1trei=ifelse(predict.glm(modLR1,newdata = dadostrei,type = "response")<0.5,0,1)
  
  modLR2=glm(Y~sex+age+htn+PAM+IPPA+HM+ALPHA,data=dadostrei,family = binomial(link = "logit"))
  YestLR2test=ifelse(predict.glm(modLR2,newdata = dadostest,type = "response")<0.5,0,1);YestLR2trei=ifelse(predict.glm(modLR2,newdata = dadostrei,type = "response")<0.5,0,1)
  
  modLR3=glm(Y~sex+age+htn+RC+ALPHA2,data=dadostrei,family = binomial(link = "logit"))
  YestLR3test=ifelse(predict.glm(modLR3,newdata = dadostest,type = "response")<0.5,0,1);YestLR3trei=ifelse(predict.glm(modLR3,newdata = dadostrei,type = "response")<0.5,0,1)
  
  modLR4=glm(Y~sex+age+htn+IPPARC+ALPHA2,data=dadostrei,family = binomial(link = "logit"))
  YestLR4test=ifelse(predict.glm(modLR4,newdata = dadostest,type = "response")<0.5,0,1);YestLR4trei=ifelse(predict.glm(modLR4,newdata = dadostrei,type = "response")<0.5,0,1)
  
  ##########################################
  #             CLASSIFICACAO
  #               ADABOOST
  ##########################################
  
  modADB1=adaboost(Y~sex+htn,data=dadostrei,10)
  YestADB1test=predict(modADB1,newdata=dadostest)$class;YestADB1trei=predict(modADB1,newdata=dadostrei)$class
  
  modADB2=adaboost(Y~sex+age+htn+PAM+IPPA+HM+ALPHA,data=dadostrei,10)
  YestADB2test=predict(modADB2,newdata=dadostest)$class;YestADB2trei=predict(modADB2,newdata=dadostrei)$class
  
  modADB3=adaboost(Y~sex+age+htn+RC+ALPHA2,data=dadostrei,10)
  YestADB3test=predict(modADB3,newdata=dadostest)$class;YestADB3trei=predict(modADB3,newdata=dadostrei)$class
  
  modADB4=adaboost(Y~sex+age+htn+IPPARC+ALPHA2,data=dadostrei,10)
  YestADB4test=predict(modADB4,newdata=dadostest)$class;YestADB4trei=predict(modADB4,newdata=dadostrei)$class
  
  #############################
  #         ACURACIAS         #
  #############################
  
  acurNBtrei[i,]=c((length(which(dadostrei$Y==0&YestNB1trei==0))+length(which(dadostrei$Y==1&YestNB1trei==1)))/length(dadostrei$Y)*100,(length(which(dadostrei$Y==0&YestNB2trei==0))+length(which(dadostrei$Y==1&YestNB2trei==1)))/length(dadostrei$Y)*100,(length(which(dadostrei$Y==0&YestNB3trei==0))+length(which(dadostrei$Y==1&YestNB3trei==1)))/length(dadostrei$Y)*100,(length(which(dadostrei$Y==0&YestNB4trei==0))+length(which(dadostrei$Y==1&YestNB4trei==1)))/length(dadostrei$Y)*100)
  acurNBtest[i,]=c((length(which(dadostest$Y==0&YestNB1test==0))+length(which(dadostest$Y==1&YestNB1test==1)))/length(dadostest$Y)*100,(length(which(dadostest$Y==0&YestNB2test==0))+length(which(dadostest$Y==1&YestNB2test==1)))/length(dadostest$Y)*100,(length(which(dadostest$Y==0&YestNB3test==0))+length(which(dadostest$Y==1&YestNB3test==1)))/length(dadostest$Y)*100,(length(which(dadostest$Y==0&YestNB4test==0))+length(which(dadostest$Y==1&YestNB4test==1)))/length(dadostest$Y)*100)
  
  acurDTtrei[i,]=c((RF1$confusion[1,1]+RF1$confusion[2,2])/sum(RF1$confusion[,-3])*100,(RF2$confusion[1,1]+RF2$confusion[2,2])/sum(RF2$confusion[,-3])*100,(RF3$confusion[1,1]+RF3$confusion[2,2])/sum(RF3$confusion[,-3])*100,(RF4$confusion[1,1]+RF4$confusion[2,2])/sum(RF4$confusion[,-3])*100)
  acurDTtest[i,]=c((RF1$test$confusion[1,1]+RF1$test$confusion[2,2])/sum(RF1$test$confusion[,-3])*100,(RF2$test$confusion[1,1]+RF2$test$confusion[2,2])/sum(RF2$test$confusion[,-3])*100,(RF3$test$confusion[1,1]+RF3$test$confusion[2,2])/sum(RF3$test$confusion[,-3])*100,(RF4$test$confusion[1,1]+RF4$test$confusion[2,2])/sum(RF4$test$confusion[,-3])*100)
  
  acurSVMtrei[i,]=c((length(which(Y[as.numeric(names(YestSVM1trei))]==0&YestSVM1trei==0))+length(which(Y[as.numeric(names(YestSVM1trei))]==1&YestSVM1trei==1)))/length(YestSVM1trei)*100,(length(which(Y[as.numeric(names(YestSVM2trei))]==0&YestSVM2trei==0))+length(which(Y[as.numeric(names(YestSVM2trei))]==1&YestSVM2trei==1)))/length(YestSVM2trei)*100,(length(which(Y[as.numeric(names(YestSVM3trei))]==0&YestSVM3trei==0))+length(which(Y[as.numeric(names(YestSVM3trei))]==1&YestSVM3trei==1)))/length(YestSVM3trei)*100,(length(which(Y[as.numeric(names(YestSVM4trei))]==0&YestSVM4trei==0))+length(which(Y[as.numeric(names(YestSVM4trei))]==1&YestSVM4trei==1)))/length(YestSVM4trei)*100)
  acurSVMtest[i,]=c((length(which(Y[as.numeric(names(YestSVM1test))]==0&YestSVM1test==0))+length(which(Y[as.numeric(names(YestSVM1test))]==1&YestSVM1test==1)))/length(YestSVM1test)*100,(length(which(Y[as.numeric(names(YestSVM2test))]==0&YestSVM2test==0))+length(which(Y[as.numeric(names(YestSVM2test))]==1&YestSVM2test==1)))/length(YestSVM2test)*100,(length(which(Y[as.numeric(names(YestSVM3test))]==0&YestSVM3test==0))+length(which(Y[as.numeric(names(YestSVM3test))]==1&YestSVM3test==1)))/length(YestSVM3test)*100,(length(which(Y[as.numeric(names(YestSVM4test))]==0&YestSVM4test==0))+length(which(Y[as.numeric(names(YestSVM4test))]==1&YestSVM4test==1)))/length(YestSVM4test)*100)
  
  acurLRtrei[i,]=c((length(which(Y[as.numeric(names(na.omit(YestLR1trei)))]==0&na.omit(YestLR1trei)==0))+length(which(Y[as.numeric(names(na.omit(YestLR1trei)))]==1&na.omit(YestLR1trei)==1)))/length(na.omit(YestLR1trei))*100,(length(which(Y[as.numeric(names(na.omit(YestLR2trei)))]==0&na.omit(YestLR2trei)==0))+length(which(Y[as.numeric(names(na.omit(YestLR2trei)))]==1&na.omit(YestLR2trei)==1)))/length(na.omit(YestLR2trei))*100,(length(which(Y[as.numeric(names(na.omit(YestLR3trei)))]==0&na.omit(YestLR3trei)==0))+length(which(Y[as.numeric(names(na.omit(YestLR3trei)))]==1&na.omit(YestLR3trei)==1)))/length(na.omit(YestLR3trei))*100,(length(which(Y[as.numeric(names(na.omit(YestLR4trei)))]==0&na.omit(YestLR4trei)==0))+length(which(Y[as.numeric(names(na.omit(YestLR4trei)))]==1&na.omit(YestLR4trei)==1)))/length(na.omit(YestLR4trei))*100)
  acurLRtest[i,]=c((length(which(Y[as.numeric(names(na.omit(YestLR1test)))]==0&na.omit(YestLR1test)==0))+length(which(Y[as.numeric(names(na.omit(YestLR1test)))]==1&na.omit(YestLR1test)==1)))/length(na.omit(YestLR1test))*100,(length(which(Y[as.numeric(names(na.omit(YestLR2test)))]==0&na.omit(YestLR2test)==0))+length(which(Y[as.numeric(names(na.omit(YestLR2test)))]==1&na.omit(YestLR2test)==1)))/length(na.omit(YestLR2test))*100,(length(which(Y[as.numeric(names(na.omit(YestLR3test)))]==0&na.omit(YestLR3test)==0))+length(which(Y[as.numeric(names(na.omit(YestLR3test)))]==1&na.omit(YestLR3test)==1)))/length(na.omit(YestLR3test))*100,(length(which(Y[as.numeric(names(na.omit(YestLR4test)))]==0&na.omit(YestLR4test)==0))+length(which(Y[as.numeric(names(na.omit(YestLR4test)))]==1&na.omit(YestLR4test)==1)))/length(na.omit(YestLR4test))*100)
  
  acurADBtrei[i,]=c((length(which(dadostrei$Y==0&YestADB1trei==0))+length(which(dadostrei$Y==1&YestADB1trei==1)))/length(dadostrei$Y)*100,(length(which(dadostrei$Y==0&YestADB2trei==0))+length(which(dadostrei$Y==1&YestADB2trei==1)))/length(dadostrei$Y)*100,(length(which(dadostrei$Y==0&YestADB3trei==0))+length(which(dadostrei$Y==1&YestADB3trei==1)))/length(dadostrei$Y)*100,(length(which(dadostrei$Y==0&YestADB4trei==0))+length(which(dadostrei$Y==1&YestADB4trei==1)))/length(dadostrei$Y)*100)
  acurADBtest[i,]=c((length(which(dadostest$Y==0&YestADB1test==0))+length(which(dadostest$Y==1&YestADB1test==1)))/length(dadostest$Y)*100,(length(which(dadostest$Y==0&YestADB2test==0))+length(which(dadostest$Y==1&YestADB2test==1)))/length(dadostest$Y)*100,(length(which(dadostest$Y==0&YestADB3test==0))+length(which(dadostest$Y==1&YestADB3test==1)))/length(dadostest$Y)*100,(length(which(dadostest$Y==0&YestADB4test==0))+length(which(dadostest$Y==1&YestADB4test==1)))/length(dadostest$Y)*100)
  
  
  sdmatconftrei=sdmatconftrei+((matrix(c(
    length(which(dadostrei$Y==0&YestNB1trei==0)),length(which(dadostrei$Y==1&YestNB1trei==0)),RF1$confusion[1,1],RF1$confusion[2,1],length(which(Y[as.numeric(names(YestSVM1trei))]==0&YestSVM1trei==0)),length(which(Y[as.numeric(names(YestSVM1trei))]==1&YestSVM1trei==0)),length(which(Y[as.numeric(names(na.omit(YestLR1trei)))]==0&na.omit(YestLR1trei)==0)),length(which(Y[as.numeric(names(na.omit(YestLR1trei)))]==1&na.omit(YestLR1trei)==0)),length(which(dadostrei$Y==0&YestADB1trei==0)),length(which(dadostrei$Y==1&YestADB1trei==0)),
    length(which(dadostrei$Y==0&YestNB1trei==1)),length(which(dadostrei$Y==1&YestNB1trei==1)),RF1$confusion[1,2],RF1$confusion[2,2],length(which(Y[as.numeric(names(YestSVM1trei))]==0&YestSVM1trei==1)),length(which(Y[as.numeric(names(YestSVM1trei))]==1&YestSVM1trei==1)),length(which(Y[as.numeric(names(na.omit(YestLR1trei)))]==0&na.omit(YestLR1trei)==1)),length(which(Y[as.numeric(names(na.omit(YestLR1trei)))]==1&na.omit(YestLR1trei)==1)),length(which(dadostrei$Y==0&YestADB1trei==1)),length(which(dadostrei$Y==1&YestADB1trei==1)),
    length(which(dadostrei$Y==0&YestNB2trei==0)),length(which(dadostrei$Y==1&YestNB2trei==0)),RF2$confusion[1,1],RF2$confusion[2,1],length(which(Y[as.numeric(names(YestSVM2trei))]==0&YestSVM2trei==0)),length(which(Y[as.numeric(names(YestSVM2trei))]==1&YestSVM2trei==0)),length(which(Y[as.numeric(names(na.omit(YestLR2trei)))]==0&na.omit(YestLR2trei)==0)),length(which(Y[as.numeric(names(na.omit(YestLR2trei)))]==1&na.omit(YestLR2trei)==0)),length(which(dadostrei$Y==0&YestADB2trei==0)),length(which(dadostrei$Y==1&YestADB2trei==0)),
    length(which(dadostrei$Y==0&YestNB2trei==1)),length(which(dadostrei$Y==1&YestNB2trei==1)),RF2$confusion[1,2],RF2$confusion[2,2],length(which(Y[as.numeric(names(YestSVM2trei))]==0&YestSVM2trei==1)),length(which(Y[as.numeric(names(YestSVM2trei))]==1&YestSVM2trei==1)),length(which(Y[as.numeric(names(na.omit(YestLR2trei)))]==0&na.omit(YestLR2trei)==1)),length(which(Y[as.numeric(names(na.omit(YestLR2trei)))]==1&na.omit(YestLR2trei)==1)),length(which(dadostrei$Y==0&YestADB2trei==1)),length(which(dadostrei$Y==1&YestADB2trei==1)),
    length(which(dadostrei$Y==0&YestNB3trei==0)),length(which(dadostrei$Y==1&YestNB3trei==0)),RF3$confusion[1,1],RF3$confusion[2,1],length(which(Y[as.numeric(names(YestSVM3trei))]==0&YestSVM3trei==0)),length(which(Y[as.numeric(names(YestSVM3trei))]==1&YestSVM3trei==0)),length(which(Y[as.numeric(names(na.omit(YestLR3trei)))]==0&na.omit(YestLR3trei)==0)),length(which(Y[as.numeric(names(na.omit(YestLR3trei)))]==1&na.omit(YestLR3trei)==0)),length(which(dadostrei$Y==0&YestADB3trei==0)),length(which(dadostrei$Y==1&YestADB3trei==0)),
    length(which(dadostrei$Y==0&YestNB3trei==1)),length(which(dadostrei$Y==1&YestNB3trei==1)),RF3$confusion[1,2],RF3$confusion[2,2],length(which(Y[as.numeric(names(YestSVM3trei))]==0&YestSVM3trei==1)),length(which(Y[as.numeric(names(YestSVM3trei))]==1&YestSVM3trei==1)),length(which(Y[as.numeric(names(na.omit(YestLR3trei)))]==0&na.omit(YestLR3trei)==1)),length(which(Y[as.numeric(names(na.omit(YestLR3trei)))]==1&na.omit(YestLR3trei)==1)),length(which(dadostrei$Y==0&YestADB3trei==1)),length(which(dadostrei$Y==1&YestADB3trei==1)),
    length(which(dadostrei$Y==0&YestNB4trei==0)),length(which(dadostrei$Y==1&YestNB4trei==0)),RF4$confusion[1,1],RF4$confusion[2,1],length(which(Y[as.numeric(names(YestSVM4trei))]==0&YestSVM4trei==0)),length(which(Y[as.numeric(names(YestSVM4trei))]==1&YestSVM4trei==0)),length(which(Y[as.numeric(names(na.omit(YestLR4trei)))]==0&na.omit(YestLR4trei)==0)),length(which(Y[as.numeric(names(na.omit(YestLR4trei)))]==1&na.omit(YestLR4trei)==0)),length(which(dadostrei$Y==0&YestADB4trei==0)),length(which(dadostrei$Y==1&YestADB4trei==0)),
    length(which(dadostrei$Y==0&YestNB4trei==1)),length(which(dadostrei$Y==1&YestNB4trei==1)),RF4$confusion[1,2],RF4$confusion[2,2],length(which(Y[as.numeric(names(YestSVM4trei))]==0&YestSVM4trei==1)),length(which(Y[as.numeric(names(YestSVM4trei))]==1&YestSVM4trei==1)),length(which(Y[as.numeric(names(na.omit(YestLR4trei)))]==0&na.omit(YestLR4trei)==1)),length(which(Y[as.numeric(names(na.omit(YestLR4trei)))]==1&na.omit(YestLR4trei)==1)),length(which(dadostrei$Y==0&YestADB4trei==1)),length(which(dadostrei$Y==1&YestADB4trei==1))
    ),2*metodos,2*models)-matconftrei)^2)/(B-1)
  
  
  sdmatconftest=sdmatconftest+((matrix(c(
    length(which(dadostest$Y==0&YestNB1test==0)),length(which(dadostest$Y==1&YestNB1test==0)),RF1$test$confusion[1,1],RF1$test$confusion[2,1],length(which(Y[as.numeric(names(YestSVM1test))]==0&YestSVM1test==0)),length(which(Y[as.numeric(names(YestSVM1test))]==1&YestSVM1test==0)),length(which(Y[as.numeric(names(na.omit(YestLR1test)))]==0&na.omit(YestLR1test)==0)),length(which(Y[as.numeric(names(na.omit(YestLR1test)))]==1&na.omit(YestLR1test)==0)),length(which(dadostest$Y==0&YestADB1test==0)),length(which(dadostest$Y==1&YestADB1test==0)),
    length(which(dadostest$Y==0&YestNB1test==1)),length(which(dadostest$Y==1&YestNB1test==1)),RF1$test$confusion[1,2],RF1$test$confusion[2,2],length(which(Y[as.numeric(names(YestSVM1test))]==0&YestSVM1test==1)),length(which(Y[as.numeric(names(YestSVM1test))]==1&YestSVM1test==1)),length(which(Y[as.numeric(names(na.omit(YestLR1test)))]==0&na.omit(YestLR1test)==1)),length(which(Y[as.numeric(names(na.omit(YestLR1test)))]==1&na.omit(YestLR1test)==1)),length(which(dadostest$Y==0&YestADB1test==1)),length(which(dadostest$Y==1&YestADB1test==1)),
    length(which(dadostest$Y==0&YestNB2test==0)),length(which(dadostest$Y==1&YestNB2test==0)),RF2$test$confusion[1,1],RF2$test$confusion[2,1],length(which(Y[as.numeric(names(YestSVM2test))]==0&YestSVM2test==0)),length(which(Y[as.numeric(names(YestSVM2test))]==1&YestSVM2test==0)),length(which(Y[as.numeric(names(na.omit(YestLR2test)))]==0&na.omit(YestLR2test)==0)),length(which(Y[as.numeric(names(na.omit(YestLR2test)))]==1&na.omit(YestLR2test)==0)),length(which(dadostest$Y==0&YestADB2test==0)),length(which(dadostest$Y==1&YestADB2test==0)),
    length(which(dadostest$Y==0&YestNB2test==1)),length(which(dadostest$Y==1&YestNB2test==1)),RF2$test$confusion[1,2],RF2$test$confusion[2,2],length(which(Y[as.numeric(names(YestSVM2test))]==0&YestSVM2test==1)),length(which(Y[as.numeric(names(YestSVM2test))]==1&YestSVM2test==1)),length(which(Y[as.numeric(names(na.omit(YestLR2test)))]==0&na.omit(YestLR2test)==1)),length(which(Y[as.numeric(names(na.omit(YestLR2test)))]==1&na.omit(YestLR2test)==1)),length(which(dadostest$Y==0&YestADB2test==1)),length(which(dadostest$Y==1&YestADB2test==1)),
    length(which(dadostest$Y==0&YestNB3test==0)),length(which(dadostest$Y==1&YestNB3test==0)),RF3$test$confusion[1,1],RF3$test$confusion[2,1],length(which(Y[as.numeric(names(YestSVM3test))]==0&YestSVM3test==0)),length(which(Y[as.numeric(names(YestSVM3test))]==1&YestSVM3test==0)),length(which(Y[as.numeric(names(na.omit(YestLR3test)))]==0&na.omit(YestLR3test)==0)),length(which(Y[as.numeric(names(na.omit(YestLR3test)))]==1&na.omit(YestLR3test)==0)),length(which(dadostest$Y==0&YestADB3test==0)),length(which(dadostest$Y==1&YestADB3test==0)),
    length(which(dadostest$Y==0&YestNB3test==1)),length(which(dadostest$Y==1&YestNB3test==1)),RF3$test$confusion[1,2],RF3$test$confusion[2,2],length(which(Y[as.numeric(names(YestSVM3test))]==0&YestSVM3test==1)),length(which(Y[as.numeric(names(YestSVM3test))]==1&YestSVM3test==1)),length(which(Y[as.numeric(names(na.omit(YestLR3test)))]==0&na.omit(YestLR3test)==1)),length(which(Y[as.numeric(names(na.omit(YestLR3test)))]==1&na.omit(YestLR3test)==1)),length(which(dadostest$Y==0&YestADB3test==1)),length(which(dadostest$Y==1&YestADB3test==1)),
    length(which(dadostest$Y==0&YestNB4test==0)),length(which(dadostest$Y==1&YestNB4test==0)),RF4$test$confusion[1,1],RF4$test$confusion[2,1],length(which(Y[as.numeric(names(YestSVM4test))]==0&YestSVM4test==0)),length(which(Y[as.numeric(names(YestSVM4test))]==1&YestSVM4test==0)),length(which(Y[as.numeric(names(na.omit(YestLR4test)))]==0&na.omit(YestLR4test)==0)),length(which(Y[as.numeric(names(na.omit(YestLR4test)))]==1&na.omit(YestLR4test)==0)),length(which(dadostest$Y==0&YestADB4test==0)),length(which(dadostest$Y==1&YestADB4test==0)),
    length(which(dadostest$Y==0&YestNB4test==1)),length(which(dadostest$Y==1&YestNB4test==1)),RF4$test$confusion[1,2],RF4$test$confusion[2,2],length(which(Y[as.numeric(names(YestSVM4test))]==0&YestSVM4test==1)),length(which(Y[as.numeric(names(YestSVM4test))]==1&YestSVM4test==1)),length(which(Y[as.numeric(names(na.omit(YestLR4test)))]==0&na.omit(YestLR4test)==1)),length(which(Y[as.numeric(names(na.omit(YestLR4test)))]==1&na.omit(YestLR4test)==1)),length(which(dadostest$Y==0&YestADB4test==1)),length(which(dadostest$Y==1&YestADB4test==1))
    ),2*metodos,2*models)-matconftest)^2)/(B-1)
  
  
  j=j+1; print(j)
}
matdesvtest=round(sqrt(sdmatconftest),2)
matdesvtrei=round(sqrt(sdmatconftrei),2)

mattreifinal=cbind(matconftrei[,1],matdesvtrei[,1],matconftrei[,2],matdesvtrei[,2],matconftrei[,3],matdesvtrei[,3],matconftrei[,4],matdesvtrei[,4],matconftrei[,5],matdesvtrei[,5],matconftrei[,6],matdesvtrei[,6],matconftrei[,7],matdesvtrei[,7],matconftrei[,8],matdesvtrei[,8])
mattestfinal=cbind(matconftest[,1],matdesvtest[,1],matconftest[,2],matdesvtest[,2],matconftest[,3],matdesvtest[,3],matconftest[,4],matdesvtest[,4],matconftest[,5],matdesvtest[,5],matconftest[,6],matdesvtest[,6],matconftest[,7],matdesvtest[,7],matconftest[,8],matdesvtrei[,8])

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







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
library(nnet) 

###### DADOS
dados=read.table("Switzerlanddata.txt", na.strings = -9,header=T,dec=".",sep = "\t")
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
X=data.frame(PAM,IPPA,RC,IPPARC,HM,ALPHA,ALPHA2)
Y=factor(ifelse(num==0,0,1))

newdata = data.frame(cbind(Y,age,trestbps,restecg,thaldur,thaltime, thalach,thalrest,tpeakbps,tpeakbpd,dummy,trestbpd,oldpeak,lvx2,lvx3,lvx4,lvf,X))

##########################################
#          TRATAMENTO DAS VARIÁVEIS
#       ACP - COMPONENTES PRINCIPAIS
##########################################

pca1 = princomp(na.omit(newdata[,-1]),cor=T) # Matriz de correlação

# Componentes
y1 = pca1$scores[,1]
y2 = pca1$scores[,2]
y3 = pca1$scores[,3]
y4 = pca1$scores[,4]
y5 = pca1$scores[,5]
y6 = pca1$scores[,6]
y7 = pca1$scores[,7]
y8 = pca1$scores[,8]
y9 = pca1$scores[,9]
y10 = pca1$scores[,10]

#Y~y1+y2+y3+y4+y5+y6+y7+y8+y9+y10

dadoscomp=data.frame(y1,y2,y3,y4,y5,y6,y7,y8,y9,y10,newdata$Y[as.numeric(rownames(na.omit(newdata[,-1])))])
colnames(dadoscomp)=c("y1","y2","y3","y4","y5","y6","y7","y8","y9","y10","Y")

B=100
models=4
metodos=6

NBtrei=matrix(,B,models);colnames(acurNBtrei)=c("Acuracia","Sensibilidade","Especificidade","VPP")
DTtrei=matrix(,B,models);colnames(acurDTtrei)=c("Acuracia","Sensibilidade","Especificidade","VPP")
SVMtrei=matrix(,B,models);colnames(acurSVMtrei)=c("Acuracia","Sensibilidade","Especificidade","VPP")
LRtrei=matrix(,B,models);colnames(acurLRtrei)=c("Acuracia","Sensibilidade","Especificidade","VPP")
RNtrei=matrix(,B,models);colnames(RNtrei)=c("Acuracia","Sensibilidade","Especificidade","VPP")
ADBtrei=matrix(,B,models);colnames(acurADBtrei)=c("Acuracia","Sensibilidade","Especificidade","VPP")

NBtest=matrix(,B,models);colnames(acurNBtest)=c("Acuracia","Sensibilidade","Especificidade","VPP")
DTtest=matrix(,B,models);colnames(acurDTtest)=c("Acuracia","Sensibilidade","Especificidade","VPP")
SVMtest=matrix(,B,models);colnames(acurSVMtest)=c("Acuracia","Sensibilidade","Especificidade","VPP")
LRtest=matrix(,B,models);colnames(acurLRtest)=c("Acuracia","Sensibilidade","Especificidade","VPP")
RNtest=matrix(,B,models);colnames(RNtest)=c("Acuracia","Sensibilidade","Especificidade","VPP")
ADBtest=matrix(,B,models);colnames(acurADBtest)=c("Acuracia","Sensibilidade","Especificidade","VPP")

tempos=matrix(,B,metodos);colnames(tempoNB)=c("NB","DT","SVM","LR","RN","Adaboost")

k=0 #contador

set.seed(26071993)

for (i in 1:B) {
  
  treinamento=sample(nrow(dadoscomp),round(nrow(dadoscomp)*0.7))
  dadostrei=dadoscomp[treinamento,]
  dadostest=dadoscomp[-treinamento,]
  
  ##########################################
  #             CLASSIFICACAO
  #              NAIVE BAYES
  ##########################################
  
  tempo0NB1=proc.time()[3]
  modNB1=naiveBayes(Y~y1+y2+y3+y4+y5+y6+y7+y8+y9+y10,data = dadostrei)
  YestNB1trei=predict(modNB1,newdata = dadostrei);YestNB1test=predict(modNB1,newdata = dadostest)
  tempos[i,1]=proc.time()[3]-tempo0NB1
  
  ##########################################
  #             CLASSIFICACAO
  #           ARVORE DE DECISAO
  ##########################################
  
  tempo0DT1=proc.time()[3]
  RF1=randomForest(Y~y1+y2+y3+y4+y5+y6+y7+y8+y9+y10,data = dadostrei,xtest=dadostest[,-11],ytest=dadostest[,11],na.action = na.omit)
  tempos[i,2]=proc.time()[3]-tempo0DT1
  
  ##########################################
  #             CLASSIFICACAO
  #     MAQUINA DE VETORES DE SUPORTE
  ##########################################
  
  tempo0SVM1=proc.time()[3]
  modSVM1=svm(Y~y1+y2+y3+y4+y5+y6+y7+y8+y9+y10,data=dadostrei,scale=F,kernel="radial",cost=1000,epsilon=1.0e-12, sigma=5)
  YestSVM1test=predict(modSVM1,dadostest);YestSVM1trei=modSVM1$fitted
  tempos[i,3]=proc.time()[3]-tempo0SVM1
 
  ##########################################
  #             CLASSIFICACAO
  #     REGRESSAO LINEAR - LOGISTICA
  ##########################################
  
  tempo0LR1=proc.time()[3]
  modLR1=glm(Y~y1+y2+y3+y4+y5+y6+y7+y8+y9+y10,data=dadostrei,family = binomial(link = "logit"),na.action = na.omit)
  YestLR1test=ifelse(predict.glm(modLR1,newdata = dadostest,type = "response")<0.5,0,1);YestLR1trei=ifelse(predict.glm(modLR1,newdata = dadostrei,type = "response")<0.5,0,1)
  tempos[i,4]=proc.time()[3]-tempo0LR1
  
  ##########################################
  #             CLASSIFICACAO
  #           REDE NEURAL - MLP
  ##########################################
  
  tempo0RN1=proc.time()[3]
  modRN=nnet(dadostrei[,-11],as.numeric(dadostrei[,11]),size=5,linout=T,rang = 0.1,decay = 5e-2, maxit = 1000)
  YestRNtest=ifelse(predict(modRN,dadostest[,-11])<1.5,0,1);YestRNtrei=ifelse(predict(modRN,dadostrei[,-11])<1.5,0,1)
  tempos[i,5]=proc.time()[3]-tempo0RN1
  
  ##########################################
  #             CLASSIFICACAO
  #               ADABOOST
  ##########################################
  
  tempo0ADB1=proc.time()[3]
  modADB1=adaboost(Y~y1+y2+y3+y4+y5+y6+y7+y8+y9+y10,data=dadostrei,10)
  YestADB1test=predict(modADB1,newdata=dadostest)$class;YestADB1trei=predict(modADB1,newdata=dadostrei)$class
  tempos[i,6]=proc.time()[3]-tempo0ADB1
  
  

  

#############################
#   METRICAS DE DESEMPENHO  #
#############################

NBtrei[i,]=c((length(which(dadostrei$Y==0&YestNB1trei==0))+length(which(dadostrei$Y==1&YestNB1trei==1)))/length(dadostrei$Y)*100,length(which(dadostrei$Y==1&YestNB1trei==1))/(length(which(dadostrei$Y==1&YestNB1trei==1))+length(which(dadostrei$Y==1&YestNB1trei==0)))*100,length(which(dadostrei$Y==0&YestNB1trei==0))/(length(which(dadostrei$Y==0&YestNB1trei==0))+length(which(dadostrei$Y==0&YestNB1trei==1)))*100,length(which(dadostrei$Y==1&YestNB1trei==1))/(length(which(dadostrei$Y==1&YestNB1trei==1))+length(which(dadostrei$Y==0&YestNB1trei==1)))*100)
NBtest[i,]=c((length(which(dadostest$Y==0&YestNB1test==0))+length(which(dadostest$Y==1&YestNB1test==1)))/length(dadostest$Y)*100,length(which(dadostest$Y==1&YestNB1test==1))/(length(which(dadostest$Y==1&YestNB1test==1))+length(which(dadostest$Y==1&YestNB1test==0)))*100,length(which(dadostest$Y==0&YestNB1test==0))/(length(which(dadostest$Y==0&YestNB1test==0))+length(which(dadostest$Y==0&YestNB1test==1)))*100,length(which(dadostest$Y==1&YestNB1test==1))/(length(which(dadostest$Y==1&YestNB1test==1))+length(which(dadostest$Y==0&YestNB1test==1)))*100)
  
DTtrei[i,]=c((RF1$confusion[1,1]+RF1$confusion[2,2])/sum(RF1$confusion[,-3])*100,RF1$confusion[2,2]/(RF1$confusion[2,1]+RF1$confusion[2,2])*100,RF1$confusion[1,1]/(RF1$confusion[1,1]+RF1$confusion[1,2])*100,RF1$confusion[2,2]/(RF1$confusion[1,2]+RF1$confusion[2,2])*100)
DTtest[i,]=c((RF1$test$confusion[1,1]+RF1$test$confusion[2,2])/sum(RF1$test$confusion[,-3])*100,RF1$test$confusion[2,2]/(RF1$test$confusion[2,1]+RF1$test$confusion[2,2])*100,RF1$test$confusion[1,1]/(RF1$test$confusion[1,1]+RF1$test$confusion[1,2])*100,RF1$test$confusion[2,2]/(RF1$test$confusion[1,2]+RF1$test$confusion[2,2])*100)

SVMtrei[i,]=c((length(which(Y[as.numeric(names(YestSVM1trei))]==0&YestSVM1trei==0))+length(which(Y[as.numeric(names(YestSVM1trei))]==1&YestSVM1trei==1)))/length(YestSVM1trei)*100,length(which(Y[as.numeric(names(YestSVM1trei))]==1&YestSVM1trei==1))/(length(which(Y[as.numeric(names(YestSVM1trei))]==1&YestSVM1trei==0))+length(which(Y[as.numeric(names(YestSVM1trei))]==1&YestSVM1trei==1)))*100,length(which(Y[as.numeric(names(YestSVM1trei))]==0&YestSVM1trei==0))/(length(which(Y[as.numeric(names(YestSVM1trei))]==0&YestSVM1trei==0))+length(which(Y[as.numeric(names(YestSVM1trei))]==0&YestSVM1trei==1)))*100,length(which(Y[as.numeric(names(YestSVM1trei))]==1&YestSVM1trei==1))/(length(which(Y[as.numeric(names(YestSVM1trei))]==0&YestSVM1trei==1))+length(which(Y[as.numeric(names(YestSVM1trei))]==1&YestSVM1trei==1)))*100)
SVMtest[i,]=c((length(which(Y[as.numeric(names(YestSVM1test))]==0&YestSVM1test==0))+length(which(Y[as.numeric(names(YestSVM1test))]==1&YestSVM1test==1)))/length(YestSVM1test)*100,length(which(Y[as.numeric(names(YestSVM1test))]==1&YestSVM1test==1))/(length(which(Y[as.numeric(names(YestSVM1test))]==1&YestSVM1test==0))+length(which(Y[as.numeric(names(YestSVM1test))]==1&YestSVM1test==1)))*100,length(which(Y[as.numeric(names(YestSVM1test))]==0&YestSVM1test==0))/(length(which(Y[as.numeric(names(YestSVM1test))]==0&YestSVM1test==0))+length(which(Y[as.numeric(names(YestSVM1test))]==0&YestSVM1test==1)))*100,length(which(Y[as.numeric(names(YestSVM1test))]==1&YestSVM1test==1))/(length(which(Y[as.numeric(names(YestSVM1test))]==0&YestSVM1test==1))+length(which(Y[as.numeric(names(YestSVM1test))]==1&YestSVM1test==1)))*100)

LRtrei[i,]=c((length(which(Y[as.numeric(names(na.omit(YestLR1trei)))]==0&na.omit(YestLR1trei)==0))+length(which(Y[as.numeric(names(na.omit(YestLR1trei)))]==1&na.omit(YestLR1trei)==1)))/length(na.omit(YestLR1trei))*100,length(which(Y[as.numeric(names(na.omit(YestLR1trei)))]==1&na.omit(YestLR1trei)==1))/(length(which(Y[as.numeric(names(na.omit(YestLR1trei)))]==1&na.omit(YestLR1trei)==0))+length(which(Y[as.numeric(names(na.omit(YestLR1trei)))]==1&na.omit(YestLR1trei)==1)))*100,length(which(Y[as.numeric(names(na.omit(YestLR1trei)))]==0&na.omit(YestLR1trei)==0))/(length(which(Y[as.numeric(names(na.omit(YestLR1trei)))]==0&na.omit(YestLR1trei)==0))+length(which(Y[as.numeric(names(na.omit(YestLR1trei)))]==0&na.omit(YestLR1trei)==1)))*100,length(which(Y[as.numeric(names(na.omit(YestLR1trei)))]==1&na.omit(YestLR1trei)==1))/(length(which(Y[as.numeric(names(na.omit(YestLR1trei)))]==0&na.omit(YestLR1trei)==1))+length(which(Y[as.numeric(names(na.omit(YestLR1trei)))]==1&na.omit(YestLR1trei)==1)))*100)
LRtest[i,]=c((length(which(Y[as.numeric(names(na.omit(YestLR1test)))]==0&na.omit(YestLR1test)==0))+length(which(Y[as.numeric(names(na.omit(YestLR1test)))]==1&na.omit(YestLR1test)==1)))/length(na.omit(YestLR1test))*100,length(which(Y[as.numeric(names(na.omit(YestLR1test)))]==1&na.omit(YestLR1test)==1))/(length(which(Y[as.numeric(names(na.omit(YestLR1test)))]==1&na.omit(YestLR1test)==0))+length(which(Y[as.numeric(names(na.omit(YestLR1test)))]==1&na.omit(YestLR1test)==1)))*100,length(which(Y[as.numeric(names(na.omit(YestLR1test)))]==0&na.omit(YestLR1test)==0))/(length(which(Y[as.numeric(names(na.omit(YestLR1test)))]==0&na.omit(YestLR1test)==0))+length(which(Y[as.numeric(names(na.omit(YestLR1test)))]==0&na.omit(YestLR1test)==1)))*100,length(which(Y[as.numeric(names(na.omit(YestLR1test)))]==1&na.omit(YestLR1test)==1))/(length(which(Y[as.numeric(names(na.omit(YestLR1test)))]==0&na.omit(YestLR1test)==1))+length(which(Y[as.numeric(names(na.omit(YestLR1test)))]==1&na.omit(YestLR1test)==1)))*100)

RNtrei[i,]=c((length(which(dadostrei$Y==0&YestRNtrei==0))+length(which(dadostrei$Y==1&YestRNtrei==1)))/length(dadostrei$Y)*100,length(which(dadostrei$Y==1&YestRNtrei==1))/(length(which(dadostrei$Y==1&YestRNtrei==0))+length(which(dadostrei$Y==1&YestRNtrei==1)))*100,length(which(dadostrei$Y==0&YestRNtrei==0))/(length(which(dadostrei$Y==0&YestRNtrei==0))+length(which(dadostrei$Y==0&YestRNtrei==1)))*100,length(which(dadostrei$Y==1&YestRNtrei==1))/(length(which(dadostrei$Y==0&YestRNtrei==1))+length(which(dadostrei$Y==1&YestRNtrei==1)))*100)
RNtest[i,]=c((length(which(dadostest$Y==0&YestRNtest==0))+length(which(dadostest$Y==1&YestRNtest==1)))/length(dadostest$Y)*100,length(which(dadostest$Y==1&YestRNtest==1))/(length(which(dadostest$Y==1&YestRNtest==0))+length(which(dadostest$Y==1&YestRNtest==1)))*100,length(which(dadostest$Y==0&YestRNtest==0))/(length(which(dadostest$Y==0&YestRNtest==0))+length(which(dadostest$Y==0&YestRNtest==1)))*100,length(which(dadostest$Y==1&YestRNtest==1))/(length(which(dadostest$Y==0&YestRNtest==1))+length(which(dadostest$Y==1&YestRNtest==1)))*100)

ADBtrei[i,]=c((length(which(dadostrei$Y==0&YestADB1trei==0))+length(which(dadostrei$Y==1&YestADB1trei==1)))/length(dadostrei$Y)*100,length(which(dadostrei$Y==1&YestADB1trei==1))/(length(which(dadostrei$Y==1&YestADB1trei==0))+length(which(dadostrei$Y==1&YestADB1trei==1)))*100,length(which(dadostrei$Y==0&YestADB1trei==0))/(length(which(dadostrei$Y==0&YestADB1trei==0))+length(which(dadostrei$Y==0&YestADB1trei==1)))*100,length(which(dadostrei$Y==1&YestADB1trei==1))/(length(which(dadostrei$Y==0&YestADB1trei==1))+length(which(dadostrei$Y==1&YestADB1trei==1)))*100)
ADBtest[i,]=c((length(which(dadostest$Y==0&YestADB1test==0))+length(which(dadostest$Y==1&YestADB1test==1)))/length(dadostest$Y)*100,length(which(dadostest$Y==1&YestADB1test==1))/(length(which(dadostest$Y==1&YestADB1test==0))+length(which(dadostest$Y==1&YestADB1test==1)))*100,length(which(dadostest$Y==0&YestADB1test==0))/(length(which(dadostest$Y==0&YestADB1test==0))+length(which(dadostest$Y==0&YestADB1test==1)))*100,length(which(dadostest$Y==1&YestADB1test==1))/(length(which(dadostest$Y==0&YestADB1test==1))+length(which(dadostest$Y==1&YestADB1test==1)))*100)


k=k+1; print(k)
}


miacuraciastrei=rbind(apply(na.omit(NBtrei),2,mean),apply(na.omit(DTtrei),2,mean),apply(na.omit(SVMtrei),2,mean),apply(na.omit(LRtrei),2,mean),apply(na.omit(RNtrei),2,mean),apply(na.omit(ADBtrei),2,mean))
rownames(miacuraciastrei)=c("NB","DT","SVM","LR","RN","Adaboost")
miacuraciastest=rbind(apply(na.omit(NBtest),2,mean),apply(na.omit(DTtest),2,mean),apply(na.omit(SVMtest),2,mean),apply(na.omit(LRtest),2,mean),apply(na.omit(RNtest),2,mean),apply(na.omit(ADBtest),2,mean))
rownames(miacuraciastest)=c("NB","DT","SVM","LR","RN","Adaboost")


sdacuraciastrei=rbind(apply(na.omit(NBtrei),2,sd),apply(na.omit(DTtrei),2,sd),apply(na.omit(SVMtrei),2,sd),apply(na.omit(LRtrei),2,sd),apply(na.omit(RNtrei),2,sd),apply(na.omit(ADBtrei),2,sd))
rownames(sdacuraciastrei)=c("NB","DT","SVM","LR","RN","Adaboost")
sdacuraciastest=rbind(apply(na.omit(NBtest),2,sd),apply(na.omit(DTtest),2,sd),apply(na.omit(SVMtest),2,sd),apply(na.omit(LRtest),2,sd),apply(na.omit(RNtest),2,sd),apply(na.omit(ADBtest),2,sd))
rownames(sdacuraciastest)=c("NB","DT","SVM","LR","RN","Adaboost")

mitempo=apply(tempos,2,mean)
sdtempo=apply(tempos,2,sd)
tempofinal=cbind(mitempo[1],sdtempo[1],mitempo[2],sdtempo[2],mitempo[3],sdtempo[3],mitempo[4],sdtempo[4],mitempo[5],sdtempo[5],mitempo[6],sdtempo[6])
colnames(tempofinal)=c("NB","NB","DT","DT","SVM","SVM","LR","LR","RN","RN","Adaboost","Adaboost")

tabelafinaltrei=cbind(miacuraciastrei[,1],sdacuraciastrei[,1],miacuraciastrei[,2],sdacuraciastrei[,2],miacuraciastrei[,3],sdacuraciastrei[,3],miacuraciastrei[,4],sdacuraciastrei[,4])
tabelafinaltest=cbind(miacuraciastest[,1],sdacuraciastest[,1],miacuraciastest[,2],sdacuraciastest[,2],miacuraciastest[,3],sdacuraciastest[,3],miacuraciastest[,4],sdacuraciastest[,4])
rownames(tabelafinaltrei)=c("NB","DT","SVM","LR","RN","Adaboost")
rownames(tabelafinaltest)=c("NB","DT","SVM","LR","RN","Adaboost")

round(tabelafinaltrei,2) #ACURACIAS TREINAMENTO
round(tabelafinaltest,2) #ACURACIAS TESTE
round(tempofinal,2) #TEMPOS DE PROCESSAMENTO

xtable(tabelafinaltrei,digits=2) #ACURACIAS TREINAMENTO
xtable(tabelafinaltest,digits=2) #ACURACIAS TESTE
xtable(tempofinal*1000,digits=2) #TEMPOS DE PROCESSAMENTO






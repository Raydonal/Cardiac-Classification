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
dados=read.table("LongBeachdata.txt",
                 na.strings = -9,header=T,dec=".",sep = "\t")
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

newdata = data.frame(cbind(age,sex,painexer,relrest,trestbps,htn,chol,smoke,cigs,years,fbs,dm,famhist,restecg,dig,prop,nitr,pro,diuretic,proto,thaldur,thaltime,met,thalach,thalrest,tpeakbps,tpeakbpd,dummy,trestbpd,exang,oldpeak,slope,rldv5,rldv5e,restef,restwm,thal,thalsev,lmt,ladprox,laddist,diag,cxmain,ramus,om1,om2,rcaprox,rcadist,lvx2,lvx3,lvx4,lvf,cathef,junk,X,Y))

B=100
models=5
metodos=5

tempos=matrix(,B,3);colnames(tempos)=c("RF","Adaboost","RF")
Longtrei1=matrix(,B,4);colnames(Longtrei1)=c("Acuracia","Sensibilidade","Especificidade","VPP")
Longtest1=matrix(,B,4);colnames(Longtest1)=c("Acuracia","Sensibilidade","Especificidade","VPP")
Longtrei2=matrix(,B,4);colnames(Longtrei2)=c("Acuracia","Sensibilidade","Especificidade","VPP")
Longtest2=matrix(,B,4);colnames(Longtest2)=c("Acuracia","Sensibilidade","Especificidade","VPP")
Longtrei3=matrix(,B,4);colnames(Longtrei3)=c("Acuracia","Sensibilidade","Especificidade","VPP")
Longtest3=matrix(,B,4);colnames(Longtest3)=c("Acuracia","Sensibilidade","Especificidade","VPP")

acurNBtrei=matrix(,B,models);colnames(acurNBtrei)=c("A","B","C","D","E")
acurDTtrei=matrix(,B,models);colnames(acurDTtrei)=c("A","B","C","D","E")
acurSVMtrei=matrix(,B,models);colnames(acurSVMtrei)=c("A","B","C","D","E")
acurLRtrei=matrix(,B,models);colnames(acurLRtrei)=c("A","B","C","D","E")
acurADBtrei=matrix(,B,models);colnames(acurADBtrei)=c("A","B","C","D","E")

acurNBtest=matrix(,B,models);colnames(acurNBtest)=c("A","B","C","D","E")
acurDTtest=matrix(,B,models);colnames(acurDTtest)=c("A","B","C","D","E")
acurSVMtest=matrix(,B,models);colnames(acurSVMtest)=c("A","B","C","D","E")
acurLRtest=matrix(,B,models);colnames(acurLRtest)=c("A","B","C","D","E")
acurADBtest=matrix(,B,models);colnames(acurADBtest)=c("A","B","C","D","E")

matconftrei=matrix(0,2*metodos,2*models);colnames(matconftrei)=rep(c(0,1),models);rownames(matconftrei)=rep(c(0,1),metodos)
matconftest=matrix(0,2*metodos,2*models);colnames(matconftest)=rep(c(0,1),models);rownames(matconftest)=rep(c(0,1),metodos)
k=0 #contador

# 1 - Y~sex+painexer+relrest+htn+smoke+fbs+famhist+dig+prop+nitr+pro+diuretic+exang+slope+ladprox+laddist+cxmain+om1+rcaprox+junk
# 2 - Y~sex+painexer+relrest+htn+chol+cigs+years+restecg+proto+lmt+ladprox+diag+cxmain+ramus+om1+rcadist+lvx2+lvx3+ALPHA
# 3 - Y~sex+painexer+rldv5e+ladprox+laddist
# 4 - Y~age+sex+painexer+relrest+trestbps+htn+chol+smoke+cigs+years+fbs+famhist+restecg+proto+thaldur+met+thalach+thalrest+trestbpd+exang+oldpeak+rldv5+rldv5e+lmt+ladprox+laddist+diag+cxmain+om1+om2+rcaprox+rcadist+lvx2+lvx3+lvx4+lvf+cathef+PAM+IPPA+RC+IPPARC+HM+ALPHA+ALPHA2
# 5 - Y~sex+painexer+relrest+htn+cigs+famhist+proto+thalrest+rldv5+lmt+laddist+cxmain+om1+om2+rcaprox+lvx3+lvf+PAM+IPPA

set.seed(26071993)

for (i in 1:B) {
  
treinamento=sample(nrow(newdata),round(nrow(newdata)*0.7))
dadostrei=newdata[treinamento,]
dadostest=newdata[-treinamento,]

##########################################
#             CLASSIFICACAO
#              NAIVE BAYES
##########################################

modNB1=naiveBayes(Y~sex+painexer+relrest+htn+smoke+fbs+famhist+dig+prop+nitr+pro+diuretic+exang+slope+ladprox+laddist+cxmain+om1+rcaprox+junk,data = dadostrei)
YestNB1trei=predict(modNB1,newdata = dadostrei);YestNB1test=predict(modNB1,newdata = dadostest)

modNB2=naiveBayes(Y~sex+painexer+relrest+htn+chol+cigs+years+restecg+proto+lmt+ladprox+diag+cxmain+ramus+om1+rcadist+lvx2+lvx3+ALPHA,data = dadostrei);YestNB2=predict(modNB2,newdata = dadostest)
YestNB2trei=predict(modNB2,newdata = dadostrei);YestNB2test=predict(modNB2,newdata = dadostest)

modNB3=naiveBayes(Y~sex+painexer+rldv5e+ladprox+laddist,data = dadostrei) #### MELHOR
YestNB3trei=predict(modNB3,newdata = dadostrei);YestNB3test=predict(modNB3,newdata = dadostest)

modNB4=naiveBayes(Y~age+sex+painexer+relrest+trestbps+htn+chol+smoke+cigs+years+fbs+famhist+restecg+proto+thaldur+met+thalach+thalrest+trestbpd+exang+oldpeak+rldv5+rldv5e+lmt+ladprox+laddist+diag+cxmain+om1+om2+rcaprox+rcadist+lvx2+lvx3+lvx4+lvf+cathef+PAM+IPPA+RC+IPPARC+HM+ALPHA+ALPHA2,data = dadostrei)
YestNB4trei=predict(modNB4,newdata = dadostrei);YestNB4test=predict(modNB4,newdata = dadostest)

modNB5=naiveBayes(Y~sex+painexer+relrest+htn+cigs+famhist+proto+thalrest+rldv5+lmt+laddist+cxmain+om1+om2+rcaprox+lvx3+lvf+PAM+IPPA,data = dadostrei)
YestNB5trei=predict(modNB5,newdata = dadostrei);YestNB5test=predict(modNB5,newdata = dadostest)

##########################################
#             CLASSIFICACAO
#           ARVORE DE DECISAO
##########################################

DadostestRF1=na.omit(data.frame(dadostest$Y,dadostest$sex,dadostest$painexer,dadostest$relrest,dadostest$htn,dadostest$smoke,dadostest$fbs,dadostest$famhist,dadostest$dig,dadostest$prop,dadostest$nitr,dadostest$pro,dadostest$diuretic,dadostest$exang,dadostest$slope,dadostest$ladprox,dadostest$laddist,dadostest$cxmain,dadostest$om1,dadostest$rcaprox,dadostest$junk))
RF1=randomForest(Y~sex+painexer+relrest+htn+smoke+fbs+famhist+dig+prop+nitr+pro+diuretic+exang+slope+ladprox+laddist+cxmain+om1+rcaprox+junk,data = dadostrei,xtest=DadostestRF1[,-1],ytest=DadostestRF1$dadostest.Y,na.action = na.omit)

DadostestRF2=na.omit(data.frame(dadostest$Y,dadostest$sex,dadostest$painexer,dadostest$relrest,dadostest$htn,dadostest$chol,dadostest$cigs,dadostest$years,dadostest$restecg,dadostest$proto,dadostest$lmt,dadostest$ladprox,dadostest$diag,dadostest$cxmain,dadostest$ramus,dadostest$om1,dadostest$rcadist,dadostest$lvx2,dadostest$lvx3,dadostest$ALPHA))
RF2=randomForest(Y~sex+painexer+relrest+htn+chol+cigs+years+restecg+proto+lmt+ladprox+diag+cxmain+ramus+om1+rcadist+lvx2+lvx3+ALPHA,data = dadostrei,xtest=DadostestRF2[,-1],ytest=DadostestRF2$dadostest.Y,na.action = na.omit)

DadostestRF3=na.omit(data.frame(dadostest$Y,dadostest$sex,dadostest$painexer,dadostest$rldv5e,dadostest$ladprox,dadostest$laddist))
RF3=randomForest(Y~sex+painexer+rldv5e+ladprox+laddist,data = dadostrei,xtest=DadostestRF3[,-1],ytest=DadostestRF3$dadostest.Y,na.action = na.omit)

DadostestRF4=na.omit(data.frame(dadostest$Y,dadostest$age,dadostest$sex,dadostest$painexer,dadostest$relrest,dadostest$trestbps,dadostest$htn,dadostest$chol,dadostest$smoke,dadostest$cigs,dadostest$years,dadostest$fbs,dadostest$famhist,dadostest$restecg,dadostest$proto,dadostest$thaldur,dadostest$met,dadostest$thalach,dadostest$thalrest,dadostest$trestbpd,dadostest$exang,dadostest$oldpeak,dadostest$rldv5,dadostest$rldv5e,dadostest$lmt,dadostest$ladprox,dadostest$laddist,dadostest$diag,dadostest$cxmain,dadostest$om1,dadostest$om2,dadostest$rcaprox,dadostest$rcadist,dadostest$lvx2,dadostest$lvx3,dadostest$lvx4,dadostest$lvf,dadostest$cathef,dadostest$PAM,dadostest$IPPA,dadostest$RC,dadostest$IPPARC,dadostest$HM,dadostest$ALPHA,dadostest$ALPHA2))
RF4=randomForest(Y~age+sex+painexer+relrest+trestbps+htn+chol+smoke+cigs+years+fbs+famhist+restecg+proto+thaldur+met+thalach+thalrest+trestbpd+exang+oldpeak+rldv5+rldv5e+lmt+ladprox+laddist+diag+cxmain+om1+om2+rcaprox+rcadist+lvx2+lvx3+lvx4+lvf+cathef+PAM+IPPA+RC+IPPARC+HM+ALPHA+ALPHA2,data = dadostrei,xtest=DadostestRF4[,-1],ytest=DadostestRF4$dadostest.Y,na.action = na.omit)

DadostestRF5=na.omit(data.frame(dadostest$Y,dadostest$sex,dadostest$painexer,dadostest$relrest,dadostest$htn,dadostest$cigs,dadostest$famhist,dadostest$proto,dadostest$thalrest,dadostest$rldv5,dadostest$lmt,dadostest$laddist,dadostest$cxmain,dadostest$om1,dadostest$om2,dadostest$rcaprox,dadostest$lvx3,dadostest$lvf,dadostest$PAM,dadostest$IPPA))
RF5=randomForest(Y~sex+painexer+relrest+htn+cigs+famhist+proto+thalrest+rldv5+lmt+laddist+cxmain+om1+om2+rcaprox+lvx3+lvf+PAM+IPPA,data = dadostrei,xtest=DadostestRF5[,-1],ytest=DadostestRF5$dadostest.Y,na.action = na.omit)

##########################################
#             CLASSIFICACAO
#     MAQUINA DE VETORES DE SUPORTE
##########################################

modSVM1=svm(Y~sex+painexer+relrest+htn+smoke+fbs+famhist+dig+prop+nitr+pro+diuretic+exang+slope+ladprox+laddist+cxmain+om1+rcaprox+junk,data=dadostrei,scale=F,kernel="poly",cost=100,epsilon=1.0e-12)
YestSVM1test=predict(modSVM1,dadostest);YestSVM1trei=modSVM1$fitted

modSVM2=svm(Y~sex+painexer+relrest+htn+chol+cigs+years+restecg+proto+lmt+ladprox+diag+cxmain+ramus+om1+rcadist+lvx2+lvx3+ALPHA,data=dadostrei,scale=F,kernel="poly",cost=100,epsilon=1.0e-12)
YestSVM2test=predict(modSVM2,dadostest);YestSVM2trei=modSVM2$fitted

modSVM3=svm(Y~sex+painexer+rldv5e+ladprox+laddist,scale=F,data=dadostrei,kernel="poly",cost=100,epsilon=1.0e-12)
YestSVM3test=predict(modSVM3,dadostest);YestSVM3trei=modSVM3$fitted

modSVM4=svm(Y~age+sex+painexer+relrest+trestbps+htn+chol+smoke+cigs+years+fbs+famhist+restecg+proto+thaldur+met+thalach+thalrest+trestbpd+exang+oldpeak+rldv5+rldv5e+lmt+ladprox+laddist+diag+cxmain+om1+om2+rcaprox+rcadist+lvx2+lvx3+lvx4+lvf+cathef+PAM+IPPA+RC+IPPARC+HM+ALPHA+ALPHA2,data=dadostrei,scale=F,kernel="poly",cost=100,epsilon=1.0e-12)
YestSVM4test=predict(modSVM4,dadostest);YestSVM4trei=modSVM4$fitted

modSVM5=svm(Y~sex+painexer+relrest+htn+cigs+famhist+proto+thalrest+rldv5+lmt+laddist+cxmain+om1+om2+rcaprox+lvx3+lvf+PAM+IPPA,data=dadostrei,scale=F,kernel="poly",cost=100,epsilon=1.0e-12)
YestSVM5test=predict(modSVM5,dadostest);YestSVM5trei=modSVM5$fitted

##########################################
#             CLASSIFICACAO
#     REGRESSAO LINEAR - LOGISTICA
##########################################

modLR1=glm(Y~painexer+relrest+htn+smoke+fbs+famhist+dig+prop+nitr+pro+diuretic+exang+slope+ladprox+laddist+cxmain+om1+rcaprox+junk,data=dadostrei,family = binomial(link = "logit"),na.action = na.omit)
YestLR1test=ifelse(predict.glm(modLR1,newdata = dadostest,type = "response")<0.5,0,1);YestLR1trei=ifelse(predict.glm(modLR1,newdata = dadostrei,type = "response")<0.5,0,1)

modLR2=glm(Y~painexer+relrest+htn+chol+cigs+years+restecg+proto+lmt+ladprox+diag+cxmain+ramus+om1+rcadist+lvx2+lvx3+ALPHA,data=dadostrei,family = binomial(link = "logit"))
YestLR2test=ifelse(predict.glm(modLR2,newdata = dadostest,type = "response")<0.5,0,1);YestLR2trei=ifelse(predict.glm(modLR2,newdata = dadostrei,type = "response")<0.5,0,1)

modLR3=glm(Y~painexer+rldv5e+ladprox+laddist,data=dadostrei,family = binomial(link = "logit"))
YestLR3test=ifelse(predict.glm(modLR3,newdata = dadostest,type = "response")<0.5,0,1);YestLR3trei=ifelse(predict.glm(modLR3,newdata = dadostrei,type = "response")<0.5,0,1)

modLR4=glm(Y~age+painexer+relrest+trestbps+htn+chol+smoke+cigs+years+fbs+famhist+restecg+proto+thaldur+met+thalach+thalrest+trestbpd+exang+oldpeak+rldv5+rldv5e+lmt+ladprox+laddist+diag+cxmain+om1+om2+rcaprox+rcadist+lvx2+lvx3+lvx4+lvf+cathef+PAM+IPPA+RC+IPPARC+HM+ALPHA+ALPHA2,data=dadostrei,family = binomial(link = "logit"))
YestLR4test=ifelse(predict.glm(modLR4,newdata = dadostest,type = "response")<0.5,0,1);YestLR4trei=ifelse(predict.glm(modLR4,newdata = dadostrei,type = "response")<0.5,0,1)

tempo0Long3=proc.time()[3]
modLR5=glm(Y~painexer+relrest+htn+cigs+famhist+proto+thalrest+rldv5+lmt+laddist+cxmain+om1+om2+rcaprox+lvx3+lvf+PAM+IPPA,data=dadostrei,family = binomial(link = "logit"))
YestLR5test=ifelse(predict.glm(modLR5,newdata = dadostest,type = "response")<0.5,0,1);YestLR5trei=ifelse(predict.glm(modLR5,newdata = dadostrei,type = "response")<0.5,0,1)
tempos[i,3]=proc.time()[3]-tempo0Long3

##########################################
#             CLASSIFICACAO
#               ADABOOST
##########################################

modADB1=adaboost(Y~sex+painexer+relrest+htn+smoke+fbs+famhist+dig+prop+nitr+pro+diuretic+exang+slope+ladprox+laddist+cxmain+om1+rcaprox+junk,data=dadostrei,10)
YestADB1test=predict(modADB1,newdata=dadostest)$class;YestADB1trei=predict(modADB1,newdata=dadostrei)$class

tempo0Long1=proc.time()[3]
modADB2=adaboost(Y~sex+painexer+relrest+htn+chol+cigs+years+restecg+proto+lmt+ladprox+diag+cxmain+ramus+om1+rcadist+lvx2+lvx3+ALPHA,data=dadostrei,10)
YestADB2test=predict(modADB2,newdata=dadostest)$class;YestADB2trei=predict(modADB2,newdata=dadostrei)$class
tempos[i,1]=proc.time()[3]-tempo0Long1

tempo0Long2=proc.time()[3]
modADB3=adaboost(Y~sex+painexer+rldv5e+ladprox+laddist,data=dadostrei,10)
YestADB3test=predict(modADB3,newdata=dadostest)$class;YestADB3trei=predict(modADB3,newdata=dadostrei)$class
tempos[i,2]=proc.time()[3]-tempo0Long2

modADB4=adaboost(Y~age+sex+painexer+relrest+trestbps+htn+chol+smoke+cigs+years+fbs+famhist+restecg+proto+thaldur+met+thalach+thalrest+trestbpd+exang+oldpeak+rldv5+rldv5e+lmt+ladprox+laddist+diag+cxmain+om1+om2+rcaprox+rcadist+lvx2+lvx3+lvx4+lvf+cathef+PAM+IPPA+RC+IPPARC+HM+ALPHA+ALPHA2,data=dadostrei,10)
YestADB4test=predict(modADB4,newdata=dadostest)$class;YestADB4trei=predict(modADB4,newdata=dadostrei)$class

modADB5=adaboost(Y~sex+painexer+relrest+htn+cigs+famhist+proto+thalrest+rldv5+lmt+laddist+cxmain+om1+om2+rcaprox+lvx3+lvf+PAM+IPPA,data=dadostrei,10)
YestADB5test=predict(modADB5,newdata=dadostest)$class;YestADB5trei=predict(modADB5,newdata=dadostrei)$class


#############################
#         ACURACIAS         #
#############################

acurNBtrei[i,]=c((length(which(dadostrei$Y==0&YestNB1trei==0))+length(which(dadostrei$Y==1&YestNB1trei==1)))/length(dadostrei$Y)*100,(length(which(dadostrei$Y==0&YestNB2trei==0))+length(which(dadostrei$Y==1&YestNB2trei==1)))/length(dadostrei$Y)*100,(length(which(dadostrei$Y==0&YestNB3trei==0))+length(which(dadostrei$Y==1&YestNB3trei==1)))/length(dadostrei$Y)*100,(length(which(dadostrei$Y==0&YestNB4trei==0))+length(which(dadostrei$Y==1&YestNB4trei==1)))/length(dadostrei$Y)*100,(length(which(dadostrei$Y==0&YestNB5trei==0))+length(which(dadostrei$Y==1&YestNB5trei==1)))/length(dadostrei$Y)*100)
acurNBtest[i,]=c((length(which(dadostest$Y==0&YestNB1test==0))+length(which(dadostest$Y==1&YestNB1test==1)))/length(dadostest$Y)*100,(length(which(dadostest$Y==0&YestNB2test==0))+length(which(dadostest$Y==1&YestNB2test==1)))/length(dadostest$Y)*100,(length(which(dadostest$Y==0&YestNB3test==0))+length(which(dadostest$Y==1&YestNB3test==1)))/length(dadostest$Y)*100,(length(which(dadostest$Y==0&YestNB4test==0))+length(which(dadostest$Y==1&YestNB4test==1)))/length(dadostest$Y)*100,(length(which(dadostest$Y==0&YestNB5test==0))+length(which(dadostest$Y==1&YestNB5test==1)))/length(dadostest$Y)*100)

acurDTtrei[i,]=c((RF1$confusion[1,1]+RF1$confusion[2,2])/sum(RF1$confusion[,-3])*100,(RF2$confusion[1,1]+RF2$confusion[2,2])/sum(RF2$confusion[,-3])*100,(RF3$confusion[1,1]+RF3$confusion[2,2])/sum(RF3$confusion[,-3])*100,(RF4$confusion[1,1]+RF4$confusion[2,2])/sum(RF4$confusion[,-3])*100,(RF5$confusion[1,1]+RF5$confusion[2,2])/sum(RF5$confusion[,-3])*100)
acurDTtest[i,]=c((RF1$test$confusion[1,1]+RF1$test$confusion[2,2])/sum(RF1$test$confusion[,-3])*100,(RF2$test$confusion[1,1]+RF2$test$confusion[2,2])/sum(RF2$test$confusion[,-3])*100,(RF3$test$confusion[1,1]+RF3$test$confusion[2,2])/sum(RF3$test$confusion[,-3])*100,(RF4$test$confusion[1,1]+RF4$test$confusion[2,2])/sum(RF4$test$confusion[,-3])*100,(RF5$test$confusion[1,1]+RF5$test$confusion[2,2])/sum(RF5$test$confusion[,-3])*100)

acurSVMtrei[i,]=c((length(which(Y[as.numeric(names(YestSVM1trei))]==0&YestSVM1trei==0))+length(which(Y[as.numeric(names(YestSVM1trei))]==1&YestSVM1trei==1)))/length(YestSVM1trei)*100,(length(which(Y[as.numeric(names(YestSVM2trei))]==0&YestSVM2trei==0))+length(which(Y[as.numeric(names(YestSVM2trei))]==1&YestSVM2trei==1)))/length(YestSVM2trei)*100,(length(which(Y[as.numeric(names(YestSVM3trei))]==0&YestSVM3trei==0))+length(which(Y[as.numeric(names(YestSVM3trei))]==1&YestSVM3trei==1)))/length(YestSVM3trei)*100,(length(which(Y[as.numeric(names(YestSVM4trei))]==0&YestSVM4trei==0))+length(which(Y[as.numeric(names(YestSVM4trei))]==1&YestSVM4trei==1)))/length(YestSVM4trei)*100,(length(which(Y[as.numeric(names(YestSVM5trei))]==0&YestSVM5trei==0))+length(which(Y[as.numeric(names(YestSVM5trei))]==1&YestSVM5trei==1)))/length(YestSVM5trei)*100)
acurSVMtest[i,]=c((length(which(Y[as.numeric(names(YestSVM1test))]==0&YestSVM1test==0))+length(which(Y[as.numeric(names(YestSVM1test))]==1&YestSVM1test==1)))/length(YestSVM1test)*100,(length(which(Y[as.numeric(names(YestSVM2test))]==0&YestSVM2test==0))+length(which(Y[as.numeric(names(YestSVM2test))]==1&YestSVM2test==1)))/length(YestSVM2test)*100,(length(which(Y[as.numeric(names(YestSVM3test))]==0&YestSVM3test==0))+length(which(Y[as.numeric(names(YestSVM3test))]==1&YestSVM3test==1)))/length(YestSVM3test)*100,(length(which(Y[as.numeric(names(YestSVM4test))]==0&YestSVM4test==0))+length(which(Y[as.numeric(names(YestSVM4test))]==1&YestSVM4test==1)))/length(YestSVM4test)*100,(length(which(Y[as.numeric(names(YestSVM5test))]==0&YestSVM5test==0))+length(which(Y[as.numeric(names(YestSVM5test))]==1&YestSVM5test==1)))/length(YestSVM5test)*100)

acurLRtrei[i,]=c((length(which(Y[as.numeric(names(na.omit(YestLR1trei)))]==0&na.omit(YestLR1trei)==0))+length(which(Y[as.numeric(names(na.omit(YestLR1trei)))]==1&na.omit(YestLR1trei)==1)))/length(na.omit(YestLR1trei))*100,(length(which(Y[as.numeric(names(na.omit(YestLR2trei)))]==0&na.omit(YestLR2trei)==0))+length(which(Y[as.numeric(names(na.omit(YestLR2trei)))]==1&na.omit(YestLR2trei)==1)))/length(na.omit(YestLR2trei))*100,(length(which(Y[as.numeric(names(na.omit(YestLR3trei)))]==0&na.omit(YestLR3trei)==0))+length(which(Y[as.numeric(names(na.omit(YestLR3trei)))]==1&na.omit(YestLR3trei)==1)))/length(na.omit(YestLR3trei))*100,(length(which(Y[as.numeric(names(na.omit(YestLR4trei)))]==0&na.omit(YestLR4trei)==0))+length(which(Y[as.numeric(names(na.omit(YestLR4trei)))]==1&na.omit(YestLR4trei)==1)))/length(na.omit(YestLR4trei))*100,(length(which(Y[as.numeric(names(na.omit(YestLR5trei)))]==0&na.omit(YestLR5trei)==0))+length(which(Y[as.numeric(names(na.omit(YestLR5trei)))]==1&na.omit(YestLR5trei)==1)))/length(na.omit(YestLR5trei))*100)
acurLRtest[i,]=c((length(which(Y[as.numeric(names(na.omit(YestLR1test)))]==0&na.omit(YestLR1test)==0))+length(which(Y[as.numeric(names(na.omit(YestLR1test)))]==1&na.omit(YestLR1test)==1)))/length(na.omit(YestLR1test))*100,(length(which(Y[as.numeric(names(na.omit(YestLR2test)))]==0&na.omit(YestLR2test)==0))+length(which(Y[as.numeric(names(na.omit(YestLR2test)))]==1&na.omit(YestLR2test)==1)))/length(na.omit(YestLR2test))*100,(length(which(Y[as.numeric(names(na.omit(YestLR3test)))]==0&na.omit(YestLR3test)==0))+length(which(Y[as.numeric(names(na.omit(YestLR3test)))]==1&na.omit(YestLR3test)==1)))/length(na.omit(YestLR3test))*100,(length(which(Y[as.numeric(names(na.omit(YestLR4test)))]==0&na.omit(YestLR4test)==0))+length(which(Y[as.numeric(names(na.omit(YestLR4test)))]==1&na.omit(YestLR4test)==1)))/length(na.omit(YestLR4test))*100,(length(which(Y[as.numeric(names(na.omit(YestLR5test)))]==0&na.omit(YestLR5test)==0))+length(which(Y[as.numeric(names(na.omit(YestLR5test)))]==1&na.omit(YestLR5test)==1)))/length(na.omit(YestLR5test))*100)

acurADBtrei[i,]=c((length(which(dadostrei$Y==0&YestADB1trei==0))+length(which(dadostrei$Y==1&YestADB1trei==1)))/length(dadostrei$Y)*100,(length(which(dadostrei$Y==0&YestADB2trei==0))+length(which(dadostrei$Y==1&YestADB2trei==1)))/length(dadostrei$Y)*100,(length(which(dadostrei$Y==0&YestADB3trei==0))+length(which(dadostrei$Y==1&YestADB3trei==1)))/length(dadostrei$Y)*100,(length(which(dadostrei$Y==0&YestADB4trei==0))+length(which(dadostrei$Y==1&YestADB4trei==1)))/length(dadostrei$Y)*100,(length(which(dadostrei$Y==0&YestADB5trei==0))+length(which(dadostrei$Y==1&YestADB5trei==1)))/length(dadostrei$Y)*100)
acurADBtest[i,]=c((length(which(dadostest$Y==0&YestADB1test==0))+length(which(dadostest$Y==1&YestADB1test==1)))/length(dadostest$Y)*100,(length(which(dadostest$Y==0&YestADB2test==0))+length(which(dadostest$Y==1&YestADB2test==1)))/length(dadostest$Y)*100,(length(which(dadostest$Y==0&YestADB3test==0))+length(which(dadostest$Y==1&YestADB3test==1)))/length(dadostest$Y)*100,(length(which(dadostest$Y==0&YestADB4test==0))+length(which(dadostest$Y==1&YestADB4test==1)))/length(dadostest$Y)*100,(length(which(dadostest$Y==0&YestADB5test==0))+length(which(dadostest$Y==1&YestADB5test==1)))/length(dadostest$Y)*100)

Longtrei1[i,]=c((length(which(dadostrei$Y==0&YestADB2trei==0))+length(which(dadostrei$Y==1&YestADB2trei==1)))/length(dadostrei$Y)*100,length(which(dadostrei$Y==1&YestADB2trei==1))/(length(which(dadostrei$Y==1&YestADB2trei==0))+length(which(dadostrei$Y==1&YestADB2trei==1)))*100,length(which(dadostrei$Y==0&YestADB2trei==0))/(length(which(dadostrei$Y==0&YestADB2trei==0))+length(which(dadostrei$Y==0&YestADB2trei==1)))*100,length(which(dadostrei$Y==1&YestADB2trei==1))/(length(which(dadostrei$Y==0&YestADB2trei==1))+length(which(dadostrei$Y==1&YestADB2trei==1)))*100)
Longtest1[i,]=c((length(which(dadostest$Y==0&YestADB2test==0))+length(which(dadostest$Y==1&YestADB2test==1)))/length(dadostest$Y)*100,length(which(dadostest$Y==1&YestADB2test==1))/(length(which(dadostest$Y==1&YestADB2test==0))+length(which(dadostest$Y==1&YestADB2test==1)))*100,length(which(dadostest$Y==0&YestADB2test==0))/(length(which(dadostest$Y==0&YestADB2test==0))+length(which(dadostest$Y==0&YestADB2test==1)))*100,length(which(dadostest$Y==1&YestADB2test==1))/(length(which(dadostest$Y==0&YestADB2test==1))+length(which(dadostest$Y==1&YestADB2test==1)))*100)

Longtrei2[i,]=c((length(which(dadostrei$Y==0&YestADB3trei==0))+length(which(dadostrei$Y==1&YestADB3trei==1)))/length(dadostrei$Y)*100,length(which(dadostrei$Y==1&YestADB3trei==1))/(length(which(dadostrei$Y==1&YestADB3trei==0))+length(which(dadostrei$Y==1&YestADB3trei==1)))*100,length(which(dadostrei$Y==0&YestADB3trei==0))/(length(which(dadostrei$Y==0&YestADB3trei==0))+length(which(dadostrei$Y==0&YestADB3trei==1)))*100,length(which(dadostrei$Y==1&YestADB3trei==1))/(length(which(dadostrei$Y==0&YestADB3trei==1))+length(which(dadostrei$Y==1&YestADB3trei==1)))*100)
Longtest2[i,]=c((length(which(dadostest$Y==0&YestADB3test==0))+length(which(dadostest$Y==1&YestADB3test==1)))/length(dadostest$Y)*100,length(which(dadostest$Y==1&YestADB3test==1))/(length(which(dadostest$Y==1&YestADB3test==0))+length(which(dadostest$Y==1&YestADB3test==1)))*100,length(which(dadostest$Y==0&YestADB3test==0))/(length(which(dadostest$Y==0&YestADB3test==0))+length(which(dadostest$Y==0&YestADB3test==1)))*100,length(which(dadostest$Y==1&YestADB3test==1))/(length(which(dadostest$Y==0&YestADB3test==1))+length(which(dadostest$Y==1&YestADB3test==1)))*100)

Longtrei3[i,]=c((length(which(Y[as.numeric(names(na.omit(YestLR5trei)))]==0&na.omit(YestLR5trei)==0))+length(which(Y[as.numeric(names(na.omit(YestLR5trei)))]==1&na.omit(YestLR5trei)==1)))/length(na.omit(YestLR5trei))*100,length(which(Y[as.numeric(names(na.omit(YestLR5trei)))]==1&na.omit(YestLR5trei)==1))/(length(which(Y[as.numeric(names(na.omit(YestLR5trei)))]==1&na.omit(YestLR5trei)==0))+length(which(Y[as.numeric(names(na.omit(YestLR5trei)))]==1&na.omit(YestLR5trei)==1)))*100,length(which(Y[as.numeric(names(na.omit(YestLR5trei)))]==0&na.omit(YestLR5trei)==0))/(length(which(Y[as.numeric(names(na.omit(YestLR5trei)))]==0&na.omit(YestLR5trei)==0))+length(which(Y[as.numeric(names(na.omit(YestLR5trei)))]==0&na.omit(YestLR5trei)==1)))*100,length(which(Y[as.numeric(names(na.omit(YestLR5trei)))]==1&na.omit(YestLR5trei)==1))/(length(which(Y[as.numeric(names(na.omit(YestLR5trei)))]==0&na.omit(YestLR5trei)==1))+length(which(Y[as.numeric(names(na.omit(YestLR5trei)))]==1&na.omit(YestLR5trei)==1)))*100)
Longtest3[i,]=c((length(which(Y[as.numeric(names(na.omit(YestLR5test)))]==0&na.omit(YestLR5test)==0))+length(which(Y[as.numeric(names(na.omit(YestLR5test)))]==1&na.omit(YestLR5test)==1)))/length(na.omit(YestLR5test))*100,length(which(Y[as.numeric(names(na.omit(YestLR5test)))]==1&na.omit(YestLR5test)==1))/(length(which(Y[as.numeric(names(na.omit(YestLR5test)))]==1&na.omit(YestLR5test)==0))+length(which(Y[as.numeric(names(na.omit(YestLR5test)))]==1&na.omit(YestLR5test)==1)))*100,length(which(Y[as.numeric(names(na.omit(YestLR5test)))]==0&na.omit(YestLR5test)==0))/(length(which(Y[as.numeric(names(na.omit(YestLR5test)))]==0&na.omit(YestLR5test)==0))+length(which(Y[as.numeric(names(na.omit(YestLR5test)))]==0&na.omit(YestLR5test)==1)))*100,length(which(Y[as.numeric(names(na.omit(YestLR5test)))]==1&na.omit(YestLR5test)==1))/(length(which(Y[as.numeric(names(na.omit(YestLR5test)))]==0&na.omit(YestLR5test)==1))+length(which(Y[as.numeric(names(na.omit(YestLR5test)))]==1&na.omit(YestLR5test)==1)))*100)

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
  length(which(dadostrei$Y==0&YestNB5trei==1)),length(which(dadostrei$Y==1&YestNB5trei==1)),RF5$confusion[1,2],RF5$confusion[2,2],length(which(Y[as.numeric(names(YestSVM5trei))]==0&YestSVM5trei==1)),length(which(Y[as.numeric(names(YestSVM5trei))]==1&YestSVM5trei==1)),length(which(Y[as.numeric(names(na.omit(YestLR5trei)))]==0&na.omit(YestLR5trei)==1)),length(which(Y[as.numeric(names(na.omit(YestLR5trei)))]==1&na.omit(YestLR5trei)==1)),length(which(dadostrei$Y==0&YestADB5trei==1)),length(which(dadostrei$Y==1&YestADB5trei==1))),2*metodos,2*models)/B


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
  length(which(dadostest$Y==0&YestNB5test==1)),length(which(dadostest$Y==1&YestNB5test==1)),RF5$test$confusion[1,2],RF5$test$confusion[2,2],length(which(Y[as.numeric(names(YestSVM5test))]==0&YestSVM5test==1)),length(which(Y[as.numeric(names(YestSVM5test))]==1&YestSVM5test==1)),length(which(Y[as.numeric(names(na.omit(YestLR5test)))]==0&na.omit(YestLR5test)==1)),length(which(Y[as.numeric(names(na.omit(YestLR5test)))]==1&na.omit(YestLR5test)==1)),length(which(dadostest$Y==0&YestADB5test==1)),length(which(dadostest$Y==1&YestADB5test==1))),2*metodos,2*models)/B


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
  
  modNB1=naiveBayes(Y~sex+painexer+relrest+htn+smoke+fbs+famhist+dig+prop+nitr+pro+diuretic+exang+slope+ladprox+laddist+cxmain+om1+rcaprox+junk,data = dadostrei)
  YestNB1trei=predict(modNB1,newdata = dadostrei);YestNB1test=predict(modNB1,newdata = dadostest)
  
  modNB2=naiveBayes(Y~sex+painexer+relrest+htn+chol+cigs+years+restecg+proto+lmt+ladprox+diag+cxmain+ramus+om1+rcadist+lvx2+lvx3+ALPHA,data = dadostrei);YestNB2=predict(modNB2,newdata = dadostest)
  YestNB2trei=predict(modNB2,newdata = dadostrei);YestNB2test=predict(modNB2,newdata = dadostest)
  
  modNB3=naiveBayes(Y~sex+painexer+rldv5e+ladprox+laddist,data = dadostrei) #### MELHOR
  YestNB3trei=predict(modNB3,newdata = dadostrei);YestNB3test=predict(modNB3,newdata = dadostest)
  
  modNB4=naiveBayes(Y~age+sex+painexer+relrest+trestbps+htn+chol+smoke+cigs+years+fbs+famhist+restecg+proto+thaldur+met+thalach+thalrest+trestbpd+exang+oldpeak+rldv5+rldv5e+lmt+ladprox+laddist+diag+cxmain+om1+om2+rcaprox+rcadist+lvx2+lvx3+lvx4+lvf+cathef+PAM+IPPA+RC+IPPARC+HM+ALPHA+ALPHA2,data = dadostrei)
  YestNB4trei=predict(modNB4,newdata = dadostrei);YestNB4test=predict(modNB4,newdata = dadostest)
  
  modNB5=naiveBayes(Y~sex+painexer+relrest+htn+cigs+famhist+proto+thalrest+rldv5+lmt+laddist+cxmain+om1+om2+rcaprox+lvx3+lvf+PAM+IPPA,data = dadostrei)
  YestNB5trei=predict(modNB5,newdata = dadostrei);YestNB5test=predict(modNB5,newdata = dadostest)
  
  ##########################################
  #             CLASSIFICACAO
  #           ARVORE DE DECISAO
  ##########################################
  
  DadostestRF1=na.omit(data.frame(dadostest$Y,dadostest$sex,dadostest$painexer,dadostest$relrest,dadostest$htn,dadostest$smoke,dadostest$fbs,dadostest$famhist,dadostest$dig,dadostest$prop,dadostest$nitr,dadostest$pro,dadostest$diuretic,dadostest$exang,dadostest$slope,dadostest$ladprox,dadostest$laddist,dadostest$cxmain,dadostest$om1,dadostest$rcaprox,dadostest$junk))
  RF1=randomForest(Y~sex+painexer+relrest+htn+smoke+fbs+famhist+dig+prop+nitr+pro+diuretic+exang+slope+ladprox+laddist+cxmain+om1+rcaprox+junk,data = dadostrei,xtest=DadostestRF1[,-1],ytest=DadostestRF1$dadostest.Y,na.action = na.omit)
  
  DadostestRF2=na.omit(data.frame(dadostest$Y,dadostest$sex,dadostest$painexer,dadostest$relrest,dadostest$htn,dadostest$chol,dadostest$cigs,dadostest$years,dadostest$restecg,dadostest$proto,dadostest$lmt,dadostest$ladprox,dadostest$diag,dadostest$cxmain,dadostest$ramus,dadostest$om1,dadostest$rcadist,dadostest$lvx2,dadostest$lvx3,dadostest$ALPHA))
  RF2=randomForest(Y~sex+painexer+relrest+htn+chol+cigs+years+restecg+proto+lmt+ladprox+diag+cxmain+ramus+om1+rcadist+lvx2+lvx3+ALPHA,data = dadostrei,xtest=DadostestRF2[,-1],ytest=DadostestRF2$dadostest.Y,na.action = na.omit)
  
  DadostestRF3=na.omit(data.frame(dadostest$Y,dadostest$sex,dadostest$painexer,dadostest$rldv5e,dadostest$ladprox,dadostest$laddist))
  RF3=randomForest(Y~sex+painexer+rldv5e+ladprox+laddist,data = dadostrei,xtest=DadostestRF3[,-1],ytest=DadostestRF3$dadostest.Y,na.action = na.omit)
  
  DadostestRF4=na.omit(data.frame(dadostest$Y,dadostest$age,dadostest$sex,dadostest$painexer,dadostest$relrest,dadostest$trestbps,dadostest$htn,dadostest$chol,dadostest$smoke,dadostest$cigs,dadostest$years,dadostest$fbs,dadostest$famhist,dadostest$restecg,dadostest$proto,dadostest$thaldur,dadostest$met,dadostest$thalach,dadostest$thalrest,dadostest$trestbpd,dadostest$exang,dadostest$oldpeak,dadostest$rldv5,dadostest$rldv5e,dadostest$lmt,dadostest$ladprox,dadostest$laddist,dadostest$diag,dadostest$cxmain,dadostest$om1,dadostest$om2,dadostest$rcaprox,dadostest$rcadist,dadostest$lvx2,dadostest$lvx3,dadostest$lvx4,dadostest$lvf,dadostest$cathef,dadostest$PAM,dadostest$IPPA,dadostest$RC,dadostest$IPPARC,dadostest$HM,dadostest$ALPHA,dadostest$ALPHA2))
  RF4=randomForest(Y~age+sex+painexer+relrest+trestbps+htn+chol+smoke+cigs+years+fbs+famhist+restecg+proto+thaldur+met+thalach+thalrest+trestbpd+exang+oldpeak+rldv5+rldv5e+lmt+ladprox+laddist+diag+cxmain+om1+om2+rcaprox+rcadist+lvx2+lvx3+lvx4+lvf+cathef+PAM+IPPA+RC+IPPARC+HM+ALPHA+ALPHA2,data = dadostrei,xtest=DadostestRF4[,-1],ytest=DadostestRF4$dadostest.Y,na.action = na.omit)
  
  DadostestRF5=na.omit(data.frame(dadostest$Y,dadostest$sex,dadostest$painexer,dadostest$relrest,dadostest$htn,dadostest$cigs,dadostest$famhist,dadostest$proto,dadostest$thalrest,dadostest$rldv5,dadostest$lmt,dadostest$laddist,dadostest$cxmain,dadostest$om1,dadostest$om2,dadostest$rcaprox,dadostest$lvx3,dadostest$lvf,dadostest$PAM,dadostest$IPPA))
  RF5=randomForest(Y~sex+painexer+relrest+htn+cigs+famhist+proto+thalrest+rldv5+lmt+laddist+cxmain+om1+om2+rcaprox+lvx3+lvf+PAM+IPPA,data = dadostrei,xtest=DadostestRF5[,-1],ytest=DadostestRF5$dadostest.Y,na.action = na.omit)
  
  ##########################################
  #             CLASSIFICACAO
  #     MAQUINA DE VETORES DE SUPORTE
  ##########################################
  
  modSVM1=svm(Y~sex+painexer+relrest+htn+smoke+fbs+famhist+dig+prop+nitr+pro+diuretic+exang+slope+ladprox+laddist+cxmain+om1+rcaprox+junk,data=dadostrei,scale=F,kernel="poly",cost=100,epsilon=1.0e-12)
  YestSVM1test=predict(modSVM1,dadostest);YestSVM1trei=modSVM1$fitted
  
  modSVM2=svm(Y~sex+painexer+relrest+htn+chol+cigs+years+restecg+proto+lmt+ladprox+diag+cxmain+ramus+om1+rcadist+lvx2+lvx3+ALPHA,data=dadostrei,scale=F,kernel="poly",cost=100,epsilon=1.0e-12)
  YestSVM2test=predict(modSVM2,dadostest);YestSVM2trei=modSVM2$fitted
  
  modSVM3=svm(Y~sex+painexer+rldv5e+ladprox+laddist,scale=F,data=dadostrei,kernel="poly",cost=100,epsilon=1.0e-12)
  YestSVM3test=predict(modSVM3,dadostest);YestSVM3trei=modSVM3$fitted
  
  modSVM4=svm(Y~age+sex+painexer+relrest+trestbps+htn+chol+smoke+cigs+years+fbs+famhist+restecg+proto+thaldur+met+thalach+thalrest+trestbpd+exang+oldpeak+rldv5+rldv5e+lmt+ladprox+laddist+diag+cxmain+om1+om2+rcaprox+rcadist+lvx2+lvx3+lvx4+lvf+cathef+PAM+IPPA+RC+IPPARC+HM+ALPHA+ALPHA2,data=dadostrei,scale=F,kernel="poly",cost=100,epsilon=1.0e-12)
  YestSVM4test=predict(modSVM4,dadostest);YestSVM4trei=modSVM4$fitted
  
  modSVM5=svm(Y~sex+painexer+relrest+htn+cigs+famhist+proto+thalrest+rldv5+lmt+laddist+cxmain+om1+om2+rcaprox+lvx3+lvf+PAM+IPPA,data=dadostrei,scale=F,kernel="poly",cost=100,epsilon=1.0e-12)
  YestSVM5test=predict(modSVM5,dadostest);YestSVM5trei=modSVM5$fitted
  
  ##########################################
  #             CLASSIFICACAO
  #     REGRESSAO LINEAR - LOGISTICA
  ##########################################
  
  modLR1=glm(Y~painexer+relrest+htn+smoke+fbs+famhist+dig+prop+nitr+pro+diuretic+exang+slope+ladprox+laddist+cxmain+om1+rcaprox+junk,data=dadostrei,family = binomial(link = "logit"),na.action = na.omit)
  YestLR1test=ifelse(predict.glm(modLR1,newdata = dadostest,type = "response")<0.5,0,1);YestLR1trei=ifelse(predict.glm(modLR1,newdata = dadostrei,type = "response")<0.5,0,1)
  
  modLR2=glm(Y~painexer+relrest+htn+chol+cigs+years+restecg+proto+lmt+ladprox+diag+cxmain+ramus+om1+rcadist+lvx2+lvx3+ALPHA,data=dadostrei,family = binomial(link = "logit"))
  YestLR2test=ifelse(predict.glm(modLR2,newdata = dadostest,type = "response")<0.5,0,1);YestLR2trei=ifelse(predict.glm(modLR2,newdata = dadostrei,type = "response")<0.5,0,1)
  
  modLR3=glm(Y~painexer+rldv5e+ladprox+laddist,data=dadostrei,family = binomial(link = "logit"))
  YestLR3test=ifelse(predict.glm(modLR3,newdata = dadostest,type = "response")<0.5,0,1);YestLR3trei=ifelse(predict.glm(modLR3,newdata = dadostrei,type = "response")<0.5,0,1)
  
  modLR4=glm(Y~age+painexer+relrest+trestbps+htn+chol+smoke+cigs+years+fbs+famhist+restecg+proto+thaldur+met+thalach+thalrest+trestbpd+exang+oldpeak+rldv5+rldv5e+lmt+ladprox+laddist+diag+cxmain+om1+om2+rcaprox+rcadist+lvx2+lvx3+lvx4+lvf+cathef+PAM+IPPA+RC+IPPARC+HM+ALPHA+ALPHA2,data=dadostrei,family = binomial(link = "logit"))
  YestLR4test=ifelse(predict.glm(modLR4,newdata = dadostest,type = "response")<0.5,0,1);YestLR4trei=ifelse(predict.glm(modLR4,newdata = dadostrei,type = "response")<0.5,0,1)
  
  modLR5=glm(Y~painexer+relrest+htn+cigs+famhist+proto+thalrest+rldv5+lmt+laddist+cxmain+om1+om2+rcaprox+lvx3+lvf+PAM+IPPA,data=dadostrei,family = binomial(link = "logit"))
  YestLR5test=ifelse(predict.glm(modLR5,newdata = dadostest,type = "response")<0.5,0,1);YestLR5trei=ifelse(predict.glm(modLR5,newdata = dadostrei,type = "response")<0.5,0,1)
  
  ##########################################
  #             CLASSIFICACAO
  #               ADABOOST
  ##########################################
  
  modADB1=adaboost(Y~sex+painexer+relrest+htn+smoke+fbs+famhist+dig+prop+nitr+pro+diuretic+exang+slope+ladprox+laddist+cxmain+om1+rcaprox+junk,data=dadostrei,10)
  YestADB1test=predict(modADB1,newdata=dadostest)$class;YestADB1trei=predict(modADB1,newdata=dadostrei)$class
  
  modADB2=adaboost(Y~sex+painexer+relrest+htn+chol+cigs+years+restecg+proto+lmt+ladprox+diag+cxmain+ramus+om1+rcadist+lvx2+lvx3+ALPHA,data=dadostrei,10)
  YestADB2test=predict(modADB2,newdata=dadostest)$class;YestADB2trei=predict(modADB2,newdata=dadostrei)$class
  
  modADB3=adaboost(Y~sex+painexer+rldv5e+ladprox+laddist,data=dadostrei,10)
  YestADB3test=predict(modADB3,newdata=dadostest)$class;YestADB3trei=predict(modADB3,newdata=dadostrei)$class
  
  modADB4=adaboost(Y~age+sex+painexer+relrest+trestbps+htn+chol+smoke+cigs+years+fbs+famhist+restecg+proto+thaldur+met+thalach+thalrest+trestbpd+exang+oldpeak+rldv5+rldv5e+lmt+ladprox+laddist+diag+cxmain+om1+om2+rcaprox+rcadist+lvx2+lvx3+lvx4+lvf+cathef+PAM+IPPA+RC+IPPARC+HM+ALPHA+ALPHA2,data=dadostrei,10)
  YestADB4test=predict(modADB4,newdata=dadostest)$class;YestADB4trei=predict(modADB4,newdata=dadostrei)$class
  
  modADB5=adaboost(Y~sex+painexer+relrest+htn+cigs+famhist+proto+thalrest+rldv5+lmt+laddist+cxmain+om1+om2+rcaprox+lvx3+lvf+PAM+IPPA,data=dadostrei,10)
  YestADB5test=predict(modADB5,newdata=dadostest)$class;YestADB5trei=predict(modADB5,newdata=dadostrei)$class
  
  
  #############################
  #         ACURACIAS         #
  #############################
  
  acurNBtrei[i,]=c((length(which(dadostrei$Y==0&YestNB1trei==0))+length(which(dadostrei$Y==1&YestNB1trei==1)))/length(dadostrei$Y)*100,(length(which(dadostrei$Y==0&YestNB2trei==0))+length(which(dadostrei$Y==1&YestNB2trei==1)))/length(dadostrei$Y)*100,(length(which(dadostrei$Y==0&YestNB3trei==0))+length(which(dadostrei$Y==1&YestNB3trei==1)))/length(dadostrei$Y)*100,(length(which(dadostrei$Y==0&YestNB4trei==0))+length(which(dadostrei$Y==1&YestNB4trei==1)))/length(dadostrei$Y)*100,(length(which(dadostrei$Y==0&YestNB5trei==0))+length(which(dadostrei$Y==1&YestNB5trei==1)))/length(dadostrei$Y)*100)
  acurNBtest[i,]=c((length(which(dadostest$Y==0&YestNB1test==0))+length(which(dadostest$Y==1&YestNB1test==1)))/length(dadostest$Y)*100,(length(which(dadostest$Y==0&YestNB2test==0))+length(which(dadostest$Y==1&YestNB2test==1)))/length(dadostest$Y)*100,(length(which(dadostest$Y==0&YestNB3test==0))+length(which(dadostest$Y==1&YestNB3test==1)))/length(dadostest$Y)*100,(length(which(dadostest$Y==0&YestNB4test==0))+length(which(dadostest$Y==1&YestNB4test==1)))/length(dadostest$Y)*100,(length(which(dadostest$Y==0&YestNB5test==0))+length(which(dadostest$Y==1&YestNB5test==1)))/length(dadostest$Y)*100)
  
  acurDTtrei[i,]=c((RF1$confusion[1,1]+RF1$confusion[2,2])/sum(RF1$confusion[,-3])*100,(RF2$confusion[1,1]+RF2$confusion[2,2])/sum(RF2$confusion[,-3])*100,(RF3$confusion[1,1]+RF3$confusion[2,2])/sum(RF3$confusion[,-3])*100,(RF4$confusion[1,1]+RF4$confusion[2,2])/sum(RF4$confusion[,-3])*100,(RF5$confusion[1,1]+RF5$confusion[2,2])/sum(RF5$confusion[,-3])*100)
  acurDTtest[i,]=c((RF1$test$confusion[1,1]+RF1$test$confusion[2,2])/sum(RF1$test$confusion[,-3])*100,(RF2$test$confusion[1,1]+RF2$test$confusion[2,2])/sum(RF2$test$confusion[,-3])*100,(RF3$test$confusion[1,1]+RF3$test$confusion[2,2])/sum(RF3$test$confusion[,-3])*100,(RF4$test$confusion[1,1]+RF4$test$confusion[2,2])/sum(RF4$test$confusion[,-3])*100,(RF5$test$confusion[1,1]+RF5$test$confusion[2,2])/sum(RF5$test$confusion[,-3])*100)
  
  acurSVMtrei[i,]=c((length(which(Y[as.numeric(names(YestSVM1trei))]==0&YestSVM1trei==0))+length(which(Y[as.numeric(names(YestSVM1trei))]==1&YestSVM1trei==1)))/length(YestSVM1trei)*100,(length(which(Y[as.numeric(names(YestSVM2trei))]==0&YestSVM2trei==0))+length(which(Y[as.numeric(names(YestSVM2trei))]==1&YestSVM2trei==1)))/length(YestSVM2trei)*100,(length(which(Y[as.numeric(names(YestSVM3trei))]==0&YestSVM3trei==0))+length(which(Y[as.numeric(names(YestSVM3trei))]==1&YestSVM3trei==1)))/length(YestSVM3trei)*100,(length(which(Y[as.numeric(names(YestSVM4trei))]==0&YestSVM4trei==0))+length(which(Y[as.numeric(names(YestSVM4trei))]==1&YestSVM4trei==1)))/length(YestSVM4trei)*100,(length(which(Y[as.numeric(names(YestSVM5trei))]==0&YestSVM5trei==0))+length(which(Y[as.numeric(names(YestSVM5trei))]==1&YestSVM5trei==1)))/length(YestSVM5trei)*100)
  acurSVMtest[i,]=c((length(which(Y[as.numeric(names(YestSVM1test))]==0&YestSVM1test==0))+length(which(Y[as.numeric(names(YestSVM1test))]==1&YestSVM1test==1)))/length(YestSVM1test)*100,(length(which(Y[as.numeric(names(YestSVM2test))]==0&YestSVM2test==0))+length(which(Y[as.numeric(names(YestSVM2test))]==1&YestSVM2test==1)))/length(YestSVM2test)*100,(length(which(Y[as.numeric(names(YestSVM3test))]==0&YestSVM3test==0))+length(which(Y[as.numeric(names(YestSVM3test))]==1&YestSVM3test==1)))/length(YestSVM3test)*100,(length(which(Y[as.numeric(names(YestSVM4test))]==0&YestSVM4test==0))+length(which(Y[as.numeric(names(YestSVM4test))]==1&YestSVM4test==1)))/length(YestSVM4test)*100,(length(which(Y[as.numeric(names(YestSVM5test))]==0&YestSVM5test==0))+length(which(Y[as.numeric(names(YestSVM5test))]==1&YestSVM5test==1)))/length(YestSVM5test)*100)
  
  acurLRtrei[i,]=c((length(which(Y[as.numeric(names(na.omit(YestLR1trei)))]==0&na.omit(YestLR1trei)==0))+length(which(Y[as.numeric(names(na.omit(YestLR1trei)))]==1&na.omit(YestLR1trei)==1)))/length(na.omit(YestLR1trei))*100,(length(which(Y[as.numeric(names(na.omit(YestLR2trei)))]==0&na.omit(YestLR2trei)==0))+length(which(Y[as.numeric(names(na.omit(YestLR2trei)))]==1&na.omit(YestLR2trei)==1)))/length(na.omit(YestLR2trei))*100,(length(which(Y[as.numeric(names(na.omit(YestLR3trei)))]==0&na.omit(YestLR3trei)==0))+length(which(Y[as.numeric(names(na.omit(YestLR3trei)))]==1&na.omit(YestLR3trei)==1)))/length(na.omit(YestLR3trei))*100,(length(which(Y[as.numeric(names(na.omit(YestLR4trei)))]==0&na.omit(YestLR4trei)==0))+length(which(Y[as.numeric(names(na.omit(YestLR4trei)))]==1&na.omit(YestLR4trei)==1)))/length(na.omit(YestLR4trei))*100,(length(which(Y[as.numeric(names(na.omit(YestLR5trei)))]==0&na.omit(YestLR5trei)==0))+length(which(Y[as.numeric(names(na.omit(YestLR5trei)))]==1&na.omit(YestLR5trei)==1)))/length(na.omit(YestLR5trei))*100)
  acurLRtest[i,]=c((length(which(Y[as.numeric(names(na.omit(YestLR1test)))]==0&na.omit(YestLR1test)==0))+length(which(Y[as.numeric(names(na.omit(YestLR1test)))]==1&na.omit(YestLR1test)==1)))/length(na.omit(YestLR1test))*100,(length(which(Y[as.numeric(names(na.omit(YestLR2test)))]==0&na.omit(YestLR2test)==0))+length(which(Y[as.numeric(names(na.omit(YestLR2test)))]==1&na.omit(YestLR2test)==1)))/length(na.omit(YestLR2test))*100,(length(which(Y[as.numeric(names(na.omit(YestLR3test)))]==0&na.omit(YestLR3test)==0))+length(which(Y[as.numeric(names(na.omit(YestLR3test)))]==1&na.omit(YestLR3test)==1)))/length(na.omit(YestLR3test))*100,(length(which(Y[as.numeric(names(na.omit(YestLR4test)))]==0&na.omit(YestLR4test)==0))+length(which(Y[as.numeric(names(na.omit(YestLR4test)))]==1&na.omit(YestLR4test)==1)))/length(na.omit(YestLR4test))*100,(length(which(Y[as.numeric(names(na.omit(YestLR5test)))]==0&na.omit(YestLR5test)==0))+length(which(Y[as.numeric(names(na.omit(YestLR5test)))]==1&na.omit(YestLR5test)==1)))/length(na.omit(YestLR5test))*100)
  
  acurADBtrei[i,]=c((length(which(dadostrei$Y==0&YestADB1trei==0))+length(which(dadostrei$Y==1&YestADB1trei==1)))/length(dadostrei$Y)*100,(length(which(dadostrei$Y==0&YestADB2trei==0))+length(which(dadostrei$Y==1&YestADB2trei==1)))/length(dadostrei$Y)*100,(length(which(dadostrei$Y==0&YestADB3trei==0))+length(which(dadostrei$Y==1&YestADB3trei==1)))/length(dadostrei$Y)*100,(length(which(dadostrei$Y==0&YestADB4trei==0))+length(which(dadostrei$Y==1&YestADB4trei==1)))/length(dadostrei$Y)*100,(length(which(dadostrei$Y==0&YestADB5trei==0))+length(which(dadostrei$Y==1&YestADB5trei==1)))/length(dadostrei$Y)*100)
  acurADBtest[i,]=c((length(which(dadostest$Y==0&YestADB1test==0))+length(which(dadostest$Y==1&YestADB1test==1)))/length(dadostest$Y)*100,(length(which(dadostest$Y==0&YestADB2test==0))+length(which(dadostest$Y==1&YestADB2test==1)))/length(dadostest$Y)*100,(length(which(dadostest$Y==0&YestADB3test==0))+length(which(dadostest$Y==1&YestADB3test==1)))/length(dadostest$Y)*100,(length(which(dadostest$Y==0&YestADB4test==0))+length(which(dadostest$Y==1&YestADB4test==1)))/length(dadostest$Y)*100,(length(which(dadostest$Y==0&YestADB5test==0))+length(which(dadostest$Y==1&YestADB5test==1)))/length(dadostest$Y)*100)
  
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
    length(which(dadostrei$Y==0&YestNB5trei==1)),length(which(dadostrei$Y==1&YestNB5trei==1)),RF5$confusion[1,2],RF5$confusion[2,2],length(which(Y[as.numeric(names(YestSVM5trei))]==0&YestSVM5trei==1)),length(which(Y[as.numeric(names(YestSVM5trei))]==1&YestSVM5trei==1)),length(which(Y[as.numeric(names(na.omit(YestLR5trei)))]==0&na.omit(YestLR5trei)==1)),length(which(Y[as.numeric(names(na.omit(YestLR5trei)))]==1&na.omit(YestLR5trei)==1)),length(which(dadostrei$Y==0&YestADB5trei==1)),length(which(dadostrei$Y==1&YestADB5trei==1))),2*metodos,2*models)-matconftrei)^2)/(B-1)
  
  
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
    length(which(dadostest$Y==0&YestNB5test==1)),length(which(dadostest$Y==1&YestNB5test==1)),RF5$test$confusion[1,2],RF5$test$confusion[2,2],length(which(Y[as.numeric(names(YestSVM5test))]==0&YestSVM5test==1)),length(which(Y[as.numeric(names(YestSVM5test))]==1&YestSVM5test==1)),length(which(Y[as.numeric(names(na.omit(YestLR5test)))]==0&na.omit(YestLR5test)==1)),length(which(Y[as.numeric(names(na.omit(YestLR5test)))]==1&na.omit(YestLR5test)==1)),length(which(dadostest$Y==0&YestADB5test==1)),length(which(dadostest$Y==1&YestADB5test==1))),2*metodos,2*models)-matconftest)^2)/(B-1)
  
  
  j=j+1; print(j)
}
matdesvtest=round(sqrt(sdmatconftest),2)
matdesvtrei=round(sqrt(sdmatconftrei),2)

mattreifinal=cbind(matconftrei[,1],matdesvtrei[,1],matconftrei[,2],matdesvtrei[,2],matconftrei[,3],matdesvtrei[,3],matconftrei[,4],matdesvtrei[,4],matconftrei[,5],matdesvtrei[,5],matconftrei[,6],matdesvtrei[,6],matconftrei[,7],matdesvtrei[,7],matconftrei[,8],matdesvtrei[,8],matconftrei[,9],matdesvtrei[,9],matconftrei[,10],matdesvtrei[,10])
mattestfinal=cbind(matconftest[,1],matdesvtest[,1],matconftest[,2],matdesvtest[,2],matconftest[,3],matdesvtest[,3],matconftest[,4],matdesvtest[,4],matconftest[,5],matdesvtest[,5],matconftest[,6],matdesvtest[,6],matconftest[,7],matdesvtest[,7],matconftest[,8],matdesvtest[,8],matconftest[,9],matdesvtest[,9],matconftest[,10],matdesvtest[,10])

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



mimetricastest=rbind(apply(na.omit(Longtest1),2,mean),apply(na.omit(Longtest2),2,mean),apply(na.omit(Longtest3),2,mean))
sdmetricastest=rbind(apply(na.omit(Longtest1),2,sd),apply(na.omit(Longtest2),2,sd),apply(na.omit(Longtest3),2,sd))

metricastest=cbind(mimetricastest[,1],sdmetricastest[,1],mimetricastest[,2],sdmetricastest[,2],
                   mimetricastest[,3],sdmetricastest[,3],mimetricastest[,4],sdmetricastest[,4])
rownames(metricastest)=c("Adaboost","Adaboost","LR")
mitempos=apply(na.omit(tempos), 2, mean)
sdtempos=apply(na.omit(tempos), 2, sd)
tempoProc=cbind(mitempos,sdtempos)

write(Longtest1,file = "MetricasLongum.txt",ncolumns = 4)
write(Longtest2,file = "MetricasLongdois.txt",ncolumns = 4)
write(Longtest3,file = "MetricasLongtres.txt",ncolumns = 4)

write(Longtrei1,file = "MetricasLongumtrei.txt",ncolumns = 4)
write(Longtrei2,file = "MetricasLongdoistrei.txt",ncolumns = 4)
write(Longtrei3,file = "MetricasLongtrestrei.txt",ncolumns = 4)
write(tempos,file = "tempoLong.txt",ncolumns = 3)

xtable(metricastest,digits = 2)
xtable(tempoProc*1000,digits=2)


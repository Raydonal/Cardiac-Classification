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
dados=read.table("Totallugares.txt",na.strings = -9,header=T,dec=".",sep = "\t")
attach(dados)

#library(WriteXLS)
#testPerl(perl = "perl", verbose = TRUE)
#WriteXLS("dados", "Cleveland.xls")

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

table(Y,country)
chisq.test(table(Y,country))
chisq.test(table(c(Y[which(country=="Cleveland")],Y[which(country=="Hungarian")]),
                 c(country[which(country=="Cleveland")],country[which(country=="Hungarian")])))
chisq.test(table(c(Y[which(country=="Cleveland")],Y[which(country=="Long Beach")]),
                 c(country[which(country=="Cleveland")],country[which(country=="Long Beach")])))
chisq.test(table(c(Y[which(country=="Cleveland")],Y[which(country=="Switzerland")]),
                 c(country[which(country=="Cleveland")],country[which(country=="Switzerland")])))
chisq.test(table(c(Y[which(country=="Long Beach")],Y[which(country=="Hungarian")]),
                 c(country[which(country=="Long Beach")],country[which(country=="Hungarian")])))
chisq.test(table(c(Y[which(country=="Hungarian")],Y[which(country=="Switzerland")]),
                 c(country[which(country=="Hungarian")],country[which(country=="Switzerland")])))
chisq.test(table(c(Y[which(country=="Long Beach")],Y[which(country=="Switzerland")]),
                 c(country[which(country=="Long Beach")],country[which(country=="Switzerland")])))

##### ANALISE DESCRITIVA DOS PARAMETROS

summary1=apply(X[which(Y==1),],2,summary)
summary0=apply(X[which(Y==0),],2,summary)
sd1=apply(X[which(Y==1),],2,sd,na.rm = TRUE)
sd0=apply(X[which(Y==0),],2,sd,na.rm = TRUE)

tabela1= as.table(cbind(t(summary1),sd1))
rownames(tabela1)=c("PAM","IPPA","RC","IPPARC","HM","alphaDelta","alpha2")
colnames(tabela1)=c("Mín.","1o Quartil","Mediana","Média","3o Quartil",'Máx.',"Desv.Pad.")

tabela0= as.table(cbind(t(summary0),sd0))
rownames(tabela0)=c("PAM","IPPA","RC","IPPARC","HM","alphaDelta","alpha2")
colnames(tabela0)=c("Mín.","1o Quartil","Mediana","Média","3o Quartil",'Máx.',"Desv.Pad.")


xtable(tabela1,digits = 4)
xtable(tabela0,digits = 4)

tabela=cbind(tabela0[,4],tabela1[,4],tabela0[,3],tabela1[,3],tabela0[,7],tabela1[,7])
colnames(tabela)=c("Média0","Média1","Mediana0","Mediana1","Desv.Pad.0","Desv. Pad.1")
xtable(tabela,digits = 4)

#####  BOXPLOTS

par(mfrow=c(1,4))
boxplot(PAM~Y,main="PAM") #NORMAL 
boxplot(IPPA~Y,main="IPPA")
boxplot(RC~Y,main="RC")
boxplot(IPPARC~Y,main="IPPARC")
par(mfrow=c(1,3))
boxplot(HM~Y,main="HM")
boxplot(ALPHA~Y,main=TeX('$\\alpha_\\Delta$'))
boxplot(ALPHA2~Y,main=TeX('$\\alpha$')) #NORMAL


##########################################
#         TESTES DE NORMALIDADE
##########################################

shapiro.test(PAM[which(Y==1)]) #NORMAL à 1%
shapiro.test(IPPA[which(Y==1)])
shapiro.test(RC[which(Y==1)])
shapiro.test(IPPARC[which(Y==1)])
shapiro.test(HM[which(Y==1)])
shapiro.test(ALPHA[which(Y==1)])
shapiro.test(ALPHA2[which(Y==1)]) #NORMAL à qualquer nível de significância

shapiro.test(PAM[which(Y==0)]) #NORMAL à qualquer nível
shapiro.test(IPPA[which(Y==0)])
shapiro.test(RC[which(Y==0)])
shapiro.test(IPPARC[which(Y==0)])
shapiro.test(HM[which(Y==0)])
shapiro.test(ALPHA[which(Y==0)])
shapiro.test(ALPHA2[which(Y==0)]) #NORMAL à qualquer nível


##########################################
#   TESTES DE DIFERENÇA (MÉDIA/MEDIANA)
##########################################

t.test(PAM~Y)         # NORMAL - MÉDIA *** diferente à 5%
wilcox.test(IPPA~Y)   # MEDIANA
wilcox.test(RC~Y)     # MEDIANA
wilcox.test(IPPARC~Y) # MEDIANA
wilcox.test(HM~Y)     # MEDIANA
wilcox.test(ALPHA~Y)  # MEDIANA
t.test(ALPHA2~Y)      # NORMAL - MÉDIA

##########################################
#              CONSISTENCIA
##########################################

mi1=apply(X[which(Y==1),],2,mean,na.rm = TRUE)
mi0=apply(X[which(Y==0),],2,mean,na.rm = TRUE)
var1=apply(X[which(Y==1),],2,var,na.rm = TRUE)
var0=apply(X[which(Y==0),],2,var,na.rm = TRUE)

consist=abs(mi1-mi0)/sqrt(var1+var0)
sort(consist,decreasing = T)


##########################################
#          SELECAO DE VARIAVEIS
#            INFORMATION GAIN
##########################################
#PAM2=ifelse(PAM<mean(PAM),0,1)
#IPPA2=ifelse(IPPA<mean(IPPA),0,1)
#RC2=ifelse(RC<mean(RC),0,1)
#IPPARC2=ifelse(IPPARC<mean(IPPARC),0,1)
#HM2=ifelse(HM<mean(HM),0,1)
#ALPHA12=ifelse(ALPHA<mean(ALPHA),0,1)
#ALPHA22=ifelse(ALPHA2<mean(ALPHA2),0,1)

#newX=cbind(PAM2,IPPA2,RC2,IPPARC2,HM2,ALPHA12,ALPHA22)
#newdata= data.frame(cbind(age,sex,htn,newX,Y))
newdata = data.frame(cbind(dados[,-c(14,52,43,44)],X,Y))
set.seed(26071993)
treinamento=sample(nrow(newdata),round(nrow(newdata)*0.7))
dadostrei=newdata[treinamento,]
dadostest=newdata[-treinamento,]
infogain=information_gain(formula = Y~., data = newdata) # INFOGAIN

infogain=information_gain(formula = Y~., data = dadostrei) # INFOGAIN
infogain$attributes[which(infogain$importance!=0)]
information_gain(formula = Y~., data = dadostrei,type = "gainratio") #RAZAO DE GANHO
information_gain(formula = Y~., data = dadostrei,type = "symuncert") #INCERTEZA SIMETRICA

sex+cp+htn+fbs+dm+famhist+dig+prop+nitr+pro+
diuretic+thaldur+thaltime+met+thalach+exang+
xhypo+oldpeak+slope+ca+thal+ladprox+laddist+
cxmain+om1+rcaprox+rcadist+lvx4

vif(glm(formula = Y~., data = dadostrei))
vif(glm(formula = Y~., data = newdata))

age+sex+painloc+	painexer+	relrest+	cp+	trestbps	+htn+	chol+	smoke	+cigs+	years	+
  +fbs+	famhist	+restecg+	ekgmo	+ekgday	+ekgyr+	dig	+prop	+nitr	+pro+	diuretic+
  proto	+thaldur+	thaltime+	met+	thalach+	thalrest+	tpeakbps+	tpeakbpd+dummy+	trestbpd+
  exang	+xhypo+	oldpeak+	slope+	rldv5+	rldv5e+	restef+	restwm+	+ca	+thal+	thalsev+
  thalpul	+cmo	+cday	+cyr+	lmt+	ladprox	+laddist+	diag	+cxmain	+ramus+	om1+	om2	+
  rcaprox	+rcadist	+lvx1	+lvx2	+lvx3	+lvx4	+lvf+	cathef+	junk	+country


vif(glm(Y~age+sex+	htn+	dig	+prop	+nitr	+	diuretic+
          exang	+exang	+xhypo+	oldpeak+	slope+	
          ladprox	+laddist+	diag	+cxmain	+	om1+
          rcaprox	+rcadist	+PAM+RC,
        data=dadostrei,na.action = na.omit,family = binomial(link = "logit")))

##########################################
#          SELECAO DE VARIAVEIS
#            GENETIC ALGORITHM
##########################################
fitness1=function(Y,age,sex,htn,PAM,IPPA,RC,IPPARC,HM,ALPHA,ALPHA2){ 
  
  Y=beta0+beta1*age+beta2*sex+beta3*htn+beta4*PAM+
    beta5*IPPA+beta6*RC+beta7*IPPARC+beta8*HM+beta9*ALPHA+beta10*ALPHA2
  }

ga(type = "binary",fitness1,popSize = nrow(newdata),pcrossover = 0.6,
   pmutation = 0.033,elitism = base::max(1, round(popSize*0.05)),
   optimArgs = list(method = "L-BFGS-B",
                    poptim = 0.05,
                    pressel = 0.5,
                    control = list(fnscale = -1, maxit = 100)),seed = 1)

##########################################
#          SELECAO DE VARIAVEIS
#                 ANOVA
##########################################
#Y~age+sex+cp+trestbps+htn+chol+cigs+years+fbs+
#  dm+famhist+restecg+ekgmo+ekgday+ekgyr+dig+prop+
#  nitr+pro+diuretic+proto+thaldur+
#  thaltime+met+thalach+thalrest+tpeakbps+tpeakbpd+
#  dummy+trestbpd+exang+xhypo+   
#  oldpeak+slope+rldv5e+ca+thal+cmo+cday+cyr+lmt+
#  ladprox+laddist+cxmain+om1+rcaprox+rcadist+
#  lvx1+lvx2+lvx3+lvx4+lvf+PAM+IPPA+RC+IPPARC+HM+
#  ALPHA+ALPHA2,data=dadostrei

modelonulo=glm(Y~sex,data=dadostrei,family = binomial(link = "logit"))
modelosaturado=glm(Y~sex+age+htn+PAM+IPPA+RC+IPPARC+HM+ALPHA+ALPHA2
,data=dadostrei,family = binomial(link = "logit"))
vif(modelosaturado)

anova(glm(Y~sex,data=dadostrei,family = binomial(link = "logit")),test = "Chisq")
anova(glm(Y~sex+age,data=dadostrei,family = binomial(link = "logit")),test = "Chisq")
anova(glm(Y~sex+age+htn,data=dadostrei,family = binomial(link = "logit")),test = "Chisq")
anova(glm(Y~sex+age+htn+PAM,data=dadostrei,family = binomial(link = "logit")),test = "Chisq")
anova(glm(Y~sex+age+htn+PAM+IPPA,data=dadostrei,family = binomial(link = "logit")),test = "Chisq")
anova(glm(Y~sex+age+htn+PAM+IPPA+RC,data=dadostrei,family = binomial(link = "logit")),test = "Chisq")
anova(glm(Y~sex+age+htn+PAM+IPPA+RC+IPPARC,data=dadostrei,family = binomial(link = "logit")),test = "Chisq")
anova(glm(Y~sex+age+htn+PAM+IPPA+RC+IPPARC+HM,data=dadostrei,family = binomial(link = "logit")),test = "Chisq")
anova(glm(Y~painexer+famhist+thalrest+exang+oldpeak,data=dadostrei,family = binomial(link = "logit")),test = "Chisq")
anova(glm(Y~age+sex+painloc+	painexer+	trestbps	+htn+
            famhist	+restecg+	dig	+prop	+nitr	+pro+	diuretic+
            thaldur+	thalach+	thalrest+	tpeakbps+	tpeakbpd+	trestbpd+
            exang	+xhypo+	oldpeak+	slope+	rldv5+	
            	laddist+
            country+PAM+IPPA+RC+IPPARC+HM+
            ALPHA+ALPHA2,data=dadostrei,family = binomial(link = "logit")),test = "Chisq")
anova(glm(Y~age+sex+painloc+	painexer+	trestbps	+htn+
            famhist	+restecg+	dig	+prop	+nitr	+pro+	diuretic+
            thaldur+	thalach+	thalrest+	tpeakbps+	tpeakbpd+	trestbpd+
            exang	+xhypo+	oldpeak+	slope+	rldv5+	
            laddist+
            country+PAM+IPPA+RC+IPPARC+HM+
            ALPHA+ALPHA2,data=newdata,family = binomial(link = "logit")),test = "Chisq")

summary(glm(Y~age+sex+cp+trestbps+dig+prop+thalach+tpeakbps+
              exang+oldpeak+ca+thal+ladprox+laddist
           ,data=dadostrei,family = binomial(link = "logit")))

vif(glm(Y~painexer+famhist+thalrest+exang+oldpeak
            ,data=dadostrei,family = binomial(link = "logit")))

summary(glm(Y~sex+trestbps+tpeakbps+oldpeak+ca+thal
            ,data=dadostrei,family = binomial(link = "logit")))
##########################################
#          SELECAO DE VARIAVEIS
#                 AIC
##########################################
#sex+age+htn+PAM+IPPA+RC+IPPARC+HM+ALPHA+ALPHA2
#Y~age+sex+cp+trestbps+htn+chol+cigs+years+fbs+
#  dm+famhist+restecg+ekgmo+ekgday+ekgyr+dig+prop+
#  nitr+pro+diuretic+proto+thaldur+
#  thaltime+met+thalach+thalrest+tpeakbps+tpeakbpd+
#  dummy+trestbpd+exang+xhypo+   
#  oldpeak+slope+rldv5e+ca+thal+cmo+cday+cyr+lmt+
#  ladprox+laddist+cxmain+om1+rcaprox+rcadist+
#  lvx1+lvx2+lvx3+lvx4+lvf+PAM+IPPA+RC+IPPARC+HM+
#  ALPHA+ALPHA2,data=dadostrei

modelosaturado=glm(Y~age+sex+painloc+	painexer+	trestbps	+htn+
                     famhist	+restecg+	dig	+prop	+nitr	+pro+	diuretic+
                     thaldur+	thalach+	thalrest+	tpeakbps+	tpeakbpd+	trestbpd+
                     exang	+xhypo+	oldpeak+	slope+	rldv5+	
                     laddist+country+PAM+IPPA+RC+IPPARC+HM+
                     ALPHA+ALPHA2,data=dadostrei,na.action = na.omit,family = binomial(link = "logit"))
modelosaturado=glm(Y~age+sex+painloc+	painexer+	trestbps	+htn+
                     famhist	+restecg+	dig	+prop	+nitr	+pro+	diuretic+
                     thaldur+	thalach+	thalrest+	tpeakbps+	tpeakbpd+	trestbpd+
                     exang	+xhypo+	oldpeak+	slope+	rldv5+	
                     laddist+country+PAM+IPPA+RC+IPPARC+HM+
                     ALPHA+ALPHA2,data=newdata,na.action = na.omit,family = binomial(link = "logit"))
step(modelosaturado)

glm(Y~age+sex+trestbps+years+fbs+famhist+restecg+dig+nitr+diuretic+
      thaldur+met+thalrest+trestbpd+exang+oldpeak+rldv5e+ca+thal+lmt+
      ladprox+laddist+cxmain+om1+rcaprox+rcadist+lvx3+lvx4+lvf+PAM+
      IPPA+RC+ALPHA2
  ,data=dadostrei,na.action = na.omit,family = binomial(link = "logit"))

vif(glm(Y ~ sex+painloc+htn+famhist+restecg+dig+nitr+pro+diuretic+thaldur+
          tpeakbps+trestbpd+exang+xhypo+oldpeak+slope+rldv5+laddist+ALPHA
    ,data=dadostrei,na.action = na.omit,family = binomial(link = "logit")))

##########################################
#             CLASSIFICACAO
#              NAIVE BAYES
##########################################
# 1 - Y~sage+sex+painexer+relrest+cp+chol+cigs+years+fbs+prop+nitr+pro+proto+thaldur+met+
thalach+thalrest+exang+oldpeak+slope+rldv5e+thal+thalsev+lmt+ladprox+laddist+diag+
cxmain+ramus+om1+om2+rcaprox+rcadist+lvx3+lvx4+lvf+country
# 2 - Y~age+sex+htn+dig+prop+nitr+diuretic+exang+exang+xhypo+oldpeak+slope+ladprox+
laddist+diag+cxmain+om1+rcaprox+rcadist+PAM+RC
# 3 - Y~painexer+famhist+thalrest+exang+oldpeak
# 4 - Y ~ sex+painloc+trestbps+htn+famhist+restecg+dig+prop+nitr+pro+diuretic+thaldur+
thalach+tpeakbps+tpeakbpd+trestbpd+exang+xhypo+oldpeak+slope+rldv5+laddist+country+PAM+
IPPA+IPPARC+HM+ALPHA+ALPHA2
# 5 - Y ~ sex+painloc+htn+famhist+restecg+dig+nitr+pro+diuretic+thaldur+
tpeakbps+trestbpd+exang+xhypo+oldpeak+slope+rldv5+laddist+ALPHA

modNB1=naiveBayes(Y~age+sex+painexer+relrest+cp+chol+cigs+years+fbs+prop+nitr+pro+proto+thaldur+met+
                    thalach+thalrest+exang+oldpeak+slope+rldv5e+thal+thalsev+lmt+ladprox+laddist+diag+
                    cxmain+ramus+om1+om2+rcaprox+rcadist+lvx3+lvx4+lvf+country,data = dadostrei)
YestNB1=predict(modNB1,newdata = dadostest);table(dadostest$Y,YestNB1)
sum(dadostest$Y==YestNB1)/length(dadostest$Y)*100

modNB2=naiveBayes(Y~age+sex+htn+dig+prop+nitr+diuretic+exang+exang+xhypo+oldpeak+slope+ladprox+
                    laddist+diag+cxmain+om1+rcaprox+rcadist+PAM+RC,data = dadostrei)
YestNB2=predict(modNB2,newdata = dadostest);table(dadostest$Y,YestNB2)
sum(dadostest$Y==YestNB2)/length(dadostest$Y)*100

modNB3=naiveBayes(Y~painexer+famhist+thalrest+exang+oldpeak,data = dadostrei) #### MELHOR
YestNB3=predict(modNB3,newdata = dadostest);table(dadostest$Y,YestNB3)
sum(dadostest$Y==YestNB3)/length(dadostest$Y)*100

modNB4=naiveBayes(Y ~ sex+painloc+trestbps+htn+famhist+restecg+dig+prop+nitr+pro+diuretic+thaldur+
                    thalach+tpeakbps+tpeakbpd+trestbpd+exang+xhypo+oldpeak+slope+rldv5+laddist+country+PAM+
                    IPPA+IPPARC+HM+ALPHA+ALPHA2,data = dadostrei)
YestNB4=predict(modNB4,newdata = dadostest);table(dadostest$Y,YestNB4)
sum(dadostest$Y==YestNB4)/length(dadostest$Y)*100

#sensibilidade - proporção de verdadeiros cardiopatas
round(table(dadostest$Y,YestNB4)[2,2]/sum(table(dadostest$Y,YestNB4)[2,])*100,2)
#Especificidade - proporção de verdadeiros não cardiopatas
round(table(dadostest$Y,YestNB4)[1,1]/sum(table(dadostest$Y,YestNB4)[1,])*100,2)
#Verdadeiro Valor Positivo - proporção de verdadeiros cardiopatas condicionados aos indivíduos com cardiopatia
round(table(dadostest$Y,YestNB4)[2,2]/sum(table(dadostest$Y,YestNB4)[,2])*100,2)


modNB5=naiveBayes(Y ~ sex+painloc+htn+famhist+restecg+dig+nitr+pro+diuretic+thaldur+
                    tpeakbps+trestbpd+exang+xhypo+oldpeak+slope+rldv5+laddist+ALPHA,data = dadostrei)
YestNB5=predict(modNB5,newdata = dadostest);table(dadostest$Y,YestNB5)
sum(dadostest$Y==YestNB5)/length(dadostest$Y)*100

modNB6=naiveBayes(Y~age+sex+years+fbs+famhist+restecg+dig+nitr+diuretic+thaldur+exang+oldpeak+rldv5e+
                    ca+thal+ladprox+laddist+cxmain+om1+rcaprox+rcadist+lvx3+lvx4+lvf+PAM+IPPA+RC,data = dadostrei)
YestNB6=predict(modNB6,newdata = dadostest);table(dadostest$Y,YestNB6)
sum(dadostest$Y==YestNB6)/length(dadostest$Y)*100


##########################################
#             CLASSIFICACAO
#           ARVORE DE DECISAO
##########################################
# 1 - Y~sex+cp+htn+fbs+famhist+dig+prop+nitr+pro+diuretic+thaldur+thaltime+met+thalach+
exang+xhypo+oldpeak+slope+ca+thal+ladprox+laddist+cxmain+om1+rcaprox+rcadist+lvx4
# 2 - Y~age+sex+cp+htn+chol+cigs+years+fbs+famhist+restecg+dig+prop+nitr+pro+diuretic+
thaldur+met+thalach+thalrest+tpeakbps+tpeakbpd+exang+oldpeak+slope+rldv5e+ca+thal+lmt+
  ladprox+laddist+cxmain+om1+rcaprox+rcadist+lvx3+lvx4+lvf+PAM+IPPA+HM+ALPHA
# 3 - Y~age+sex+cp+trestbps+dig+prop+thalach+tpeakbps+exang+oldpeak+ca+thal+ladprox+laddist
# 4 - Y~sex+trestbps+tpeakbps+oldpeak+ca+thal
# 5 - Y~age+sex+trestbps+years+fbs+famhist+restecg+dig+nitr+diuretic+thaldur+met+thalrest+
trestbpd+exang+oldpeak+rldv5e+ca+thal+lmt+ladprox+laddist+cxmain+om1+rcaprox+rcadist+
  lvx3+lvx4+lvf+PAM+IPPA+RC+ALPHA2
# 6 - Y~age+sex+years+fbs+famhist+restecg+dig+nitr+diuretic+thaldur+exang+oldpeak+rldv5e+
ca+thal+ladprox+laddist+cxmain+om1+rcaprox+rcadist+lvx3+lvx4+lvf+PAM+IPPA+RC


DadostestRF1=na.omit(data.frame(dadostest$Y,dadostest$sex,dadostest$cp,dadostest$htn,dadostest$fbs,
                                dadostest$famhist,dadostest$dig,dadostest$prop,
                                dadostest$nitr,dadostest$pro,dadostest$diuretic,dadostest$thaldur,
                                dadostest$thaltime,dadostest$met,dadostest$thalach,
                                dadostest$exang,dadostest$xhypo,dadostest$oldpeak,
                                dadostest$slope,dadostest$ca,dadostest$thal,dadostest$ladprox,
                                dadostest$laddist,dadostest$cxmain,dadostest$om1,dadostest$rcaprox,
                                dadostest$rcadist,dadostest$lvx4))
randomForest(Y~sex+cp+htn+fbs+famhist+dig+prop+nitr+pro+
               diuretic+thaldur+thaltime+met+thalach+exang+
               xhypo+oldpeak+slope+ca+thal+ladprox+laddist+
               cxmain+om1+rcaprox+rcadist+lvx4,
             data = dadostrei,xtest=DadostestRF1[,-1],ytest=DadostestRF1$dadostest.Y,na.action = na.omit)

100-5.17

DadostestRF2=na.omit(data.frame(dadostest$Y,dadostest$age,dadostest$sex,dadostest$cp,
                                dadostest$htn,dadostest$chol,dadostest$cigs,dadostest$years,dadostest$fbs,
                                dadostest$famhist,dadostest$restecg,dadostest$dig,dadostest$prop,dadostest$nitr,dadostest$pro,
                                dadostest$diuretic,dadostest$thaldur,dadostest$met,dadostest$thalach,dadostest$thalrest,
                                dadostest$tpeakbps,dadostest$tpeakbpd,dadostest$exang,dadostest$oldpeak,dadostest$slope,
                                dadostest$rldv5e,dadostest$ca,dadostest$thal,dadostest$lmt,dadostest$ladprox,dadostest$laddist,
                                dadostest$cxmain,dadostest$om1,dadostest$rcaprox,dadostest$rcadist,dadostest$lvx3,
                                dadostest$lvx4,dadostest$lvf,dadostest$PAM,dadostest$IPPA,dadostest$HM,dadostest$ALPHA))
randomForest(Y~age+sex+cp+htn+chol+cigs+years+fbs+famhist+restecg+dig+prop+nitr+pro+diuretic+
               thaldur+met+thalach+thalrest+tpeakbps+tpeakbpd+exang+oldpeak+slope+rldv5e+ca+thal+lmt+
               ladprox+laddist+cxmain+om1+rcaprox+rcadist+lvx3+lvx4+lvf+PAM+IPPA+HM+ALPHA,
             data = dadostrei,xtest=DadostestRF2[,-1],ytest=DadostestRF2$dadostest.Y,na.action = na.omit)
100-4.94

DadostestRF3=na.omit(data.frame(dadostest$Y,dadostest$age,dadostest$sex,dadostest$cp,dadostest$trestbps,
           dadostest$dig,dadostest$prop,dadostest$thalach,dadostest$tpeakbps,dadostest$exang,dadostest$oldpeak,
           dadostest$ca,dadostest$thal,dadostest$ladprox,dadostest$laddist))
randomForest(Y~age+sex+cp+trestbps+dig+prop+thalach+tpeakbps+exang+oldpeak+ca+thal+ladprox+laddist,
             data = dadostrei,xtest=DadostestRF3[,-1],ytest=DadostestRF3$dadostest.Y,na.action = na.omit)
100-16.87

DadostestRF4=na.omit(data.frame(dadostest$Y,dadostest$sex,dadostest$trestbps,dadostest$tpeakbps,dadostest$oldpeak,
                                dadostest$ca,dadostest$thal))
randomForest(Y~sex+trestbps+tpeakbps+oldpeak+ca+thal,data = dadostrei,xtest=DadostestRF4[,-1],
             ytest=DadostestRF4$dadostest.Y,na.action = na.omit)
100-22.89

DadostestRF5=na.omit(data.frame(dadostest$Y,dadostest$age,dadostest$sex,dadostest$trestbps,dadostest$years,dadostest$fbs,
                                dadostest$famhist,dadostest$restecg,dadostest$dig,dadostest$nitr,dadostest$diuretic,
                                dadostest$thaldur,dadostest$met,dadostest$thalrest,dadostest$trestbpd,dadostest$exang,dadostest$oldpeak,
                                dadostest$rldv5e,dadostest$ca,dadostest$thal,dadostest$lmt,dadostest$ladprox,dadostest$laddist,
                                dadostest$cxmain,dadostest$om1,dadostest$rcaprox,dadostest$rcadist,dadostest$lvx3,
                                dadostest$lvx4,dadostest$lvf,dadostest$PAM,dadostest$IPPA,dadostest$RC,dadostest$ALPHA2))
randomForest(Y~age+sex+trestbps+years+fbs+famhist+restecg+dig+nitr+diuretic+thaldur+met+thalrest+
               trestbpd+exang+oldpeak+rldv5e+ca+thal+lmt+ladprox+laddist+cxmain+om1+rcaprox+rcadist+
               lvx3+lvx4+lvf+PAM+IPPA+RC+ALPHA2,data = dadostrei,xtest=DadostestRF5[,-1],ytest=DadostestRF5$dadostest.Y,na.action = na.omit)
100-7.32

DadostestRF6=na.omit(data.frame(dadostest$Y,dadostest$age,dadostest$sex,dadostest$years,dadostest$fbs,
                                dadostest$famhist,dadostest$restecg,dadostest$dig,dadostest$nitr,dadostest$diuretic,
                                dadostest$thaldur,dadostest$exang,dadostest$oldpeak,dadostest$rldv5e,dadostest$ca,dadostest$thal,
                                dadostest$ladprox,dadostest$laddist,dadostest$cxmain,dadostest$om1,dadostest$rcaprox,
                                dadostest$rcadist,dadostest$lvx3,dadostest$lvx4,dadostest$lvf,dadostest$PAM,dadostest$IPPA,dadostest$RC))
randomForest(Y~age+sex+years+fbs+famhist+restecg+dig+nitr+diuretic+thaldur+exang+oldpeak+rldv5e+
             ca+thal+ladprox+laddist+cxmain+om1+rcaprox+rcadist+lvx3+lvx4+lvf+PAM+IPPA+RC,
             data = dadostrei,xtest=DadostestRF6[,-1],ytest=DadostestRF6$dadostest.Y,na.action = na.omit)

100-2.44  ##MELHOR

##########################################
#             CLASSIFICACAO
#     MAQUINA DE VETORES DE SUPORTE
##########################################
# 1 - Y~sex+cp+htn+fbs+famhist+dig+prop+nitr+pro+diuretic+thaldur+thaltime+met+thalach+
exang+xhypo+oldpeak+slope+ca+thal+ladprox+laddist+cxmain+om1+rcaprox+rcadist+lvx4
# 2 - Y~age+sex+cp+htn+chol+cigs+years+fbs+famhist+restecg+dig+prop+nitr+pro+diuretic+
thaldur+met+thalach+thalrest+tpeakbps+tpeakbpd+exang+oldpeak+slope+rldv5e+ca+thal+lmt+
  ladprox+laddist+cxmain+om1+rcaprox+rcadist+lvx3+lvx4+lvf+PAM+IPPA+HM+ALPHA
# 3 - Y~age+sex+cp+trestbps+dig+prop+thalach+tpeakbps+exang+oldpeak+ca+thal+ladprox+laddist
# 4 - Y~sex+trestbps+tpeakbps+oldpeak+ca+thal
# 5 - Y~age+sex+trestbps+years+fbs+famhist+restecg+dig+nitr+diuretic+thaldur+met+thalrest+
trestbpd+exang+oldpeak+rldv5e+ca+thal+lmt+ladprox+laddist+cxmain+om1+rcaprox+rcadist+
  lvx3+lvx4+lvf+PAM+IPPA+RC+ALPHA2
# 6 - Y~age+sex+years+fbs+famhist+restecg+dig+nitr+diuretic+thaldur+exang+oldpeak+rldv5e+
ca+thal+ladprox+laddist+cxmain+om1+rcaprox+rcadist+lvx3+lvx4+lvf+PAM+IPPA+RC


modSVM1=svm(Y~sex+cp+htn+fbs+famhist+dig+prop+nitr+pro+diuretic+thaldur+thaltime+met+thalach+
              exang+xhypo+oldpeak+slope+ca+thal+ladprox+laddist+cxmain+om1+rcaprox+rcadist+lvx4,data=dadostrei,
            scale=F,kernel="radial",cost=1000,epsilon=1.0e-12, sigma=5)
YestSVM1=predict(modSVM1,dadostest);table(Y[as.numeric(names(YestSVM1))],YestSVM1)
sum(Y[as.numeric(names(YestSVM1))]==YestSVM1)/length(YestSVM1)*100

modSVM2=svm(Y~age+sex+cp+htn+chol+cigs+years+fbs+famhist+restecg+dig+prop+nitr+pro+diuretic+
              thaldur+met+thalach+thalrest+tpeakbps+tpeakbpd+exang+oldpeak+slope+rldv5e+ca+thal+lmt+
              ladprox+laddist+cxmain+om1+rcaprox+rcadist+lvx3+lvx4+lvf+PAM+IPPA+HM+ALPHA,data=dadostrei,
            scale=F,kernel="poly",cost=100,epsilon=1.0e-12)
YestSVM2=predict(modSVM2,dadostest);table(Y[as.numeric(names(YestSVM2))],YestSVM2)
sum(Y[as.numeric(names(YestSVM2))]==YestSVM2)/length(YestSVM2)*100

modSVM3=svm(Y~age+sex+cp+trestbps+dig+prop+thalach+tpeakbps+exang+oldpeak+ca+thal+ladprox+laddist,
            scale=F,data=dadostrei,kernel="poly",cost=100,epsilon=1.0e-12)
YestSVM3=predict(modSVM3,dadostest);table(Y[as.numeric(names(YestSVM3))],YestSVM3)
sum(Y[as.numeric(names(YestSVM3))]==YestSVM3)/length(YestSVM3)*100

modSVM4=svm(Y~sex+trestbps+tpeakbps+oldpeak+ca+thal,data=dadostrei,scale=F,kernel="poly",cost=100,epsilon=1.0e-12)
YestSVM4=predict(modSVM4,dadostest);table(Y[as.numeric(names(YestSVM4))],YestSVM4)
sum(Y[as.numeric(names(YestSVM4))]==YestSVM4)/length(YestSVM4)*100

modSVM5=svm(Y~age+sex+trestbps+years+fbs+famhist+restecg+dig+nitr+diuretic+thaldur+met+thalrest+
              trestbpd+exang+oldpeak+rldv5e+ca+thal+lmt+ladprox+laddist+cxmain+om1+rcaprox+rcadist+
              lvx3+lvx4+lvf+PAM+IPPA+RC+ALPHA2,data=dadostrei,scale=F,kernel="poly",cost=100,epsilon=1.0e-12)
YestSVM5=predict(modSVM5,dadostest);table(Y[as.numeric(names(YestSVM5))],YestSVM5)
sum(Y[as.numeric(names(YestSVM5))]==YestSVM5)/length(YestSVM5)*100

modSVM6=svm(Y~age+sex+years+fbs+famhist+restecg+dig+nitr+diuretic+thaldur+exang+oldpeak+rldv5e+
              ca+thal+ladprox+laddist+cxmain+om1+rcaprox+rcadist+lvx3+lvx4+lvf+PAM+IPPA+RC,
            scale=F,data=dadostrei,kernel="poly",cost=100,epsilon=1.0e-12)
YestSVM6=predict(modSVM6,dadostest);table(Y[as.numeric(names(YestSVM6))],YestSVM6)
sum(Y[as.numeric(names(YestSVM6))]==YestSVM6)/length(YestSVM6)*100

##########################################
#             CLASSIFICACAO
#     REGRESSAO LINEAR - LOGISTICA
##########################################
# 1 - Y~sex+cp+htn+fbs+famhist+dig+prop+nitr+pro+diuretic+thaldur+thaltime+met+thalach+
exang+xhypo+oldpeak+slope+ca+thal+ladprox+laddist+cxmain+om1+rcaprox+rcadist+lvx4
# 2 - Y~age+sex+cp+htn+chol+cigs+years+fbs+famhist+restecg+dig+prop+nitr+pro+diuretic+
thaldur+met+thalach+thalrest+tpeakbps+tpeakbpd+exang+oldpeak+slope+rldv5e+ca+thal+lmt+
  ladprox+laddist+cxmain+om1+rcaprox+rcadist+lvx3+lvx4+lvf+PAM+IPPA+HM+ALPHA
# 3 - Y~age+sex+cp+trestbps+dig+prop+thalach+tpeakbps+exang+oldpeak+ca+thal+ladprox+laddist
# 4 - Y~sex+trestbps+tpeakbps+oldpeak+ca+thal
# 5 - Y~age+sex+trestbps+years+fbs+famhist+restecg+dig+nitr+diuretic+thaldur+met+thalrest+
trestbpd+exang+oldpeak+rldv5e+ca+thal+lmt+ladprox+laddist+cxmain+om1+rcaprox+rcadist+
  lvx3+lvx4+lvf+PAM+IPPA+RC+ALPHA2
# 6 - Y~age+sex+years+fbs+famhist+restecg+dig+nitr+diuretic+thaldur+exang+oldpeak+rldv5e+
ca+thal+ladprox+laddist+cxmain+om1+rcaprox+rcadist+lvx3+lvx4+lvf+PAM+IPPA+RC

modelo0=glm(Y~sex+cp+htn+fbs+famhist+dig+prop+nitr+pro+diuretic+thaldur+thaltime+met+thalach+
              exang+xhypo+oldpeak+slope+ca+thal+ladprox+laddist+cxmain+om1+rcaprox+rcadist+lvx4,
            data=dadostrei,family = binomial(link = "logit"),na.action = na.omit)
Yestmod0=predict.glm(modelo0,newdata = dadostest,type = "response");Ychapmod0=ifelse(Yestmod0<0.5,0,1)
sum(na.omit(dadostest$Y==Ychapmod0))/length(na.omit(Ychapmod0))*100;table(dadostest$Y,Ychapmod0)

#sensibilidade - proporção de verdadeiros cardiopatas
round(table(dadostest$Y,Ychapmod0)[2,2]/sum(table(dadostest$Y,Ychapmod0)[2,])*100,2)
#Especificidade - proporção de verdadeiros não cardiopatas
round(table(dadostest$Y,Ychapmod0)[1,1]/sum(table(dadostest$Y,Ychapmod0)[1,])*100,2)
#Verdadeiro Valor Positivo - proporção de verdadeiros cardiopatas condicionados aos indivíduos com cardiopatia
round(table(dadostest$Y,Ychapmod0)[2,2]/sum(table(dadostest$Y,Ychapmod0)[,2])*100,2)


modelo1=glm(Y~age+sex+cp+htn+chol+cigs+years+fbs+famhist+restecg+dig+prop+nitr+pro+diuretic+
              thaldur+met+thalach+thalrest+tpeakbps+tpeakbpd+exang+oldpeak+slope+rldv5e+ca+thal+lmt+
              ladprox+laddist+cxmain+om1+rcaprox+rcadist+lvx3+lvx4+lvf+PAM+IPPA+HM+ALPHA,
            data=dadostrei,family = binomial(link = "logit"))
Yestmod1=predict.glm(modelo1,newdata = dadostest,type = "response");Ychapmod1=ifelse(Yestmod1<0.5,0,1)
sum(na.omit(dadostest$Y==Ychapmod1))/length(na.omit(Ychapmod1))*100;table(dadostest$Y,Ychapmod1)

#sensibilidade - proporção de verdadeiros cardiopatas
round(table(dadostest$Y,Ychapmod1)[2,2]/sum(table(dadostest$Y,Ychapmod1)[2,])*100,2)
#Especificidade - proporção de verdadeiros não cardiopatas
round(table(dadostest$Y,Ychapmod1)[1,1]/sum(table(dadostest$Y,Ychapmod1)[1,])*100,2)
#Verdadeiro Valor Positivo - proporção de verdadeiros cardiopatas condicionados aos indivíduos com cardiopatia
round(table(dadostest$Y,Ychapmod1)[2,2]/sum(table(dadostest$Y,Ychapmod1)[,2])*100,2)


modeloANOVA1=glm(Y~age+sex+cp+trestbps+dig+prop+thalach+tpeakbps+exang+oldpeak+ca+thal+ladprox+laddist,
                 data=dadostrei,family = binomial(link = "logit"))
YestAOV1=predict.glm(modeloANOVA1,newdata = dadostest,type = "response");YchapAOV1=ifelse(YestAOV1<0.5,0,1)
sum(na.omit(dadostest$Y==YchapAOV1))/length(na.omit(YchapAOV1))*100;table(dadostest$Y,YchapAOV1)

modeloANOVA2=glm(Y~sex+trestbps+tpeakbps+oldpeak+ca+thal,
                 data=dadostrei,family = binomial(link = "logit"))
YestAOV2=predict.glm(modeloANOVA2,newdata = dadostest,type = "response");YchapAOV2=ifelse(YestAOV2<0.5,0,1)
sum(na.omit(dadostest$Y==YchapAOV2))/length(na.omit(YchapAOV2))*100;table(dadostest$Y,YchapAOV2)

modeloAIC1=glm(Y~age+sex+trestbps+years+fbs+famhist+restecg+dig+nitr+diuretic+thaldur+met+thalrest+
                 trestbpd+exang+oldpeak+rldv5e+ca+thal+lmt+ladprox+laddist+cxmain+om1+rcaprox+rcadist+
                 lvx3+lvx4+lvf+PAM+IPPA+RC+ALPHA2,data=dadostrei,family = binomial(link = "logit"))
YestAIC1=predict.glm(modeloAIC1,newdata = dadostest,type = "response");YchapAIC1=ifelse(YestAIC1<0.5,0,1)
sum(na.omit(dadostest$Y==YchapAIC1))/length(na.omit(YchapAIC1))*100;table(dadostest$Y,YchapAIC1)

modeloAIC2=glm(Y~age+sex+years+fbs+famhist+restecg+dig+nitr+diuretic+thaldur+exang+oldpeak+rldv5e+
                 ca+thal+ladprox+laddist+cxmain+om1+rcaprox+rcadist+lvx3+lvx4+lvf+PAM+IPPA+RC,
               family=binomial(link="logit"),data=dadostrei)
YestAIC2=predict.glm(modeloAIC2,newdata = dadostest,type = "response");YchapAIC2=ifelse(YestAIC2<0.5,0,1)
sum(na.omit(dadostest$Y==YchapAIC2))/length(na.omit(YchapAIC2))*100;table(dadostest$Y,YchapAIC2)


##########################################
#             CLASSIFICACAO
#           REDE NEURAL - MLP
##########################################
# 1 - Y~sex+cp+htn+fbs+dm+famhist+dig+prop+nitr+pro+diuretic+thaldur+thaltime+met+thalach+
exang+xhypo+oldpeak+slope+ca+thal+ladprox+laddist+cxmain+om1+rcaprox+rcadist+lvx4
# 2 - Y~age+sex+cp+htn+chol+cigs+years+fbs+famhist+restecg+dig+prop+nitr+pro+diuretic+
thaldur+met+thalach+thalrest+tpeakbps+tpeakbpd+exang+oldpeak+slope+rldv5e+ca+thal+lmt+
  ladprox+laddist+cxmain+om1+rcaprox+rcadist+lvx3+lvx4+lvf+PAM+IPPA+HM+ALPHA
# 3 - Y~age+sex+cp+trestbps+dig+prop+thalach+tpeakbps+exang+oldpeak+ca+thal+ladprox+laddist
# 4 - Y~sex+trestbps+tpeakbps+oldpeak+ca+thal
# 5 - Y~age+sex+trestbps+years+fbs+famhist+restecg+dig+nitr+diuretic+thaldur+met+thalrest+
trestbpd+exang+oldpeak+rldv5e+ca+thal+lmt+ladprox+laddist+cxmain+om1+rcaprox+rcadist+
  lvx3+lvx4+lvf+PAM+IPPA+RC+ALPHA2
# 6 - Y~age+sex+years+fbs+famhist+restecg+dig+nitr+diuretic+thaldur+exang+oldpeak+rldv5e+
ca+thal+ladprox+laddist+cxmain+om1+rcaprox+rcadist+lvx3+lvx4+lvf+PAM+IPPA+RC


XRN=cbind(as.numeric(sex),age,as.numeric(htn),PAM,IPPA,IPPARC,HM,ALPHA)
YRN=ifelse(Y==0,0,1)
RNMLP=mlp(XRN[treinamento,],YRN[treinamento],inputsTest = XRN[-treinamento,], targetsTest =YRN[-treinamento])
names(RNMLP)
RNMLP$IterativeTestError
RNMLP$IterativeFitError
RNMLP$fittedTestValues

##########################################
#             CLASSIFICACAO
#               ADABOOST
##########################################
# 1 - Y~sex+cp+htn+fbs+famhist+dig+prop+nitr+pro+diuretic+thaldur+thaltime+met+thalach+
exang+xhypo+oldpeak+slope+ca+thal+ladprox+laddist+cxmain+om1+rcaprox+rcadist+lvx4
# 2 - Y~age+sex+cp+htn+chol+cigs+years+fbs+famhist+restecg+dig+prop+nitr+pro+diuretic+
thaldur+met+thalach+thalrest+tpeakbps+tpeakbpd+exang+oldpeak+slope+rldv5e+ca+thal+lmt+
  ladprox+laddist+cxmain+om1+rcaprox+rcadist+lvx3+lvx4+lvf+PAM+IPPA+HM+ALPHA
# 3 - Y~age+sex+cp+trestbps+dig+prop+thalach+tpeakbps+exang+oldpeak+ca+thal+ladprox+laddist
# 4 - Y~sex+trestbps+tpeakbps+oldpeak+ca+thal
# 5 - Y~age+sex+trestbps+years+fbs+famhist+restecg+dig+nitr+diuretic+thaldur+met+thalrest+
trestbpd+exang+oldpeak+rldv5e+ca+thal+lmt+ladprox+laddist+cxmain+om1+rcaprox+rcadist+
  lvx3+lvx4+lvf+PAM+IPPA+RC+ALPHA2
# 6 - Y~age+sex+years+fbs+famhist+restecg+dig+nitr+diuretic+thaldur+exang+oldpeak+rldv5e+
ca+thal+ladprox+laddist+cxmain+om1+rcaprox+rcadist+lvx3+lvx4+lvf+PAM+IPPA+RC


modADAB1=adaboost(Y~sex+cp+htn+fbs+famhist+dig+prop+nitr+pro+diuretic+thaldur+thaltime+met+thalach+
                    exang+xhypo+oldpeak+slope+ca+thal+ladprox+laddist+cxmain+om1+rcaprox+rcadist+lvx4
                  ,data=dadostrei,10)
YestADAB1=predict(modADAB1,newdata=dadostest)$class;table(dadostest$Y,YestADAB1)
sum(dadostest$Y==YestADAB1)/length(dadostest$Y)*100

modADAB2=adaboost(Y~age+sex+cp+htn+chol+cigs+years+fbs+famhist+restecg+dig+prop+nitr+pro+diuretic+
                    thaldur+met+thalach+thalrest+tpeakbps+tpeakbpd+exang+oldpeak+slope+rldv5e+ca+thal+lmt+
                    ladprox+laddist+cxmain+om1+rcaprox+rcadist+lvx3+lvx4+lvf+PAM+IPPA+HM+ALPHA
                  ,data=dadostrei,10)
YestADAB2=predict(modADAB2,newdata=dadostest)$class;table(dadostest$Y,YestADAB2)
sum(dadostest$Y==YestADAB2)/length(dadostest$Y)*100 ##MELHOR

modADAB3=adaboost(Y~age+sex+cp+trestbps+dig+prop+thalach+tpeakbps+exang+oldpeak+ca+thal+ladprox+laddist
                  ,data=dadostrei,10)
YestADAB3=predict(modADAB3,newdata=dadostest)$class;table(dadostest$Y,YestADAB3)
sum(dadostest$Y==YestADAB3)/length(dadostest$Y)*100

modADAB4=adaboost(Y~sex+trestbps+tpeakbps+oldpeak+ca+thal,data=dadostrei,10)
YestADAB4=predict(modADAB4,newdata=dadostest)$class;table(dadostest$Y,YestADAB4)
sum(dadostest$Y==YestADAB4)/length(dadostest$Y)*100

modADAB5=adaboost(Y~age+sex+trestbps+years+fbs+famhist+restecg+dig+nitr+diuretic+thaldur+met+thalrest+
                    trestbpd+exang+oldpeak+rldv5e+ca+thal+lmt+ladprox+laddist+cxmain+om1+rcaprox+rcadist+
                    lvx3+lvx4+lvf+PAM+IPPA+RC+ALPHA2,data=dadostrei,10)
YestADAB5=predict(modADAB5,newdata=dadostest)$class;table(dadostest$Y,YestADAB5)
sum(dadostest$Y==YestADAB5)/length(dadostest$Y)*100

modADAB6=adaboost(Y~age+sex+years+fbs+famhist+restecg+dig+nitr+diuretic+thaldur+exang+oldpeak+rldv5e+
                    ca+thal+ladprox+laddist+cxmain+om1+rcaprox+rcadist+lvx3+lvx4+lvf+PAM+IPPA+RC,
                  data=dadostrei,10)
YestADAB6=predict(modADAB6,newdata=dadostest)$class;table(dadostest$Y,YestADAB6)
sum(dadostest$Y==YestADAB6)/length(dadostest$Y)*100

#sensibilidade - proporção de verdadeiros cardiopatas
round(table(dadostest$Y,YestADAB6)[2,2]/sum(table(dadostest$Y,YestADAB6)[2,])*100,2)
#Especificidade - proporção de verdadeiros não cardiopatas
round(table(dadostest$Y,YestADAB6)[1,1]/sum(table(dadostest$Y,YestADAB6)[1,])*100,2)
#Verdadeiro Valor Positivo - proporção de verdadeiros cardiopatas condicionados aos indivíduos com cardiopatia
round(table(dadostest$Y,YestADAB6)[2,2]/sum(table(dadostest$Y,YestADAB6)[,2])*100,2)


roc.curve(dadostest$Y,YchapAOV)


ggroc(data = cbind(YchapAOV,dadostest$Y), bin = 0.01, roccol = "green", sp = 19, output = "roc.pdf")


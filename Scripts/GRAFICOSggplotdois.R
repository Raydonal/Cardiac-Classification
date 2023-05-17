rm(list=ls())
library(plyr)
library(ggplot2)
library(reshape2)

# scale_fill_grey()+ #deixar na escala cinza

##################################
#		CLEVELAND		#
##################################

#### CENARIO 1
dados<-read.table("acuraciaClevcenumtrein.txt",header = T)
attach(dados)

limits <- aes(ymax = dados$Acuracias+dados$Desvios,ymin=dados$Acuracias-dados$Desvios)

ggplot(dados, aes(Modelos, Acuracias, fill =  Classificadores)) +
  geom_bar(position="dodge", stat="identity") +
  geom_errorbar(limits, position="dodge") +
  labs(x="Modelos", y="Acurácias", title="Treinamento",fill="Classificadores") +
  scale_fill_manual(values=c("lightpink", "plum3", "darkturquoise","seagreen2", "gold")) +
  theme_bw()


dados<-read.table("acuraciaClevcenumtest.txt",header = T)
attach(dados)

limits <- aes(ymax = dados$Acuracias+dados$Desvios,ymin=dados$Acuracias-dados$Desvios)

ggplot(dados, aes(Modelos, Acuracias, fill =  Classificadores)) +
  geom_bar(position="dodge", stat="identity") +
  geom_errorbar(limits, position="dodge") +
  labs(x="Modelos", y="Acurácias", title="Teste",fill="Classificadores") +
  scale_fill_manual(values=c("lightpink", "plum3", "darkturquoise","seagreen2", "gold")) +
  theme_bw()


#### CENARIO 2
dados<-read.table("acuraciaClevcendoistrein.txt",
header = T)
attach(dados)

limits <- aes(ymax = dados$Acuracias+dados$Desvios,ymin=dados$Acuracias-dados$Desvios)

ggplot(dados, aes(Modelos, Acuracias, fill =  Classificadores)) +
  geom_bar(position="dodge", stat="identity") +
  geom_errorbar(limits, position="dodge") +
  labs(x="Modelos", y="Acurácias", title="Treinamento",fill="Classificadores") +
  scale_fill_manual(values=c("lightpink", "plum3", "darkturquoise","seagreen2", "gold")) +
  theme_bw()


dados<-read.table("acuraciaClevcendoistest.txt",header = T)
attach(dados)

limits <- aes(ymax = dados$Acuracias+dados$Desvios,ymin=dados$Acuracias-dados$Desvios)

ggplot(dados, aes(Modelos, Acuracias, fill =  Classificadores)) +
  geom_bar(position="dodge", stat="identity") +
  geom_errorbar(limits, position="dodge") +
  labs(x="Modelos", y="Acurácias", title="Teste",fill="Classificadores") +
  scale_fill_manual(values=c("lightpink", "plum3", "darkturquoise","seagreen2", "gold")) +
  theme_bw()




##################################
#		HUNGARIAN		#
##################################

#### CENARIO 1
dados<-read.table("acuraciaHungcenumtrein.txt",header = T)
attach(dados)

limits <- aes(ymax = dados$Acuracias+dados$Desvios,ymin=dados$Acuracias-dados$Desvios)

ggplot(dados, aes(Modelos, Acuracias, fill =  Classificadores)) +
  geom_bar(position="dodge", stat="identity") +
  geom_errorbar(limits, position="dodge") +
  labs(x="Modelos", y="Acurácias", title="Treinamento",fill="Classificadores") +
  scale_fill_manual(values=c("lightpink", "plum3", "darkturquoise","seagreen2", "gold")) +
  theme_bw()


dados<-read.table("acuraciaHungcenumtest.txt",header = T)
attach(dados)

limits <- aes(ymax = dados$Acuracias+dados$Desvios,ymin=dados$Acuracias-dados$Desvios)

ggplot(dados, aes(Modelos, Acuracias, fill =  Classificadores)) +
  geom_bar(position="dodge", stat="identity") +
  geom_errorbar(limits, position="dodge") +
  labs(x="Modelos", y="Acurácias", title="Teste",fill="Classificadores") +
  scale_fill_manual(values=c("lightpink", "plum3", "darkturquoise","seagreen2", "gold")) +
  theme_bw()


#### CENARIO 2
dados<-read.table("acuraciaHungcendoistrein.txt",
header = T)
attach(dados)

limits <- aes(ymax = dados$Acuracias+dados$Desvios,ymin=dados$Acuracias-dados$Desvios)

ggplot(dados, aes(Modelos, Acuracias, fill =  Classificadores)) +
  geom_bar(position="dodge", stat="identity") +
  geom_errorbar(limits, position="dodge") +
  labs(x="Modelos", y="Acurácias", title="Treinamento",fill="Classificadores") +
  scale_fill_manual(values=c("lightpink", "plum3", "darkturquoise","seagreen2", "gold")) +
  theme_bw()


dados<-read.table("acuraciaHungcendoistest.txt",header = T)
attach(dados)

limits <- aes(ymax = dados$Acuracias+dados$Desvios,ymin=dados$Acuracias-dados$Desvios)

ggplot(dados, aes(Modelos, Acuracias, fill =  Classificadores)) +
  geom_bar(position="dodge", stat="identity") +
  geom_errorbar(limits, position="dodge") +
  labs(x="Modelos", y="Acurácias", title="Teste",fill="Classificadores") +
  scale_fill_manual(values=c("lightpink", "plum3", "darkturquoise","seagreen2", "gold")) +
  theme_bw()




##################################
#		LONG BEACH		#
##################################

#### CENARIO 1
dados<-read.table("acuraciaLongBcenumtrein.txt",header = T)
attach(dados)

limits <- aes(ymax = dados$Acuracias+dados$Desvios,ymin=dados$Acuracias-dados$Desvios)

ggplot(dados, aes(Modelos, Acuracias, fill =  Classificadores)) +
  geom_bar(position="dodge", stat="identity") +
  geom_errorbar(limits, position="dodge") +
  labs(x="Modelos", y="Acurácias", title="Treinamento",fill="Classificadores") +
  scale_fill_manual(values=c("lightpink", "plum3", "darkturquoise","seagreen2", "gold")) +
  theme_bw()


dados<-read.table("acuraciaLongBcenumtest.txt",header = T)
attach(dados)

limits <- aes(ymax = dados$Acuracias+dados$Desvios,ymin=dados$Acuracias-dados$Desvios)

ggplot(dados, aes(Modelos, Acuracias, fill =  Classificadores)) +
  geom_bar(position="dodge", stat="identity") +
  geom_errorbar(limits, position="dodge") +
  labs(x="Modelos", y="Acurácias", title="Teste",fill="Classificadores") +
  scale_fill_manual(values=c("lightpink", "plum3", "darkturquoise","seagreen2", "gold")) +
  theme_bw()


#### CENARIO 2
dados<-read.table("acuraciaLongBcendoistrein.txt",
header = T)
attach(dados)

limits <- aes(ymax = dados$Acuracias+dados$Desvios,ymin=dados$Acuracias-dados$Desvios)

ggplot(dados, aes(Modelos, Acuracias, fill =  Classificadores)) +
  geom_bar(position="dodge", stat="identity") +
  geom_errorbar(limits, position="dodge") +
  labs(x="Modelos", y="Acurácias", title="Treinamento",fill="Classificadores") +
  scale_fill_manual(values=c("lightpink", "plum3", "darkturquoise","seagreen2", "gold")) +
  theme_bw()


dados<-read.table("acuraciaLongBcendoistest.txt",na.strings = "NA",header = T)
attach(dados)

limits <- aes(ymax = dados$Acuracias+dados$Desvios,ymin=dados$Acuracias-dados$Desvios)

ggplot(dados, aes(Modelos, Acuracias, fill =  Classificadores)) +
  geom_bar(position="dodge", stat="identity") +
  geom_errorbar(limits, position="dodge") +
  labs(x="Modelos", y="Acurácias", title="Teste",fill="Classificadores") +
  scale_fill_manual(values=c("lightpink", "plum3", "darkturquoise","seagreen2", "gold")) +
  theme_bw()




##################################
#		SWITZERLAND		#
##################################

#### CENARIO 1
dados<-read.table("acuraciaSwitcenumtrein.txt",header = T)
attach(dados)

limits <- aes(ymax = dados$Acuracias+dados$Desvios,ymin=dados$Acuracias-dados$Desvios)

ggplot(dados, aes(Modelos, Acuracias, fill =  Classificadores)) +
  geom_bar(position="dodge", stat="identity") +
  geom_errorbar(limits, position="dodge") +
  labs(x="Modelos", y="Acurácias", title="Treinamento",fill="Classificadores") +
  scale_fill_manual(values=c("lightpink", "plum3", "darkturquoise","seagreen2", "gold")) +
  theme_bw()


dados<-read.table("acuraciaSwitcenumtest.txt",header = T)
attach(dados)

limits <- aes(ymax = dados$Acuracias+dados$Desvios,ymin=dados$Acuracias-dados$Desvios)

ggplot(dados, aes(Modelos, Acuracias, fill =  Classificadores)) +
  geom_bar(position="dodge", stat="identity") +
  geom_errorbar(limits, position="dodge") +
  labs(x="Modelos", y="Acurácias", title="Teste",fill="Classificadores") +
  scale_fill_manual(values=c("lightpink", "plum3", "darkturquoise","seagreen2", "gold")) +
  theme_bw()


#### CENARIO 2
dados<-read.table("acuraciaSwitcendoistrein.txt",
header = T)
attach(dados)

limits <- aes(ymax = dados$Acuracias+dados$Desvios,ymin=dados$Acuracias-dados$Desvios)

ggplot(dados, aes(Modelos, Acuracias, fill =  Classificadores)) +
  geom_bar(position="dodge", stat="identity") +
  geom_errorbar(limits, position="dodge") +
  labs(x="Modelos", y="Acurácias", title="Treinamento",fill="Classificadores") +
  scale_fill_manual(values=c("lightpink", "plum3", "darkturquoise","seagreen2", "gold")) +
  theme_bw()


dados<-read.table("acuraciaSwitcendoistest.txt",header = T)
attach(dados)

limits <- aes(ymax = dados$Acuracias+dados$Desvios,ymin=dados$Acuracias-dados$Desvios)

ggplot(dados, aes(Modelos, Acuracias, fill =  Classificadores)) +
  geom_bar(position="dodge", stat="identity") +
  geom_errorbar(limits, position="dodge") +
  labs(x="Modelos", y="Acurácias", title="Teste",fill="Classificadores") +
  scale_fill_manual(values=c("lightpink", "plum3", "darkturquoise","seagreen2", "gold")) +
  theme_bw()




































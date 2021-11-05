
################################################################################
#               INSTALAÇÃO E CARREGAMENTO DE PACOTES NECESSÁRIOS               #
################################################################################
# Pacotes para trabalhar árvovore

pacotes <- c("plotly","tidyverse","reshape2","knitr","kableExtra","tidymodels","ISLR2",
             "modeldata","pROC","vip","glmnet","rpart.plot","jtools","readr","readxl",
             "car","lmtest","stringr", "writexl","ranger","ggplot2","parsnip", "visdat")

if(sum(as.numeric(!pacotes %in% installed.packages())) != 0){
  instalador <- pacotes[!pacotes %in% installed.packages()]
  for(i in 1:length(instalador)) {
    install.packages(instalador, dependencies = T)
    break()}
  sapply(pacotes, require, character = T) 
} else {
  sapply(pacotes, require, character = T) 
}



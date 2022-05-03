# TCC Data Science USP
# Previsão de inadimplência
# Boa Vista 10/02/2022

library(tidyverse)
library(readxl)
library(skimr)
library(kableExtra)
library(tidymodels)
library(readxl)
library(stringr)
library(rpart.plot)
library(rpart)
library(performanceEstimation)


##### 0. Importar base de dados
base_tratada_inadimplentes <- read.csv("db\\DB_tratada_inadimplentes_estudo.csv")
base_tratada_inadimplentes <- base_tratada_inadimplentes[,-1]
base_tratada_inadimplentes <- base_tratada_inadimplentes %>% rename(Estado_Civil = Estado.Civil, Qtd_Processos = Quantidade.Processos)

skim(base_tratada_inadimplentes)

base_tratada_inadimplentes %>% kable() %>% kable_styling(bootstrap_options = "striped",
                                                         full_width = FALSE,
                                                         font_size = 13)

##### Tratamento da base de dados ###############################################################
#* Esta fase do tratamento da base de dados é um grau mais sofisticada que a feita anteriormente,
#* pois aqui precisaremo lidar com NAs em um dataset com poucas observações, onde cada linha retirada
#* vai fazer falta ao modelo. Outras estratratégias de tratamento de NAs precisam ser implementada, que
#* não seja retirar essas linhas do banco de dados.
#* 
#* Outro fator de destaque é o uso de dados sintéticos na tentativa de contornar a situação de possuir
#* poucas observações para construir o modelo. Antes de construir o modelo, faremos uma verificação da 
#* importância das variáveis utilizando uma árvore de decisão.


##### 1. Trabalhar as NAs da base de dados | preparação da base de dados

inad_recipe <- recipe(Adimplemento ~ ., data = base_tratada_inadimplentes) %>% 
  step_impute_bag(all_numeric_predictors()) %>% 
  step_impute_mode(all_nominal_predictors())

base_completa <- bake(prep(inad_recipe),new_data = NULL)
base_completa %>% kable() %>% kable_styling(bootstrap_options = "striped",
                                            full_width = FALSE,
                                            font_size = 13)
skim(base_completa)

#* Os valores das variáveis categóricas estavam gravadas de forma diferente. Sendo necessário padronizar
#* a fim que não haja excesso de fatores.


numericos <- Filter(is.numeric, base_completa)
fatores <- Filter(is.factor,base_completa)

map(fatores,unique)

fatores <- map(fatores,str_to_lower)
fatores <- as_tibble(fatores)
fatores$Adimplemento %>% unique()

base_completa <- cbind(numericos, fatores) %>% as_tibble()
base_completa %>% kable() %>% kable_styling(bootstrap_options = "striped",
                                           full_width = FALSE,
                                           font_size = 13)

for (i in seq_along(base_completa)){
  if(base_completa[[i]] %>% is_character() == TRUE){
    base_completa [[i]] <- as.factor(base_completa[[i]])
    }
  }

skim(base_completa)

write.csv(base_completa,"db//DB_base_completa.csv")

table(base_completa$Adimplemento)

##### 2. Rodar árvore para verificar dados relevantes

#* Selection features: Verificar quais variáveis poderão impactar mais no modelo.
inad_tree2$variable.importance -> importancia
importancia <- as.data.frame(importancia)
importancia <- `colnames<-`(importancia,"Valores")
importancia <- rownames_to_column(importancia,var = "Variavel") %>% as_tibble()
importancia

#* Fazendo visualização da arvores

inad_tree <- rpart(Adimplemento ~ ., 
                   data = base_completa, 
                   method = "class", 
                   parms = list(split = "gini"), 
                   xval=5,
                   control = rpart.control(cp = 0.025,
                                           minsplit = 1,
                                           maxdepth = 10))

prp(inad_tree, type = 0, extra = "auto",nn = TRUE, branch = 1, under = TRUE, compress = TRUE, varlen = 0, yesno = 2)

##### 3. Utilizar dados sintéticos para aumentar a base de dados de teste
sint_base <- smote(Adimplemento ~ ., base_completa, perc.over = 6, perc.under = 1.175)
skim(sint_base)

table(sint_base$Adimplemento)

write.csv(sint_base,"db//DB_inad_sint.csv")

####### 0. IMPORTANDO PACOTES ##############################################################################

library(skimr)
library(tidyr)
library(ggplot2)
library(tidymodels)
library(patchwork) #dividir a janela para vários gráficos
library(NeuralNetTools)
library(vip)
library(plotly)

####### 1.IMPORTANDO BASE DE DADOS ########################################################################

base_original <- read.csv("campanha_venda_model.csv")
skim_without_charts(base_original)

####### 2.TRATANDO VARIÁVEIS ##############################################################################

base_mkt <- base_original[-1] # retirando a coluna de ID
base_mkt$Sexo %>% is.na() %>% sum()
base_mkt$Sexo %>% is.null() %>% sum()
base_mkt$Sexo %>% is_empty() %>% sum()
sum(base_mkt$Sexo == "")

#* A variável sexo tem 2.512 valores "". Para tratar será necessário realizar um laço para imputar valores 
#* NA, que mais a frente serão substituídos com a utilização do tidymodels.

Sex_input = c(rep(NULL,length(base_mkt$Sexo)))
for(i in seq_along(base_mkt$Sexo)){
  if(base_mkt$Sexo[[i]] == ""){
    Sex_input[i] <- NA
  }
  else{
    Sex_input[i] <- base_mkt$Sexo[[i]] }
}

Sex_input

### Trocando a variável sexo da base de dados por Sex_input criado anteriormente

base_mkt$Sexo_input <- Sex_input
base_mkt <- base_mkt[-6]
base_mkt

### Verificando frequência da distribuição

base_mkt$Engaj %>% unique() %>% sort(decreasing = FALSE)        # valores únicos da variável Engaj
base_mkt$Tipo_Cliente %>% unique() %>% sort(decreasing = FALSE) # valores únicos da variável Tipo_Cliente
base_mkt$Status %>% unique %>% sort(decreasing = FALSE)         # valores únicos da variável Engaj
table(base_mkt$Engaj, base_mkt$Tipo_Cliente) %>% as.data.frame()
table(base_mkt$Status,base_mkt$Engaj)

# Analisando variável Tipo_Cliente

base_mkt$Tipo_Cliente %>% unique() %>% sort(decreasing = FALSE) # Lista de valores únicos da variável

table(base_mkt$Tipo_Cliente, base_mkt$Status) # Tabela de frequência

#* Várias tabelas foram criadas experimentando a variável Tipo_Cliente com as demais, na busca por uma concentração de frequência.
#* Entretanto não foi possível chegar a conclusões úteis.Como não há muito por hora a ser dito sobre a variável, ela entrará no 
#* modelo assim mesmo.

####### 3.MODELAGEM #######################################################################################
#. 3.1. Split para base de dados

data_split <- base_mkt %>% initial_split(strata = "Campanha", prop = 3/4)
treino_split <- training(data_split)
teste_split <- testing(data_split)

#* 3.2. Processamento baseado em dados de treinamento. 
#* 
#* Iremos usar modelos onde várias transformações serão necessárias.
#* Uma vez que estamos lindado com um problema de classificação, usaremos os
#* seguintes modelos: decision tree,random forest,xgb,SVM, rede neural e
#* regressão logística.
#* 
#* 3.2. Receita para o grupo 1 com os modelos mlp, svm, logistic_reg

receita_g1 <- recipe(Campanha ~ ., data = treino_split) %>% 
  step_impute_bag(all_predictors()) %>%
  step_normalize(Total_gasto)%>% 
  step_scale(Engaj, Idade, Tipo_Cliente, Tempo_relacionamento) %>% 
  step_other(c(Regiao, Regiao_int, Sexo_input, Status), threshold = 0.05) %>% 
  step_bin2factor(Campanha) %>% 
  step_dummy(all_nominal_predictors(), -all_outcomes()) %>% 
  step_zv(all_numeric()) %>%  
  step_novel(all_nominal_predictors())

colnames(base_mkt)
#* Visualizando como ficou a base transformada
#* 
dados_g1 <- bake(prep(receita_g1), new_data = NULL)
# dados_g11 <- juice(prep(receita_g1)) # outra forma de extrair os dados de recipe
skim_without_charts(dados_g1)
#*
#* A base de dados aumentou o número de variáveis. De 10 passou para 22, muito disso
#* certamente devido a necessidade de criar variáveis dummy.
#*
#* 3.3. Receita paa o grupo 2 com os modelos rand_forest, boost_tree, decision_tree
receita_g2 <- recipe(Campanha ~ ., data = treino_split) %>%
  step_impute_bag(all_predictors()) %>% 
  step_other(c(Regiao, Regiao_int, Sexo_input, Status), threshold = 0.05) %>% 
  step_bin2factor(Campanha) %>% 
  step_dummy(all_nominal_predictors(), -all_outcomes()) %>% 
  step_zv(all_numeric()) %>%  
  step_novel(all_nominal_predictors())

#* Visualizando como ficou a base transformada
#* 
dados_g2 <- juice(prep(receita_g2))
skim_without_charts(dados_g2)

#* A base de dados aumentou o número de variáveis. De 10 passou para 22, muito disso
#* certamente devido a necessidade de criar variáveis dummy.

#* 3.4. Criação da estrutura dos modelos
#* 
#* 3.4.1. Grupo 1: modelos mlp, svm, logistic_reg
#* 
#* #* Modelo SVM
#* 
modsvm <- svm_rbf(cost = tune(),
                  rbf_sigma = tune(),
                  margin = tune()) %>% 
  set_engine("kernlab") %>% 
  set_mode("classification")

#* Modelo mlp
#* 
modmlp <- mlp(hidden_units = tune(),
              penalty = tune(),
              epochs = tune()) %>% 
  set_engine("nnet", num.threads = 4) %>% 
  set_mode("classification") %>% 
  translate()

#* Modelo regressão logística
#* 
modreglog <- logistic_reg(penalty = tune(),
                          mixture = tune()) %>% 
  set_engine("glmnet") %>% 
  set_mode("classification")

#* 3.4.2. Grupo 2: modelos rand_forest, boost_tree, decision_tree
#* 
#* Modelo decision_tree
#* 
modtree <- decision_tree(cost_complexity = tune(),
                         min_n = tune()) %>% 
  set_engine("rpart") %>% 
  set_mode("classification")

#* Modelo rand_forest
#* 
modforest <- rand_forest(mtry = tune(),
                         min_n = tune(),
                         trees = tune()) %>% 
  set_engine("ranger") %>% 
  set_mode("classification")

#* Modelo boost_tree
#* 
modxgbst <- boost_tree(tree_depth = tune(),
                       learn_rate = tune(),
                       loss_reduction = tune(),
                       min_n = tune(),
                       sample_size = tune(),
                       trees = tune()) %>% 
  set_engine("xgboost") %>% 
  set_mode("classification")

#* 3.5. Montagem do workflow
#* 
#* Primeiro vamos criar um workflow para cada grupo de modelos, depois vamos juntar todos em um só.
#* 
#* 3.5.1. Workflow do Grupo 1: modelos mlp, svm, logistic_reg e nearest_neighbor
#* 
wk_g1 <- workflow_set(preproc = list(grupo_1 = receita_g1),
                      models = list(rede_neural = modmlp,
                                    svm = modsvm,
                                    reg_log = modreglog))

#* 3.5.2. Workflow do Grupo 2: modelos rand_forest, boost_tree, decision_tree
#* 
wk_g2 <- workflow_set(preproc = list(grupo_2 = receita_g2),
                      models = list(rd_forest = modforest,
                                    xgb = modxgbst,
                                    d_tree = modtree))

#* 3.5.3. Workflow global, união dos dois workflows
#* 
wk_global <- bind_rows(wk_g1,wk_g2) %>% 
  mutate(wflow_id = gsub("(grupo_1_)|(grupo_2_)","",wflow_id))

#* 3.6. Selecionando parâmetros usando cross validation
#* 
cv_split <- vfold_cv(treino_split, v = 5, strata = "Campanha")
#* 
#* 
#* 3.7. Treinamento do modelo usando o grid
#* 
#* 3.7.1. Consolidação do grid
#* 
grid_ctrl <- control_grid(
  save_pred = TRUE,
  parallel_over = "everything",
  save_workflow = TRUE
)
#* 3.7.2. Treinamento do modelo
grid_result <- wk_global %>% 
  workflow_map(
    resamples = cv_split,
    grid = 5,
    control = grid_ctrl,
    metrics = metric_set(accuracy,roc_auc,specificity,recall))
#*
#* 3.8. Apresentando o melhor modelo
#* 
autoplot(grid_result)
#* 
autoplot(grid_result,
         rank_metric = "roc_auc",
         metric = "roc_auc",
         select_best = TRUE)
#* 
autoplot(grid_result,
         rank_metric = "accuracy",
         metric = "accuracy",
         select_best = TRUE)

#* 
autoplot(grid_result,
         rank_metric = "recall",
         metric = "recall",
         select_best = TRUE)

grid_result$result[[5]] %>% show_best(metric = "roc_auc", n = 6 )


#* 3.9.  Workflow do modelo

wrkFlow <- workflow() %>% 
  add_model(modxgbst) %>% 
  add_recipe(receita_g1)
#*
#* 4.0 Grid do modelo
#*
gridFinal <- expand.grid(
  trees = 1083 ,
  min_n = 39,
  tree_depth = 6,
  learn_rate = 0.00532,
  loss_reduction = 0.00197,
  sample_size = 0.677
)

#*
#* 4.1 Tunnando o modelo
mlpTunando <- tune_grid(
  wrkFlow,
  resamples = cv_split,
  grid = gridFinal,
  metric = metric_set(roc_auc, accuracy, f_meas),
  control = control_grid(verbose = TRUE, allow_par = FALSE)
)

#* 5.0 Seleção dos parâmetros
wrkFlowFinal <- wrkFlow %>% finalize_workflow(select_best(mlpTunando,"roc_auc"))
fitmlp <- last_fit(wrkFlowFinal, data_split, metrics = metric_set(accuracy,roc_auc, f_meas, specificity, recall))
collect_metrics(fitmlp)
#*
#* 6.0 Predições
testemlp <- collect_predictions(fitmlp)
#*
#* 6.01 Gráfico curva ROC
roc_mlp <- testemlp %>% 
  roc_curve(Campanha, .pred_yes) %>% 
  autoplot()
#*
#*
#* 6.02 Curva lift
lift_mlp  <-  testemlp %>% 
  lift_curve(Campanha, .pred_yes) %>% 
  autoplot()
#*
#*
#* 6.03 Curva ks
ksmlp <- testemlp %>% 
  ggplot(aes(x = .pred_yes, colour = Campanha))+
  labs(title = "Gráfico Ks")+
  stat_ecdf(show.legend = FALSE)
#*
#*
#* 6.04 Distribuição
distmlp <- testemlp %>% 
  ggplot(aes(x = .pred_yes, colour = Campanha))+
  geom_density()+
  labs(title = "Curva de distribuição")+
  theme(axis.title = element_blank())

#* 6.05 Apresentação dos gráficos
roc_mlp+lift_mlp+ksmlp+distmlp

#* 6.06 Explorando o lift
ggplotly(lift_mlp)

#* Pelo gráfico percebemos que ao nível de 25%, o lift é de 2,35.
#* De outro forma, usar o modelo é 135% melhor que selecionar aleatoriamente. 

#* 7.0 Importância das variáveis
#* 
modbstfinal <- fit(wrkFlowFinal,base_mkt)
vip(modbstfinal$fit$fit)+aes(fill = cumsum(Variable == "Tempo_relacionamento"))





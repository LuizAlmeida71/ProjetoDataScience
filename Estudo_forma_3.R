# TCC Data Science USP
# Previsão de inadimplência
# Boa Vista 26/03/2022
install.packages("xgboost")
library(tidymodels)
library(skimr)
library(ggplot2)
library(stringr)
library(cv)
library(knitr)
library(vip)
library(usethis)
library(kableExtra)
library(patchwork) #dividir a janela para vários gráficos
library(doParallel)
library(NeuralNetTools)
library(kernlab)
library(ranger)
library(xgboost)

####--- 0. Importação da base de dados

#base_inadimplencia <- read.csv("db//DB_inad_sint.csv")
base_inadimplencia <- read.csv("db//DB_base_completa.csv")

base_inadimplencia <- base_inadimplencia[-1]
#skim(base_inadimplencia)
#str(base_inadimplencia)
#colnames(base_inadimplencia)

###--- 1. Pipe line para selecionar um algoritmo

#. 1.1. Split para base de dados

data_split <- base_inadimplencia %>% initial_split(strata = "Adimplemento", prop = 3/4)
treino_split <- training(data_split)
teste_split <- testing(data_split)

#* 1.2. Processamento baseado em dados de treinamento. 
#* Iremos usar vários modelos onde várias transformações serão necessárias.
#* Uma vez que estamos lindado com um problema de classificação, usaremos os
#* seguintes modelos: decision tree,random forest,xgb,SVM, rede neural, naive bayes e
#* vizinhos mais próximos.
#* 
#* 1.2.1. Receita para os modelos mlp, svm, logistic_reg Chamaremos de grupo 1

receita_g1 <- recipe(Adimplemento ~ ., data = treino_split) %>% 
  step_other(c(Estado,Bairro, Curso_Nome), threshold = 0.1) %>% 
  step_normalize(all_numeric_predictors()) %>% 
  step_novel(all_nominal_predictors()) %>% 
  step_zv(all_predictors()) %>% 
  step_impute_mode(all_nominal_predictors()) %>% 
  step_dummy(all_nominal_predictors(), -all_outcomes())

?step_impute_mode
#* Visualizando como ficou a base transformada
#* 
#skim(bake(prep(receita_g1),new_data = NULL))
bake(prep(receita_g1),new_data = NULL) %>% kable() %>% kable_styling(bootstrap_options = "striped", font_size = 13, full_width = FALSE)
#*
#*A base de dados aumentou consideralvelmente o número de variáveis. De 14 passou para 84, muito disso
#*certamente devido a necessidade de criar variáveis dummy.
#*
#* 1.2.2. Receita para os modelos rand_forest, boost_tree, decision_tree. Chamaremos de 
#* grupo 2.

receita_g2 <- recipe(Adimplemento ~ ., data = treino_split) %>% 
  step_other(c(Estado,Bairro, Curso_Nome), threshold = 0.1) %>%
  step_zv(all_predictors()) %>% 
  step_impute_mode(all_nominal_predictors()) %>% 
  step_novel(all_nominal_predictors()) %>% 
  step_dummy(all_nominal_predictors(), -all_outcomes())

#* Visualizando como ficou a base transformada
#* 
skim(bake(prep(receita_g2), new_data = NULL))

#* 1.3. Criação da estrutura dos modelos
#* 
#* 1.3.1. Grupo 1: modelos mlp, svm, logistic_reg
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
  set_engine("glm") %>% 
  set_mode("classification")

#* 1.3.2. Grupo 2: modelos rand_forest, boost_tree, decision_tree
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

#* 1.4. Montagem do workflow
#* Primeiro vamos criar um workflow para cada grupo de modelos, depois vamos juntar todos em um só.
#* 
#* 1.4.1. Workflow do Grupo 1: modelos mlp, svm, logistic_reg e nearest_neighbor
#* 
wk_g1 <- workflow_set(preproc = list(grupo_1 = receita_g1),
                      models = list(rede_neural = modmlp,
                                    svm = modsvm,
                                    reg_log = modreglog))

#* 1.4.2. Workflow do Grupo 2: modelos rand_forest, boost_tree, decision_tree
#* 
wk_g2 <- workflow_set(preproc = list(grupo_2 = receita_g2),
                      models = list(rd_forest = modforest,
                                    xgb = modxgbst,
                                    d_tree = modtree))
#* 1.4.3. Workflow global, união dos dois workflows
#* 
wk_global <- bind_rows(wk_g1,wk_g2) %>% 
  mutate(wflow_id = gsub("(grupo_1_)|(grupo_2_)","",wflow_id))

#* 1.5.1. Selecionando parâmetros usando cross validation
#* 
cv_split <- vfold_cv(treino_split, v = 10, strata = "Adimplemento")
#* 
#* 
#* 1.5.2. Treinamento do modelo usando o grid
#* 
grid_ctrl <- control_grid(
  save_pred = TRUE,
  parallel_over = "everything",
  save_workflow = TRUE
)
#*
grid_result <- wk_global %>% 
  workflow_map(
    resamples = cv_split,
    grid = 15,
    control = grid_ctrl,
    metrics = metric_set(accuracy,roc_auc))
#* 
#*
#* 1.5.3. Apresentando o melhor modelo
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

grid_result$result[[1]] %>% show_best()

#* 2.0 Workflow do modelo

wrkFlow <- workflow() %>% 
  add_model(modmlp) %>% 
  add_recipe(receita_g1)
#*
#* 3.0 Grid do modelo
#*
gridFinal <- expand.grid(
  hidden_units = 2,
  penalty = 0.913,
  epochs = 401
)
#*
#* 4.0 Tunnando o modelo
mlpTunando <- tune_grid(
  wrkFlow,
  resamples = cv_split,
  grid = gridFinal,
  metric = metric_set(roc_auc, precision, accuracy, f_meas),
  control = control_grid(verbose = TRUE, allow_par = FALSE)
)
#*
#* 4.0.1 Saídas do modelo
autoplot(mlpTunando)
mlpTunando %>% show_best(metric = "roc_auc")
#*
#* 5.0 Seleção dos parâmetros
wrkFlowFinal <- wrkFlow %>% finalize_workflow(select_best(mlpTunando,"roc_auc"))
fitmlp <- last_fit(wrkFlowFinal, data_split, metrics = metric_set(accuracy,roc_auc, f_meas, specificity, precision, recall))
collect_metrics(fitmlp)
#*
#* 6.0 Predições
testemlp <- collect_predictions(fitmlp)
#*
#* 6.01 Gráfico curva ROC
roc_mlp <- testemlp %>% 
  roc_curve(Adimplemento, .pred_não) %>% 
  autoplot()
#*
#*
#* 6.02 Curva lift
lift_mlp  <-  testemlp %>% 
  lift_curve(Adimplemento, .pred_não) %>% 
  autoplot()
#*
#*
#* 6.03 Curva ks
ksmlp <- testemlp %>% 
  ggplot(aes(x = .pred_não, colour = Adimplemento))+
  labs(title = "Gráfico Ks")+
  stat_ecdf(show.legend = FALSE)
#*
#*
#* 6.04 Distribuição
distmlp <- testemlp %>% 
  ggplot(aes(x = .pred_não, fill = Adimplemento))+
  geom_density()+
  labs(title = "Curva de distribuição")+
  theme(axis.title = element_blank())

roc_mlp+lift_mlp+ksmlp+distmlp
#*
#*
#* 7.0 Modelo final
#* #* 7.1 Ajuste usando base inteira
modelomlpFinal <- fit(wrkFlowFinal,base_inadimplencia)
#*
#* 7.2 Variáveis importantes

vip(modelomlpFinal$fit$fit) + aes(fill = cumsum(Variable == "Estado_Civil_união.estável"))
#*
#* 7.3 Gravação do modelo
saveRDS(modelomlpFinal,"db//modeloFinalmlp.rds")
usethis::use_data(modelomlpFinal, overwrite = TRUE)
#* 
#*
#*
#*
dadosTratados <- read.csv("db//DB_base_completa_mlp.csv")
dadosTratados %>% kable(digits = 10) %>% kable_styling(bootstrap_options = "striped",
                                                       full_width = FALSE,
                                                       font_size = 13)
#*
#*
#* 8.0 Testando projeções
#*
#*
#*
aluno_novo <- data.frame(
  idade = 45,
  tempo_graduacao = 12,
  maturidade_formatura = 5,
  min_temp_processo = 1,
  max_temp_processo = 2,
  Carga_Horaria = 580,
  Valor_Curso = 57600.00,
  Qtd_Processos = 2,
  Orientação_Sexual = "feminino",
  Estado_Civil = "divorciado(a)",
  Estado = "ceará",
  Bairro = "pricumã",
  Curso_Nome = "especialização em prótese dentária")

predict(modelomlpFinal, new_data = aluno_novo, type = "prob")

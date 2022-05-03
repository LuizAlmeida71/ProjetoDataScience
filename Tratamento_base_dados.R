# TCC Data Science USP
# Previsão de inadimplência
# Boa Vista 09/02/2022

# Tratamento da base de dados

require(tidyverse)
require(readxl)
require(skimr)
require(knitr)
require(kableExtra)
require(lubridate)

# 0. Importando a base de dados

base_bruta <- read_excel("db//DB_Inadimplentes_Estudo_2.2.xlsx")
base_bruta$`Ano Processo Recente` <-  base_bruta$`Ano Processo Recente` %>% replace_na(0) %>% as.numeric()
base_bruta$`Ano Processo Antigo` <- base_bruta$`Ano Processo Antigo` %>% replace_na(0) %>% as.numeric()
base_bruta <- base_bruta[1:138,-c(1,2,16)]
#base_bruta$`Data de Nascimento` <- map(base_bruta$`Data de Nascimento`, gsub(x = base_bruta$`Data de Nascimento`, pattern = padrao, replacement = reposicao))

reposicao <- c(06071990,06031983,04011987,07091997,26071992)
base_bruta$`Data de Nascimento` <- replace(base_bruta$`Data de Nascimento`, c(12,13,14,15,16),reposicao)

base_bruta[90,3] <- c(17121986)

base_bruta %>% kable() %>% kable_styling(bootstrap_options = "striped",
                                         font_size = 13,
                                         full_width = TRUE)
base_bruta %>% skim()
base_bruta %>% colnames()

str(base_bruta)

# A base veio com várias linhas NA, depois do término da inserção dos dados e foi necessário retirar essas linhas, além de variáveis que idenficavam 
# o cliente.

# Tratando colunas com datas

base_tratada <- base_bruta %>% transmute(ano_graduacao = base_bruta$Data_Graduação %>% as.character() %>% str_extract("[:digit:]{4}$") %>% as.numeric(),
                                         ano_nascimento = base_bruta$`Data de Nascimento` %>% as.character() %>% str_extract("[:digit:]{4}$") %>% as.numeric(),
                                         ano_matricula = base_bruta$Data_Matrícula %>% as.character() %>% str_extract("[:digit:]{4}$") %>% as.numeric())

base_tratada <- base_tratada %>% transmute(idade = (now() %>% year()) - ano_nascimento,
                                          tempo_graduacao = (now() %>% year()) - ano_graduacao,
                                          maturidade_formatura = ano_matricula - ano_graduacao,
                                          min_temp_processo =  ifelse((now() %>% year() - base_bruta$`Ano Processo Recente`)<=90,now() %>% year() - base_bruta$`Ano Processo Recente`,0),
                                          max_temp_processo = ifelse((now() %>% year() - base_bruta$`Ano Processo Antigo`)<=60,now() %>% year() - base_bruta$`Ano Processo Recente`,0 ))

base_tratada %>% kable() %>% kable_styling(bootstrap_options = "striped",
                                           font_size = 13,
                                           full_width = FALSE)

str(base_tratada)
skim(base_tratada)
summary(base_tratada)

# localizando a linha onde tem valor estranho na variável idade. Alguém com 259 anos de idade certamente é erro de inserção de dados.
#linha = 0
# for(i in base_tratada$idade){
#   linha = linha + 1
#   if(i == 259){
#     break
#   }
# }

# Foi detectado problema na linha 90, que tinha registrado o ano de 1763, como ano de nascimento.
# print(linha)
# idade_estranha <- base_tratada[(base_tratada[,1] == 259),]
# idade_estranha <- idade_estranha[1,]
# idade_estranha

# base_tratada[90,2]
# base_bruta[90,3] <- c(17121986)
# Nesse ponto foi verificado que na base bruta havia um erro, deixei o registro aqui, entretanto a correção foi levada para a parte onde tratamos base bruta.

## Verificar a variável tempo de graduação

ggplot(data = base_tratada, mapping = aes(x = tempo_graduacao))+
  geom_histogram()

base_tratada %>% filter(tempo_graduacao > 10)

# Os valores que a priori pareciam muito distintos, estão corretos, pois pertecem a alunos com as maiores idades.

## Verificar a variável maturidade formatura

ggplot(data = base_tratada, mapping = aes(x = maturidade_formatura))+
  geom_histogram()

base_tratada %>% filter(maturidade_formatura > 10)
# Analisada a variável, verificou-se que os dados apresentam-se consistentes.

## Compor base fina de dados

base_bruta %>% str()

base_parcial <- base_bruta %>% select("Orientação_Sexual","Estado Civil","Estado","Bairro","Curso_Nome","Carga_Horaria","Valor_Curso","Quantidade Processos","Adimplemento")
base_tratada <- cbind(base_tratada,base_parcial)

## Gravar a base dados tratada
write_excel_csv2(base_tratada,"db//DB_tratada_inadimplentes_estudo.xlsx")

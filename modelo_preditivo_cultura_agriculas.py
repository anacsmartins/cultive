# Instalação das bibliotecas necessárias
# !pip install pandas numpy scikit-learn imbalanced-learn chardet gdown joblib

# Importações
import pandas as pd
import numpy as np
import os
import gdown # Usado para download de arquivos do Google Drive
import chardet # Usado para detecção de codificação de arquivos
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from imblearn.over_sampling import SMOTE # Usado para balanceamento de classes
import joblib # Usado para salvar e carregar modelos
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import unicodedata # Usado para normalizar strings (remover acentos, etc.)
import re # Importar regex para a função normalizar_string

warnings.filterwarnings('ignore') # Ignora avisos, como os de versões futuras de bibliotecas

# --- Funções de Suporte ---
def detectar_codificacao_e_separador(caminho):
    """
    Detecta a codificação e o separador de um arquivo CSV.
    Isso ajuda a carregar arquivos CSV de diversas fontes sem erros.
    """
    with open(caminho, 'rb') as f:
        # Lê uma parte do arquivo para detectar a codificação
        resultado = chardet.detect(f.read(50000))
    encoding = resultado['encoding']
    with open(caminho, 'r', encoding=encoding, errors='ignore') as f:
        primeira_linha = f.readline()
    separadores = [';', ',', '\t']
    # Tenta determinar o separador mais provável contando ocorrências na primeira linha
    melhor_sep = max(separadores, key=lambda sep: len(primeira_linha.split(sep)))
    return encoding, melhor_sep

def carregar_bdsolos_csv(pasta):
    """
    Carrega e concatena arquivos CSV de uma pasta específica (dados do solo),
    detectando automaticamente a codificação e o separador de cada arquivo.
    """
    arquivos = [f for f in os.listdir(pasta) if f.endswith(".csv")]
    frames = []
    for arquivo in arquivos:
        caminho = os.path.join(pasta, arquivo)
        encoding, separador = detectar_codificacao_e_separador(caminho)
        df = pd.read_csv(caminho, encoding=encoding, sep=separador, engine='python', on_bad_lines='skip')
        df.columns = df.columns.str.strip().str.lower() # Normaliza nomes das colunas
        if 'município' in df.columns: # Correção de grafia comum
            df.rename(columns={'município': 'municipio'}, inplace=True)
        # Padroniza 'municipio' e 'uf' para merge
        for col in ["municipio", "uf"]:
            if col in df.columns:
                # --- MUDANÇA CRÍTICA AQUI: Aplicar normalizar_string para consistência ---
                df[col] = df[col].astype(str).apply(normalizar_string)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)

def normalizar_string(text):
    if pd.isna(text) or text is None:
        return 'nan' # Retorna 'nan' string para nulos, que será mapeada para 'Outras'

    text = str(text).strip().lower()

    # 1. Pré-limpeza de caracteres pontuais (parênteses, vírgulas, pontos)
    # Ajustado para remover apóstrofos e traços que podem ser indesejados
    text = text.replace('(', '').replace(')', '').replace(',', '').replace('.', '')
    text = text.replace("'", "").replace("´", "") # Remover apóstrofos

    # 2. SUBSTITUIÇÕES CRÍTICAS DE PADRÕES LATIN1/ERROS COMUNS (PRIORIDADE ALTA)
    # Adicionado tratamento para 'Ã­', 'Ã§', 'Ã£', 'Ãº' e outras variações comuns
    text = text.replace('indaostria', 'industria')
    text = text.replace('produao', 'producao')
    text = text.replace('implantao', 'implantacao')
    text = text.replace('granafero', 'granifero') # Correção: granafero para granifero
    text = text.replace('macaoba', 'macauba')
    text = text.replace('paassego', 'pessego')
    text = text.replace('passego', 'pessego')
    text = text.replace('cevendish', 'cavendish')
    text = text.replace('cafa', 'cafe') # 'cafa' para 'cafe'
    text = text.replace('cana_de_aocar', 'cana_de_acucar')
    text = text.replace('aaaocar', 'acucar')
    text = text.replace('banana_ma', 'banana_maca')
    text = text.replace('ã£o', 'acao')
    text = text.replace('ã¡', 'a') # Para 'á'
    text = text.replace('ãº', 'u') # Para 'ú'
    text = text.replace('Ãº', 'u') # Para 'Ú'
    text = text.replace('Ã­', 'i') # Para 'í'
    text = text.replace('Ã§', 'c') # Para 'ç'
    text = text.replace('Ã£', 'a') # Para 'ã'
    text = text.replace('Ã©', 'e') # Para 'é'
    text = text.replace('Ã³', 'o') # Para 'ó'
    text = text.replace('Ã¡', 'a') # Para 'á'
    text = text.replace('Ã­pio', 'ipio') # Município -> Municipio
    text = text.replace('aucar', 'acucar') # Corrigir 'cana_de_aucar' para 'cana_de_acucar'
    text = text.replace('gracaos', 'graos') # Padronizar 'gracaos' para 'graos' (afeta cevada)

    # 3. Normalização Unicode (captura a maioria dos acentos e caracteres especiais que sobraram)
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8')

    # 4. Pós-Normalização e Limpeza Final:
    text = re.sub(r'a{2,}', 'a', text) # Remove múltiplos 'a's (ex: "laranjaaa" -> "laranja")
    text = text.replace(' ', '_')
    text = text.replace('-', '_')

    # Remover múltiplos underscores para um único underscore
    text = text.replace('___', '_').replace('__', '_')

    # Remover underscores do início ou fim
    text = text.strip('_')

    # Correções específicas de nomes após normalização geral
    text = text.replace('feijacao', 'feijao') # feijao (base)
    text = text.replace('algodacao', 'algodao') # algodao (base)
    text = text.replace('limacao', 'limao') # limao (base)
    text = text.replace('ma_implantacao', 'maca_implantacao') # Maçã
    text = text.replace('ma_producao', 'maca_producao') # Maçã
    text = text.replace('a_producao', 'acai_producao') # Assumindo 'a' seja açaí, se não, reavaliar
    text = text.replace('a_implatacao', 'acai_implantacao') # Assumindo 'a' seja açaí
    text = text.replace('mamacao', 'mamao') # mamão

    return text
# --- 1. Coleta e Preparação de Dados ---

# Download dos dados do solo
print("Baixando dados do solo...")
links_arquivos = [
    "https://drive.google.com/uc?id=1FVLU3XniibzCVhd2tl1-SNJTkb7ayF6C",
    "https://drive.google.com/uc?id=1F50lRM7q_plY4ud7Xh2U2mMhyUWtvO3Y",
    "https://drive.google.com/uc?id=1JCSAYLWrt4kyI5kcMyYtnAgfsZkJjSc5",
    "https://drive.google.com/uc?id=1TvQQIOzMu0w6H9p4OpkHCkGzqWBwfe5N"
]
nomes_arquivos = ["solo_1.csv", "solo_2.csv", "solo_3.csv", "solo_4.csv"]
os.makedirs("dados_bdsolos", exist_ok=True) # Cria a pasta se não existir

# Descomente as linhas abaixo para habilitar o download automático
for link, nome in zip(links_arquivos, nomes_arquivos):
    gdown.download(link, os.path.join("dados_bdsolos", nome), quiet=True, fuzzy=True)
print("Download dos dados do solo concluído. ✅")

# Carregar dados de solo
df_solo_real = carregar_bdsolos_csv("dados_bdsolos")
# Remove linhas com município nulo e duplicações baseadas no município (mantém a primeira ocorrência)
df_solo_real = df_solo_real.dropna(subset=["municipio"]).drop_duplicates(subset=["municipio", "uf"]) # Adicionado UF para garantir unicidade mais robusta


# Carregar dados ZARC
print("Baixando dados ZARC...")
zarc_url = "https://drive.google.com/uc?id=1ZVdeBmpBWDEINS8QdQ3KiPPexirAMx0-"
zarc_path = "zarc.csv"
# Descomente a linha abaixo para habilitar o download automático
gdown.download(zarc_url, zarc_path, quiet=True)

# --- BLOCO DE VERIFICAÇÃO DE ARQUIVO ZARC ---
print(f"\nVerificando o arquivo: {zarc_path}")
if not os.path.exists(zarc_path):
    print(f"ERRO CRÍTICO: O arquivo '{zarc_path}' NÃO foi encontrado no diretório atual.")
    print("Por favor, garanta que você fez o upload manual para o local correto ou descomente a linha de download.")
    raise FileNotFoundError(f"Arquivo {zarc_path} não encontrado.")
else:
    tamanho_arquivo = os.path.getsize(zarc_path)
    print(f"Tamanho do arquivo '{zarc_path}': {tamanho_arquivo} bytes.")
    if tamanho_arquivo == 0:
        print(f"ALERTA: O arquivo '{zarc_path}' está VAZIO! Por favor, baixe-o novamente.")
        raise ValueError(f"Arquivo {zarc_path} está vazio.")
    else:
        print(f"Arquivo '{zarc_path}' encontrado e não está vazio.")

# --- BLOCO DE IMPRESSÃO DO CONTEÚDO BRUTO (para depuração) ---
print(f"\n--- Conteúdo BRUTO das primeiras 5 linhas de '{zarc_path}' ---")
try:
    with open(zarc_path, 'r', encoding='latin1', errors='ignore') as f:
        for i, line in enumerate(f):
            print(f"Linha {i+1}: {line.strip()}")
            if i >= 4:
                break
except UnicodeDecodeError:
    print("Erro de decodificação com 'latin1'. Tentando 'utf-8'...")
    with open(zarc_path, 'r', encoding='utf-8', errors='ignore') as f:
        for i, line in enumerate(f):
            print(f"Linha {i+1}: {line.strip()}")
            if i >= 4:
                break
except Exception as e:
    print(f"Não foi possível ler as primeiras linhas do arquivo: {e}")
print("------------------------------------------------------------------")

# --- Carregar df_zarc ---
try:
    # MUDANÇA CRÍTICA AQUI: Adicionar decimal=',' para produtividade
    df_zarc = pd.read_csv(zarc_path, encoding='latin1', sep=',',
                          na_values=['-', 'Nao se aplica', 'NÃ£o se aplica'],
                          decimal=',', # Adicionado para tratar vírgulas como decimais
                          engine='python', on_bad_lines='skip')

    # Renomear as colunas para minúsculas e sem acentos, se o cabeçalho já existe
    df_zarc.columns = [
        col.strip().lower()
        .replace('município', 'municipio')
        .replace('municã­pio', 'municipio') # This was the specific issue identified earlier
        .replace('munica__pio', 'municipio') # Fallback for double underscore issues
        .replace('ã§', 'c')
        .replace('ã£', 'a')
        .replace('Ã§', 'c')
        .replace('Ã£', 'a')
        .replace('Ã©', 'e')
        .replace('ã¡', 'a') # Added to handle 'Produtividade' -> 'produtividade' and 'Cultura' -> 'cultura'
        .replace('ãº', 'u') # For 'ú' characters
        .replace('Ãº', 'u') # For 'Ú' characters
        for col in df_zarc.columns
    ]

    # Aplica normalização nas colunas de texto de df_zarc
    # Garantir que 'grupo', 'solo', 'clima' sejam tratadas como strings antes da normalização
    for col in ['cultura', 'grupo', 'solo', 'clima', 'municipio', 'uf']:
        if col in df_zarc.columns:
            df_zarc[col] = df_zarc[col].astype(str).apply(normalizar_string)

    print("\nTentativa final (latin1, ',', na_values='-', manual_columns) - Colunas após leitura:")
    print(df_zarc.columns)

    # --- NOVO DEBUG: ANÁLISE DE 'PRODUTIVIDADE' EM DF_ZARC ORIGINAL ---
    print("\n--- DEBUG: Análise de 'produtividade' em df_zarc ---")
    if 'produtividade' in df_zarc.columns:
        print(f"Tipo de dados da coluna 'produtividade' no df_zarc: {df_zarc['produtividade'].dtype}")
        print("Contagem de valores na coluna 'produtividade' (Top 5):")
        print(df_zarc['produtividade'].value_counts(dropna=False).head(5))
        print(f"Total de NaNs em 'produtividade' no df_zarc original: {df_zarc['produtividade'].isnull().sum()}")
    else:
        print("'produtividade' não encontrada no df_zarc após o carregamento.")
    print("--------------------------------------------------")

except Exception as e:
    print(f"Erro CRÍTICO ao carregar zarc.csv com Pandas: {e}")
    print("Isso indica um problema grave com a formatação ou conteúdo do arquivo.")
    print("Por favor, verifique o arquivo zarc.csv manualmente novamente, linha por linha.")
    raise

# --- DEBUG DO MERGE ---
print("\n--- DEBUG DO MERGE ---")
print("Colunas de df_solo_real:")
print(df_solo_real.columns)
print("Amostra de df_solo_real['municipio']:")
print(df_solo_real['municipio'].head())
print("Amostra de df_solo_real['uf']:")
print(df_solo_real['uf'].head())
print("-" * 20)
print("Colunas de df_zarc:")
print(df_zarc.columns)
print("Amostra de df_zarc['municipio']:")
print(df_zarc['municipio'].head())
print("Amostra de df_zarc['uf']:")
print(df_zarc['uf'].head())
print("Amostra de df_zarc['cultura']:")
print(df_zarc['cultura'].head())
print("-" * 20)

# Realizar o merge dos dataframes
# Garantir que todas as colunas do ZARC que não são chaves de merge sejam diferenciadas
df_merged = pd.merge(df_solo_real, df_zarc, on=["municipio", "uf"], how="left", suffixes=('_solo', '_zarc'))
print(f"Número de linhas após o merge: {len(df_merged)}")

# --- Ajuste final nas colunas de merge e 'cultura' ---
# Prioriza 'cultura_zarc' para a coluna 'cultura' final
# Isso é crucial, pois 'cultura_zarc' é a que está ligada à produtividade e ZARC.
if 'cultura_zarc' in df_merged.columns:
    df_merged['cultura'] = df_merged['cultura_zarc']
    # MUDANÇA AQUI: Renomear as colunas do ZARC com sufixo para seus nomes originais
    # Isso é importante para que sejam reconhecidas como categóricas mais tarde
    df_merged.rename(columns={
        'grupo_zarc': 'grupo',
        'solo_zarc': 'solo',
        'clima_zarc': 'clima',
        'produtividade_zarc': 'produtividade' # Renomear de volta a produtividade mergeada
    }, inplace=True)

    # Apenas remove as colunas '_solo' e a 'cultura_zarc' (agora redundante)
    cols_to_drop_after_merge = [col for col in df_merged.columns if '_solo' in col]
    if 'cultura_zarc' in df_merged.columns: # Remove a original agora que já foi copiada
        cols_to_drop_after_merge.append('cultura_zarc')
    df_merged.drop(columns=cols_to_drop_after_merge, inplace=True, errors='ignore')
else:
    # Se 'cultura_zarc' não foi criada (merge falhou completamente para cultura do ZARC)
    if 'cultura' not in df_merged.columns:
        raise KeyError("A coluna 'cultura' (do ZARC ou fallback) não foi encontrada após o merge. Verifique o processo de merge e renomeação.")


print("Colunas de df_merged após ajustes para 'cultura':")
print(df_merged.columns)
print("----------------------")

# --- DEBUG: Amostra da coluna 'cultura' ANTES da padronização (após ajustes de merge) ---
print("\nDEBUG: Amostra da coluna 'cultura' ANTES da padronização final:")
print(df_merged['cultura'].head())
print("DEBUG: Tipo de dados da coluna 'cultura' ANTES da padronização final:")
print(df_merged['cultura'].dtype)
print("DEBUG: Contagem de valores nulos na coluna 'cultura' ANTES da padronização final:")
print(df_merged['cultura'].isnull().sum())

# Padroniza a coluna 'cultura' em df_merged (garantindo consistência após o merge)
df_merged['cultura'] = df_merged['cultura'].astype(str).str.strip().str.lower()

print("\nDEBUG: Amostra da coluna 'cultura' DEPOIS da padronização final:")
print(df_merged['cultura'].head())

# --- DEBUG: Valores únicos da coluna 'cultura' após normalização, antes do agrupamento final (Top 50) ---
print("\n--- DEBUG: Valores únicos da coluna 'cultura' após normalização, antes do agrupamento final (Top 50) ---")
# Agora df_merged já está definido e a coluna 'cultura' já foi normalizada
culturas_normalizadas_unicas = df_merged['cultura'].value_counts().head(50)
print(culturas_normalizadas_unicas)
print("----------------------------------------------------------------------------------------------------\n")


# --- 2. Engenharia de Features e Pré-processamento ---

# Colunas de interesse selecionadas - Expandidas para incluir a produtividade e mais microelementos
colunas_interesse_selecionadas = [
    "ph - h2o", "ph - kcl",
    "composição granulométrica da terra fina - areia total (g/kg)",
    "composição granulométrica da terra fina - silte (g/kg)",
    "composição granulométrica da terra fina - argila (g/kg)",
    "carbono orgânico", "nitrogênio total", "relação c/n (%)", "fósforo assimilável (mg/kg)",
    "complexo sortivo - potássio",
    "complexo sortivo - alumínio trocável (al3+)",
    "complexo sortivo - valor t (s+h++al3+)", "complexo sortivo - valor s (ca2++mg2++k++na+)",
    "densidade - solo (aparente)", "densidade - partículas (real)",
    "condutividade hidráulica (mm/h)",
    "microelementos - ferro", "microelementos - boro", "microelementos - zinco", "microelementos - cobre",
    "microelementos - manganês",
    "produtividade" # Incluída novamente
]

# Agrupamento das culturas em categorias mais amplas (DICIONÁRIO FINAL REFINADO E VERIFICADO)
agrupamento_nome_cultura_bruto = {
    # Cereais (incluindo as correções de 'graos')
    "milho": "Cereais", "milho_1a_safra": "Cereais", "milho_1a_safra_consorciado_com_braquiaria": "Cereais",
    "milho_2a_safra": "Cereais", "milho_2a_safra_consorciado_com_braquiaria": "Cereais",
    "arroz": "Cereais", "arroz_sequeiro": "Cereais", "trigo": "Cereais", "trigo_irrigado": "Cereais",
    "trigo_sequeiro": "Cereais", "cevada": "Cereais", "cevada_graos_irrigada": "Cereais", # Agora com 'graos'
    "cevada_cervejeira": "Cereais", "cevada_graos": "Cereais", "cevada_graos_sequeiro": "Cereais", # Agora com 'graos'
    "centeio": "Cereais", "aveia": "Cereais", "aveia_irrigada": "Cereais", "aveia_sequeiro": "Cereais",
    "sorgo_granifero": "Cereais", "sorgo_granifero_2a_safra": "Cereais",
    "sorgo_forrageiro": "Cereais", "sorgo_forrageiro_2a_safra": "Cereais",
    "triticale": "Cereais", "milheto": "Cereais",

    # Oleaginosas
    "soja": "Oleaginosas", "girassol": "Oleaginosas", "mamona": "Oleaginosas",
    "mamona_semi_arido_sequeiro": "Oleaginosas",
    "gergelim": "Oleaginosas", "canola": "Oleaginosas",

    # Leguminosas (corrigido para 'feijao' e adicionado 'grao_de_bico')
    "feijao": "Leguminosas", "feijao_2a_safra": "Leguminosas", "feijao_1a_safra": "Leguminosas",
    "feijao_caupi": "Leguminosas", "amendoim": "Leguminosas", "ervilha": "Leguminosas",
    "grao_de_bico": "Leguminosas",

    # Tubérculos
    "batata": "Tubérculos", "batata_industria": "Tubérculos", "batata_mesa": "Tubérculos",
    "mandioca": "Tubérculos", "mandioca_aipim_macaxeira": "Tubérculos", "batata_doce": "Tubérculos",

    # Frutíferas (incluindo as novas entradas da lista "Outras")
    "abacaxi": "Frutíferas", "maca": "Frutíferas", "maca_implantacao": "Frutíferas", "maca_producao": "Frutíferas", # Maçã adicionada
    "laranja": "Frutíferas", "laranja_implantacao": "Frutíferas", "laranja_producao": "Frutíferas",
    "limao": "Frutíferas", "limao_implantacao": "Frutíferas", "limao_producao": "Frutíferas", # limao adicionado
    "tangerina": "Frutíferas", "tangerina_implantacao": "Frutíferas", "tangerina_producao": "Frutíferas",
    "banana": "Frutíferas", "banana_cavendish_implantacao": "Frutíferas", "banana_maca_producao": "Frutíferas",
    "banana_cavendish_producao": "Frutíferas", "banana_prata_producao": "Frutíferas",
    "banana_prata_implantacao": "Frutíferas", "banana_maca_implantacao": "Frutíferas", # Banana ma/maca
    "uva": "Frutíferas", "uva_mesa": "Frutíferas", "uva_industrial": "Frutíferas",
    "melancia": "Frutíferas", "maracuja": "Frutíferas", "pessego": "Frutíferas",
    "pessego_implantacao_industria": "Frutíferas", "pessego_producao_industria": "Frutíferas",
    "pessego_implantacao_mesa": "Frutíferas", "pessego_producao_mesa": "Frutíferas",
    "nectarina": "Frutíferas", "nectarina_implantacao_industria": "Frutíferas",
    "nectarina_implantacao_mesa": "Frutíferas", "nectarina_producao_industria": "Frutíferas",
    "nectarina_producao_mesa": "Frutíferas",
    "macauba_acrocomia_aculeata_producao": "Frutíferas", "macauba_acrocomia_aculeata_implantacao": "Frutíferas",
    "macauba_acrocomia_totai_producao": "Frutíferas", "macauba_acrocomia_totai_implantacao": "Frutíferas",
    "macauba_acrocomia_intumescens_implantacao": "Frutíferas", "macauba_acrocomia_intumescens_producao": "Frutíferas",
    "caju": "Frutíferas", "caju_implantacao": "Frutíferas", "caju_producao": "Frutíferas",
    "lima": "Frutíferas", "lima_producao": "Frutíferas", "lima_implantacao": "Frutíferas", # Nova: lima
    "toranja": "Frutíferas", "toranja_producao": "Frutíferas", "toranja_implantacao": "Frutíferas", # Nova: toranja
    "pomelo": "Frutíferas", "pomelo_producao": "Frutíferas", "pomelo_implantacao": "Frutíferas", # Nova: pomelo
    "cacau": "Frutíferas", "cacau_producao": "Frutíferas", "cacau_implantacao": "Frutíferas", # Nova: cacau
    "mamao": "Frutíferas", # Nova: mamão
    "acai": "Frutíferas", "acai_producao": "Frutíferas", "acai_implantacao": "Frutíferas", # Nova: açaí (assumindo 'a' seja acai)

    # Fibras (corrigido para 'algodao')
    "algodao": "Fibras", "algodao_herbaceo": "Fibras",

    # Energéticas (corrigido para 'acucar')
    "cana_de_acucar": "Energéticas", "cana_de_acucar_acucar_e_alcool": "Energéticas",
    "cana_de_acucar_outros_fins": "Energéticas",

    # Estimulantes (corrigido para 'cafe')
    "cafe": "Estimulantes", "cafe_canafora_implantacao": "Estimulantes", "cafe_canafora_producao": "Estimulantes",
    "cafe_arabica_implantacao": "Estimulantes", "cafe_arabica_producao": "Estimulantes",

    # Hortaliças
    "tomate": "Hortaliças", "alface": "Hortaliças", "cenoura": "Hortaliças",
    "cebola": "Hortaliças", "cebola_plantio_com_bulbinho": "Hortaliças",

    # Forrageiras
    "forrageira_pecuaria": "Forrageiras", "pastagem": "Forrageiras", "palma_forrageira": "Forrageiras",

    # Outras (para NaN ou qualquer coisa que realmente não se encaixe)
    "nan": "Outras", # Explicitamente mapeia a string 'nan' para 'Outras'
}


# Padroniza a coluna 'cultura' em df_merged (garantindo consistência após o merge)
df_merged['cultura'] = df_merged['cultura'].astype(str).str.strip().str.lower()

print("\nDEBUG: Amostra da coluna 'cultura' DEPOIS da padronização final:")
print(df_merged['cultura'].head())

# --- DEBUG: Valores únicos da coluna 'cultura' após normalização, antes do agrupamento final (Top 50) ---
print("\n--- DEBUG: Valores únicos da coluna 'cultura' após normalização, antes do agrupamento final (Top 50) ---")
# Agora df_merged já está definido e a coluna 'cultura' já foi normalizada
culturas_normalizadas_unicas = df_merged['cultura'].value_counts().head(50)
print(culturas_normalizadas_unicas)
print("----------------------------------------------------------------------------------------------------\n")


# --- Análise da Categoria 'Outras' ---
print("\n--- Análise da Categoria 'Outras' ---")
# Certifique-se de que agrupamento_nome_cultura_bruto está definido antes de usar esta seção
# (Já está definido na parte superior do seu script, então está ok)

# Reaplicar a categorização para ter certeza, caso o script tenha sido modificado
df_merged["categoria_cultura"] = df_merged["cultura"].replace(agrupamento_nome_cultura_bruto)
df_merged["categoria_cultura"] = df_merged["categoria_cultura"].apply(
    lambda x: x if x in agrupamento_nome_cultura_bruto.values() else "Outras"
)

culturas_na_categoria_outras = df_merged[df_merged['categoria_cultura'] == 'Outras']['cultura']
contagem_culturas_outras = culturas_na_categoria_outras.value_counts()

if not contagem_culturas_outras.empty:
    print("Culturas originais que foram agrupadas na categoria 'Outras' (Todas):")
    # Imprime todas as culturas na categoria 'Outras'
    # Limitando a um número razoável, digamos, as 50 principais, ou todas se forem poucas
    if len(contagem_culturas_outras) > 50:
        print(contagem_culturas_outras.head(50))
    else:
        print(contagem_culturas_outras)

    print(f"\nTotal de culturas distintas em 'Outras': {len(contagem_culturas_outras)}")
else:
    print("Nenhuma cultura foi agrupada na categoria 'Outras'.")
print("-------------------------------------")
# --- FIM DA INSERÇÃO ---


# Definir o alvo e criar df_modelo
alvo = "categoria_cultura"

# ... (restante do código) ...

# Padronizar os nomes das colunas em df_merged para minúsculas e sem espaços extras
df_merged.columns = df_merged.columns.str.strip().str.lower()

# Filtrar colunas de interesse que realmente existem no df_merged
colunas_presentes_numericas = [col for col in colunas_interesse_selecionadas if col in df_merged.columns]
colunas_ausentes_numericas = [col for col in colunas_interesse_selecionadas if col not in df_merged.columns]

if colunas_ausentes_numericas:
    print(f"Atenção: As seguintes colunas de interesse NUMÉRICAS NÃO foram encontradas no dataframe mesclado e serão ignoradas: {colunas_ausentes_numericas}")
    if len(colunas_presentes_numericas) < 5:
        print("ALERTA CRÍTICO: Poucas colunas numéricas de interesse foram encontradas. O modelo pode ter baixo desempenho.")

categorical_features_to_keep = ['uso atual', 'grupo', 'solo', 'clima']
# Filtrar para ter certeza que essas colunas existem no df_merged
categorical_features_to_keep = [col for col in categorical_features_to_keep if col in df_merged.columns]

# Criar df_modelo com as colunas numéricas presentes + categóricas + o alvo
df_modelo = df_merged[colunas_presentes_numericas + categorical_features_to_keep + [alvo]].copy()

# --- NOVO PASSO CRÍTICO: FORÇAR COLUNAS NUMÉRICAS PARA O TIPO NUMÉRICO ---
# Iterar apenas sobre as colunas que ESPERAMOS que sejam numéricas.
# 'produtividade' já deve ter sido lida corretamente devido ao `decimal=','`
for col in colunas_presentes_numericas:
    if col in df_modelo.columns:
        # Se 'produtividade' já foi lida como float, essa linha não fará nada.
        # Caso contrário, tentará novamente, mas agora o decimal já foi tratado.
        df_modelo[col] = pd.to_numeric(df_modelo[col], errors='coerce')


# --- REVISÃO DA ESTRATÉGIA DE TRATAMENTO DE NaNs EM COLUNAS ---
print("\n--- Estratégia de Tratamento de NaNs em Colunas ---")
# Identifica colunas numéricas em df_modelo APÓS a coerção de tipo
numeric_cols_df_modelo = df_modelo.select_dtypes(include=np.number).columns.tolist()

# Dropar colunas numéricas com grande percentual de NaNs ANTES da imputação, se o percentual for muito alto
colunas_para_dropar_por_nan = []
for col in numeric_cols_df_modelo:
    # APLICANDO O NOVO LIMIAR MAIS ALTO AQUI
    if df_modelo[col].isnull().sum() / len(df_modelo) > 0.95: # MUDANÇA AQUI: de 0.70 para 0.95
        colunas_para_dropar_por_nan.append(col)

if colunas_para_dropar_por_nan:
    print(f"Dropando as seguintes colunas com mais de 95% de NaNs: {colunas_para_dropar_por_nan}")
    df_modelo.drop(columns=colunas_para_dropar_por_nan, inplace=True)
    # Atualiza a lista de colunas numéricas após dropar
    numeric_cols_df_modelo = [col for col in numeric_cols_df_modelo if col not in colunas_para_dropar_por_nan]


# Remover linhas onde o alvo é NaN (essencial para o treinamento)
df_modelo = df_modelo.dropna(subset=[alvo])

print(f"Número de linhas após limpeza inicial e drop de colunas com muitos NaNs: {len(df_modelo)}")
print(f"Número de colunas após limpeza inicial e drop de colunas com muitos NaNs: {df_modelo.shape[1]}")
print("-" * 20)


# --- Análise Exploratória de Dados (EDA) ---
print("\n--- Análise da Relação entre Features e Alvo ---")

print("\nColunas numéricas em df_modelo ANTES da EDA:")
current_numeric_cols = df_modelo.select_dtypes(include=np.number).columns.tolist()
print(current_numeric_cols)
print("-" * 20)

# Selecione algumas das features numéricas mais relevantes para visualização.
features_para_eda = [
    "ph - h2o",
    "carbono orgânico",
    "fósforo assimilável (mg/kg)",
    "produtividade", # Agora deve ser plotado se for mantido
    "composição granulométrica da terra fina - argila (g/kg)"
]

for feature in features_para_eda:
    if feature in current_numeric_cols: # Verifica se a feature ainda existe após as limpezas
        plt.figure(figsize=(10, 6))
        sns.boxplot(x=alvo, y=feature, data=df_modelo)
        plt.title(f'Distribuição de {feature} por Categoria de Cultura')
        plt.xlabel('Categoria de Cultura')
        plt.ylabel(feature.title())
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        # plt.show() # Descomente para exibir os plots durante a execução
    else:
        print(f"AVISO: A feature '{feature}' não foi encontrada em df_modelo para plotagem após a limpeza. Ela pode ter sido removida devido a muitos NaNs.")

print("---------------------------------------------")

# --- 3. Separação de Dados ---
# Features (X) e Alvo (y)
X = df_modelo.drop(columns=[alvo])
y = df_modelo[alvo]

# --- VERIFICAÇÃO CRÍTICA: X não pode estar vazio ---
if X.empty or X.shape[1] == 0:
    raise ValueError("ERRO CRÍTICO: DataFrame de features (X) está vazio ou não contém colunas após o pré-processamento. Revise as etapas de limpeza e seleção de colunas.")

# Codificação de variáveis categóricas (One-Hot Encoding para X)
# Garantir que as colunas 'grupo', 'solo', 'clima' sejam tratadas como categóricas
# As colunas categóricas devem ter o tipo 'object' para serem detectadas pelo select_dtypes
# Se elas foram renomeadas, o nome original 'grupo', 'solo', 'clima' deve ser usado.
colunas_categoricas = X.select_dtypes(include='object').columns
if not colunas_categoricas.empty:
    X = pd.get_dummies(X, columns=colunas_categoricas, drop_first=True)
    print(f"One-Hot Encoding aplicado às colunas categóricas: {list(colunas_categoricas)}")
else:
    print("Nenhuma coluna categórica encontrada para One-Hot Encoding.")


# --- TRATAMENTO DE VALORES NULOS EM X (AGORA MAIS ROBUSTO E APÓS ONE-HOT) ---
print("\n--- Tratamento Final de Valores Nulos em Features (X) ---")
# Identificar colunas numéricas (agora incluindo as geradas pelo get_dummies que são 0/1)
cols_to_impute = X.select_dtypes(include=np.number).columns
for col in cols_to_impute:
    if X[col].isnull().sum() > 0:
        mediana_col = X[col].median()
        X[col].fillna(mediana_col, inplace=True)
        print(f"NaNs na coluna numérica '{col}' preenchidos com a mediana: {mediana_col}")

if X.isnull().sum().sum() > 0:
    print(f"ALERTA: Ainda existem {X.isnull().sum().sum()} NaNs em X após a imputação. Verifique as colunas:")
    print(X.isnull().sum()[X.isnull().sum() > 0])
else:
    print("Nenhum NaN restante em X. Pronto para SMOTE e escalonamento.")
print("-" * 20)
# --- FIM DO TRATAMENTO DE VALORES NULOS ---


# Divisão em conjuntos de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# --- 4. Balanceamento de Classes com SMOTE ---
print("\n--- Balanceamento de Classes com SMOTE ---")
print(f"Contagem de classes antes do SMOTE (treino):\n{y_train.value_counts()}")

# Certifica-se de que LabelEncoder é aplicado apenas se y_train não for numérico
le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)
y_test_encoded = le.transform(y_test)
print(f"Classes originais: {le.classes_}")

# Ajusta k_neighbors para evitar erros em classes pequenas
unique_classes, counts = np.unique(y_train_encoded, return_counts=True)
min_samples_in_class = counts.min()

if min_samples_in_class <= 1:
    print(f"AVISO: A menor classe tem {min_samples_in_class} amostra(s) no conjunto de treino.")
    print("SMOTE pode não ser aplicável a classes com 1 amostra. Ajustando 'k_neighbors' e 'sampling_strategy'.")

    smote_k_neighbors = 1 # O menor valor possível para k_neighbors para gerar amostras
    # Cria uma estratégia de amostragem que exclui classes com menos de (k_neighbors + 1) amostras
    # Convert unique_classes to a list of original class names to use in sampling_strategy
    classes_with_enough_samples = [
        cls_name for cls_idx, cls_name in enumerate(le.classes_)
        if counts[cls_idx] > smote_k_neighbors
    ]

    if not classes_with_enough_samples: # Se todas as classes tiverem 1 ou 0 amostras
        print("ALERTA: Nenhuma classe tem amostras suficientes para SMOTE.")
        smote_applied = False
    else:
        # Use sampling_strategy as a list of class labels to be oversampled
        smote = SMOTE(random_state=42, k_neighbors=smote_k_neighbors, sampling_strategy=classes_with_enough_samples)
        smote_applied = True
elif min_samples_in_class < 5: # Se for entre 2 e 4
    print(f"AVISO: A menor classe tem {min_samples_in_class} amostras. Ajustando k_neighbors para {min_samples_in_class - 1}.")
    smote = SMOTE(random_state=42, k_neighbors=min_samples_in_class - 1)
    smote_applied = True
else: # Se a menor classe tem 5 ou mais amostras
    smote = SMOTE(random_state=42) # Usa o k_neighbors padrão de 5
    smote_applied = True

if smote_applied:
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train_encoded)
    print(f"Contagem de classes após SMOTE (treino):\n{pd.Series(y_train_res).value_counts()}")
else:
    X_train_res, y_train_res = X_train.copy(), y_train_encoded.copy() # Não aplica SMOTE
    print("SMOTE não foi aplicado devido a classes com poucas amostras.")
print("------------------------------------------")

# --- 5. Escalonamento de Features Numéricas ---
print("\n--- Escalonamento de Features Numéricas ---")
# Identificar colunas numéricas para escalonamento
colunas_numericas_para_escalar = X_train_res.select_dtypes(include=np.number).columns.tolist()

scaler = StandardScaler()
X_train_res[colunas_numericas_para_escalar] = scaler.fit_transform(X_train_res[colunas_numericas_para_escalar])
X_test[colunas_numericas_para_escalar] = scaler.transform(X_test[colunas_numericas_para_escalar])

print("Features numéricas escalonadas com sucesso.")
print("------------------------------------------")

# --- 6. Treinamento do Modelo de Classificação ---
print("\n--- Treinamento do Modelo de Classificação (Random Forest) ---")
model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
model.fit(X_train_res, y_train_res)
print("Modelo Random Forest treinado com sucesso. 🌳")
print("-------------------------------------------------")

# --- 7. Avaliação do Modelo ---
print("\n--- Avaliação do Modelo ---")
y_pred = model.predict(X_test)

# Decodifica as previsões para nomes de classes originais
y_pred_decoded = le.inverse_transform(y_pred)
y_test_decoded = le.inverse_transform(y_test_encoded) # Usa y_test_encoded para decodificar

accuracy = accuracy_score(y_test_decoded, y_pred_decoded)
print(f"Acurácia do modelo: {accuracy:.2f}")

print("\nRelatório de Classificação:")
print(classification_report(y_test_decoded, y_pred_decoded))

# Matriz de Confusão
plt.figure(figsize=(12, 8))
sns.heatmap(confusion_matrix(y_test_decoded, y_pred_decoded), annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
plt.title('Matriz de Confusão')
plt.xlabel('Previsto')
plt.ylabel('Real')
plt.tight_layout()
# plt.show()

# --- IMPORTÂNCIA DAS FEATURES ---
print("\n--- Importância das Features ---")
# Certifique-se de que o modelo e X_train_res estão definidos
if hasattr(model, 'feature_importances_') and X_train_res is not None:
    importances = model.feature_importances_
    feature_names = X_train_res.columns
    feature_importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
    feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False)
    print(feature_importance_df.head(15)) # Mostra as 15 features mais importantes
else:
    print("Não foi possível calcular a importância das features. Modelo ou X_train_res não disponíveis.")
print("------------------------------")


# --- 8. Salvamento do Modelo e Scaler ---
print("\n--- Salvamento do Modelo e Scaler ---")
os.makedirs("modelos_treinados", exist_ok=True)
joblib.dump(model, 'modelos_treinados/random_forest_model.joblib')
joblib.dump(scaler, 'modelos_treinados/scaler.joblib')
joblib.dump(le, 'modelos_treinados/label_encoder.joblib')
print("Modelo, scaler e LabelEncoder salvos em 'modelos_treinados/'. ✅")
print("-------------------------------------")
# --- 9. Análise de Features e Seleção ---
print("\n--- Análise de Features e Seleção (PCA ou SelectKBest) ---")

# Usaremos o SelectKBest para selecionar as features mais importantes
# A métrica f_classif é adequada para classificação com features numéricas
from sklearn.feature_selection import SelectKBest, f_classif

# O número de features a selecionar (pode ser ajustado)
# Vamos começar selecionando, por exemplo, as 20 melhores features
k_best_features = 20

# Inicializa o SelectKBest
# Usamos X_train_res (após SMOTE e escalonamento) para selecionar as features
# Usamos y_train_res (após SMOTE e encoding) como alvo
selector = SelectKBest(score_func=f_classif, k=k_best_features)

# Ajusta o seletor aos dados de treino (resampleados e escalonados)
selector.fit(X_train_res, y_train_res)

# Obtém as features selecionadas
selected_features_mask = selector.get_support()
selected_features_names = X_train_res.columns[selected_features_mask]

print(f"\nAs {k_best_features} features mais importantes (SelectKBest com f_classif) são:")
print(list(selected_features_names))

# Opcional: Visualizar os scores das features
feature_scores = pd.DataFrame({'Feature': X_train_res.columns, 'Score': selector.scores_})
feature_scores = feature_scores.sort_values(by='Score', ascending=False).reset_index(drop=True)

print("\nTop 30 Scores das Features:")
display(feature_scores.head(30))

# --- Aplicação da Seleção de Features ---
# Cria novos dataframes com apenas as features selecionadas
X_train_res_selected = selector.transform(X_train_res)
X_test_selected = selector.transform(X_test) # Aplica a mesma transformação ao conjunto de teste

# Converte de volta para DataFrame para manter nomes de colunas (opcional, mas útil)
X_train_res_selected = pd.DataFrame(X_train_res_selected, columns=selected_features_names, index=X_train_res.index)
X_test_selected = pd.DataFrame(X_test_selected, columns=selected_features_names, index=X_test.index)


print(f"\nShape de X_train_res após seleção de features: {X_train_res_selected.shape}")
print(f"Shape de X_test após seleção de features: {X_test_selected.shape}")
print("-----------------------------------------------------")

# --- Nota sobre PCA ---
print("\n--- Nota sobre PCA ---")
print("PCA é uma técnica de redução de dimensionalidade que cria novas componentes.")
print("Ela é útil para reduzir o número total de features e lidar com multicolinearidade,")
print("mas as novas componentes não têm uma interpretação direta como as features originais.")
print("Se a interpretabilidade das features for importante, SelectKBest é preferível.")
print("Se a redução de dimensionalidade e a remoção de multicolinearidade forem o foco principal,")
print("PCA pode ser explorado separadamente.")
print("----------------------")
# --- 10. Treinamento e Avaliação de Modelos Robustos (XGBoost e CatBoost) ---
print("\n--- Treinamento e Avaliação de Modelos Robustos ---")

# Instalar as bibliotecas se ainda não estiverem instaladas
try:
    import xgboost
    import catboost
except ImportError:
    print("Instalando XGBoost e CatBoost...")
    !pip install xgboost catboost lightgbm -q
    import xgboost
    import catboost

from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import time # Para medir o tempo de treinamento

# Vamos usar os dados com as features selecionadas
X_train_model = X_train_res_selected
X_test_model = X_test_selected
y_train_model = y_train_res # Continua usando os rótulos codificados para treinamento
y_test_model = y_test_encoded # Continua usando os rótulos codificados para avaliação

# --- Treinamento do XGBoost ---
print("\nTreinando modelo XGBoost...")
start_time = time.time()

# XGBoost Classifier
# class_weight não é diretamente suportado no XGBoost com múltiplas classes
# Uma alternativa é usar scale_pos_weight para cada classe, mas é mais complexo para multiclasse.
# Vamos usar os dados balanceados por SMOTE aqui.
xgb_model = XGBClassifier(objective='multi:softmax', # Para classificação multiclasse
                          num_class=len(le.classes_), # Número total de classes
                          eval_metric='mlogloss', # Métrica de avaliação para multiclasse
                          use_label_encoder=False, # Desativa o aviso de deprecated
                          n_estimators=100,
                          learning_rate=0.1,
                          random_state=42,
                          n_jobs=-1) # Usa todos os cores disponíveis

xgb_model.fit(X_train_model, y_train_model)

end_time = time.time()
print(f"Treinamento do XGBoost concluído em {end_time - start_time:.2f} segundos. 🚀")

# --- Avaliação do XGBoost ---
print("\nAvaliação do XGBoost:")
y_pred_xgb = xgb_model.predict(X_test_model)

# Decodifica as previsões de volta para os nomes originais das classes
y_pred_xgb_decoded = le.inverse_transform(y_pred_xgb)

accuracy_xgb = accuracy_score(y_test_decoded, y_pred_xgb_decoded)
print(f"Acurácia do XGBoost: {accuracy_xgb:.2f}")

print("\nRelatório de Classificação (XGBoost):")
print(classification_report(y_test_decoded, y_pred_xgb_decoded, target_names=le.classes_)) # Usa target_names para exibir nomes das classes

# Matriz de Confusão (XGBoost)
plt.figure(figsize=(12, 8))
sns.heatmap(confusion_matrix(y_test_decoded, y_pred_xgb_decoded), annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
plt.title('Matriz de Confusão - XGBoost')
plt.xlabel('Previsto')
plt.ylabel('Real')
plt.tight_layout()
# plt.show() # Descomente para exibir a matriz de confusão


# --- Treinamento do CatBoost ---
print("\nTreinando modelo CatBoost...")
start_time = time.time()

# CatBoost Classifier
# CatBoost pode lidar diretamente com features categóricas, mas já aplicamos One-Hot Encoding.
# Ele tem bons mecanismos internos para lidar com desbalanceamento.
cat_model = CatBoostClassifier(iterations=100, # Número de árvores
                               learning_rate=0.1,
                               loss_function='MultiClass', # Função de perda para multiclasse
                               eval_metric='MultiClass', # Métrica de avaliação
                               random_state=42,
                               verbose=0, # Não imprime o progresso durante o treinamento
                               # auto_class_weights='Balanced', # Pode ser explorado para desbalanceamento
                               thread_count=-1) # Usa todos os cores disponíveis


cat_model.fit(X_train_model, y_train_model)

end_time = time.time()
print(f"Treinamento do CatBoost concluído em {end_time - start_time:.2f} segundos. 🚀")

# --- Avaliação do CatBoost ---
print("\nAvaliação do CatBoost:")
y_pred_cat = cat_model.predict(X_test_model)
y_pred_cat = y_pred_cat.flatten() # Flatten a matriz de previsão se necessário

# Decodifica as previsões de volta para os nomes originais das classes
y_pred_cat_decoded = le.inverse_transform(y_pred_cat)

accuracy_cat = accuracy_score(y_test_decoded, y_pred_cat_decoded)
print(f"Acurácia do CatBoost: {accuracy_cat:.2f}")

print("\nRelatório de Classificação (CatBoost):")
print(classification_report(y_test_decoded, y_pred_cat_decoded, target_names=le.classes_)) # Usa target_names para exibir nomes das classes

# Matriz de Confusão (CatBoost)
plt.figure(figsize=(12, 8))
sns.heatmap(confusion_matrix(y_test_decoded, y_pred_cat_decoded), annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
plt.title('Matriz de Confusão - CatBoost')
plt.xlabel('Previsto')
plt.ylabel('Real')
plt.tight_layout()
# plt.show() # Descomente para exibir a matriz de confusão

print("\n--- Comparativo de Acurácia ---")
print(f"Acurácia Random Forest (modelo anterior): {accuracy:.2f}")
print(f"Acurácia XGBoost: {accuracy_xgb:.2f}")
print(f"Acurácia CatBoost: {accuracy_cat:.2f}")
print("-------------------------------")

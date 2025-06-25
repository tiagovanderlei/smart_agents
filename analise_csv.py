# -*- coding: utf-8 -*-
"""RAG (LangChain Expression Language) para dados fiscais
-------------------------------------------------------
Agora com suporte a **upload de um arquivo ZIP** contendo **dois CSVs**:
1. Itens das notas (ex.: `*_Itens.csv`)
2. Cabeçalho das notas (ex.: `*_Cabecalho.csv`)

O pipeline detecta automaticamente qual é qual, faz o merge e alimenta a cadeia RAG.
"""
from __future__ import annotations

import io
import os
import shutil
import zipfile
from typing import List, Tuple

import pandas as pd
from dotenv import load_dotenv
from langchain_community.document_loaders import DataFrameLoader
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter

###############################################################################
#  Utilidades de CSV/ZIP
###############################################################################


def detectar_dfs_itens_cabecalho(df1: pd.DataFrame, df2: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Tenta identificar qual dataframe é de itens e qual é de cabeçalho.

    Estratégia simples: se a primeira coluna contém a palavra "ITEM" ou "PROD",
    assume que é o dataframe de itens.
    """
    def eh_itens(df: pd.DataFrame) -> bool:
        primeira_col = df.columns[0].upper()
        return any(palavra in primeira_col for palavra in ("ITEM", "PROD", "DESCR"))

    if eh_itens(df1):
        return df1, df2
    if eh_itens(df2):
        return df2, df1
    # fallback: mantém ordem original
    return df1, df2


def carregar_csvs_do_zip(zip_bytes: bytes) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Extrai dois CSVs (itens e cabeçalho) de um arquivo ZIP em memória."""
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        csv_names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if len(csv_names) < 2:
            raise ValueError("O arquivo ZIP deve conter pelo menos dois CSVs (itens e cabeçalho).")

        # Carrega os dois primeiros CSVs encontrados
        with zf.open(csv_names[0]) as f1, zf.open(csv_names[1]) as f2:
            df1 = pd.read_csv(f1)
            df2 = pd.read_csv(f2)

        return detectar_dfs_itens_cabecalho(df1, df2)

###############################################################################
#  Preparação do DataFrame completo
###############################################################################


def preparar_dataframe_merged(df_itens: pd.DataFrame, df_cabecalho: pd.DataFrame) -> pd.DataFrame:
    """Merge, limpeza de colunas duplicadas e criação de coluna 'conteudo'."""
    df_completo = pd.merge(df_itens, df_cabecalho, on="CHAVE DE ACESSO", how="left")

    # Remove colunas duplicadas (_y)
    cols_to_drop = [col for col in df_completo.columns if col.endswith("_y")]
    df_completo = df_completo.drop(columns=cols_to_drop)

    # Limpa sufixo _x
    df_completo.columns = [
        col.replace("_x", "") if col.endswith("_x") else col for col in df_completo.columns
    ]

    # Concatena todas as colunas em texto
    df_completo["conteudo"] = df_completo.apply(
        lambda row: "\n".join(f"{col}: {row[col]}" for col in df_completo.columns), axis=1
    )
    return df_completo

###############################################################################
#  LangChain helpers
###############################################################################


def dataframe_para_docs(df: pd.DataFrame):
    return DataFrameLoader(df, page_content_column="conteudo").load()


def construir_rag_chain(docs, google_api_key: str):
    splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=0)
    splits = splitter.split_documents(docs)  # Mantém chunk por linha

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001", google_api_key=google_api_key
    )

    # Limpa vetorstore antiga
    shutil.rmtree("./chroma_db", ignore_errors=True)

    db = Chroma.from_documents(documents=splits, embedding=embeddings, persist_directory="./chroma_db")
    retriever = db.as_retriever(search_kwargs={"k": 100})

    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash", temperature=0, google_api_key=google_api_key
    )

    prompt = PromptTemplate.from_template(
        """Você é um assistente que responde com base nos dados abaixo:\n{context}\nPergunta: {question}\nResposta:"""
    )

    return (
        {
            "context": lambda x: retriever.invoke(x["question"]),
            "question": lambda x: x["question"],
        }
        | prompt
        | llm
        | StrOutputParser()
    )

###############################################################################
#  Função principal interativa (CLI)
###############################################################################


def main(path_zip: str | None = None):
    """Se `path_zip` for fornecido, usa o ZIP para gerar o dataframe; caso contrário, utiliza CSVs locais."""
    load_dotenv()
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        raise EnvironmentError("🚨 GOOGLE_API_KEY não encontrado no .env")

    # if path_zip:
    with open(path_zip, "rb") as f:
        itens_df, cab_df = carregar_csvs_do_zip(f.read())
    df = preparar_dataframe_merged(itens_df, cab_df)
    # else:
    #     # Fallback: arquivos locais convencionais
    #     itens_df = pd.read_csv("202401_NFs/202401_NFs_Itens.csv")
    #     cab_df = pd.read_csv("202401_NFs/202401_NFs_Cabecalho.csv")
    #     df = preparar_dataframe_merged(itens_df, cab_df)

    docs = dataframe_para_docs(df)
    rag_chain = construir_rag_chain(docs, google_api_key)

    # Loop interativo
    while True:
        query = input("🧠 Pergunta (ou 'sair'): ")
        if query.lower() in {"sair", "exit", "quit"}:
            break
        result = rag_chain.invoke({"question": query})
        print("📎 Resposta:", result)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="RAG Fiscal com suporte a ZIP")
    parser.add_argument("--zip", help="Caminho para o arquivo ZIP contendo 2 CSVs", required=False)
    args = parser.parse_args()

    main(path_zip=args.zip)

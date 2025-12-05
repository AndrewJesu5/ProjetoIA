# RAG Local

Um sistema de **Retrieval-Augmented Generation (RAG)** rodando 100% localmente, capaz de ingerir documentos personalizados e forçar a Inteligência Artificial a responder com base exclusivamente nesse contexto, ignorando seu treinamento original.


## Sobre o Projeto

Este projeto é uma implementação prática de uma arquitetura RAG utilizando Python. O objetivo principal foi criar um **Assistente Especialista** que roda offline, garantindo privacidade total dos dados e custo zero de API.

Para validar a eficácia do RAG, o sistema foi alimentado com um **Dataset de "Fatos Absurdos"**. O sucesso do projeto é demonstrado quando o modelo (Llama 3), que sabe que a capital do Brasil é Brasília, é "forçado" pelo contexto a responder que a capital é **Atlântida**.

A arquitetura utiliza `LangChain` para orquestração, `ChromaDB` para vetorização persistente e `Ollama` para a inferência do LLM local.


### Funcionalidades Implementadas

-   **Ingestão de Dados Híbrida:** Suporte automático para leitura de arquivos `.pdf` e `.txt` colocados na pasta de documentos.
-   **Banco Vetorial Persistente:** Utiliza `ChromaDB` para transformar textos em embeddings e salvá-los no disco, eliminando a necessidade de reprocessar arquivos a cada execução.
-   **Raciocínio Baseado em Contexto:** O sistema utiliza um *System Prompt* restritivo que instrui a IA a priorizar os documentos locais sobre seu conhecimento geral.
-   **Interface de Chat:** UI construída com `Streamlit` para uma experiência de conversação fluida, com histórico de mensagens e feedback visual de processamento ("pensando").
-   **Gerenciamento de Memória:** Otimizado para rodar em hardware doméstico (i7 + 12GB RAM) utilizando modelos quantizados via Ollama.


### Prova de Conceito (O Teste do Absurdo)

Para provar que a IA está lendo os arquivos e não "alucinando" com conhecimento da internet, o sistema foi alimentado com "fatos" absurdos:

<table>
  <tr>
    <td align="center"><strong>Chat Interativo</strong></td>
  </tr>
  <tr>
    <td><img src="readme-assets/chat_screenshot.png" width="400" alt="Tela do chat respondendo"></td>
  </tr>
</table>


### Estrutura e Tecnologias

-   **Linguagem:** Python
-   **Frontend:** `Streamlit` (Criação da interface web reativa).
-   **Orquestração de IA:** `LangChain` (Criação das Chains de recuperação e resposta).
-   **LLM Local:** `Ollama` (Executando modelos Llama 3 8B).
-   **Banco de Dados Vetorial:** `ChromaDB` (Armazenamento dos embeddings dos textos).
-   **Embeddings:** `HuggingFace` (`all-MiniLM-L6-v2`) para transformar texto em vetores numéricos leves.
-   **Carregadores:** `PyPDF` e `TextLoader` para ingestão de arquivos.
import json
import os
import time
from pathlib import Path
from google import genai
from google.genai import types
from google.genai.errors import APIError

# Inicializa o cliente usando a variável de ambiente GEMINI_API_KEY
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

# Caminhos dos documentos PDF e do arquivo de saída
arquivos_pdf = [
    "arquivos/Organograma.pdf",
    "arquivos/Cadeia de Valor.pdf",
    "arquivos/Regulamento Geral.pdf",
]
arquivo_saida = Path("resposta_modelo.json")

# Fila de modelos gratuitos caso o principal esteja sobrecarregado
modelos_disponiveis = [
    "gemini-4-hydrogen",
    "gemini-3.8-flash",
    "gemini-3.5-flash-lite",
]

arquivos_enviados = []

try:
    # 1. Upload dos PDFs via Files API
    for caminho in arquivos_pdf:
        arquivo_path = Path(caminho)
        if not arquivo_path.exists():
            raise FileNotFoundError(f"Arquivo não encontrado: {caminho}")

        print(f"Enviando {arquivo_path.name}...")
        upload = client.files.upload(file=arquivo_path)
        arquivos_enviados.append((arquivo_path.name, upload))

    # 2. Monta a lista de conteúdos rotulando cada PDF antes do arquivo enviado
    conteudos = []
    for nome_arquivo, arquivo_api in arquivos_enviados:
        conteudos.append(f"Documento anexado: {nome_arquivo}")
        conteudos.append(arquivo_api)

    prompt = (
        """Os três documentos fornecidos tratam da estrutura organizacional do Tribunal Regional do Trabalho da Quinta Região, com o Organograma e a Cadeia de Valor servindo de complemento ao Regulamento Geral. Extraia a lista de unidades administrativas que possuem atuação direta no ciclo de vida da estrutura organizacional do TRT. Inclua cargos individuais e coordenadorias caso os mesmos sejam relevantes e de alta hierarquia. Apresente o resultado apenas no formato de lista simples {"SIGLA": "Nome da Unidade Principal",}, e organize a saída por ordem decrescente de acordo com a hierarquia interna da organização. Caso não haja uma sigla predefinida, crie a sigla usando as iniciais do termo ignorando os conectivos . A saída DEVE ser um schema JSON nos moldes apresentados.
            Exemplo:
            DG - Diretoria-Geral
            CSAC - Coordenadoria de Serviços de Apoio e Conservação
            CSI - Coordenadoria de Segurança Institucional
            SVC - Coordenadoria de Contabilidade
            CML - Coordenadoria de Material e Logística
            SA - Secretaria de Administração
            SOF - Secretaria de Orçamento e Finanças
            CLC - Coordenadoria de Licitações e Contratos"""
    )
    conteudos.append(prompt)

    configuracao = types.GenerateContentConfig(
        system_instruction=(
            "Você é um assistente especializado em análise normativa e organizacional. "
            "Responda estritamente com base nos PDFs fornecidos e retorne apenas "
            "um objeto JSON válido, sem textos explicativos adicionais."
        ),
        response_mime_type="application/json",
    )

    # 3. Chamada ao modelo com retentativas e fallback automático
    response = None
    for modelo in modelos_disponiveis:
        for tentativa in range(1, 4):
            try:
                print(
                    f"Gerando resposta com {modelo} (tentativa {tentativa}/3)...")
                response = client.models.generate_content(
                    model=modelo,
                    contents=conteudos,
                    config=configuracao,
                )
                break
            except APIError as e:
                if e.code in (503, 429):
                    print(
                        f"Modelo {modelo} indisponível (erro {e.code}). Aguardando 10s...")
                    time.sleep(10)
                else:
                    raise
        if response is not None:
            break

    if response is None:
        raise RuntimeError(
            "Todos os modelos estão temporariamente sobrecarregados. Tente novamente em alguns minutos."
        )

    # 4. Valida e salva a resposta no arquivo .json preservando acentuação
    dados_json = json.loads(response.text)

    with arquivo_saida.open("w", encoding="utf-8") as f:
        json.dump(dados_json, f, ensure_ascii=False, indent=2)

    print(f"Arquivo salvo com sucesso em: {arquivo_saida.resolve()}")

finally:
    # 5. Remove os arquivos temporários dos servidores
    for _, arquivo_api in arquivos_enviados:
        client.files.delete(name=arquivo_api.name)

# Wav2Vec2 Audio Transcription

Esse repositório contém um script para transcrição de áudios usando o modelo Wav2Vec2 para a língua portuguesa brasileira.

## Requisitos

Antes de executar o script, certifique-se de ter instalado os seguintes pacotes:

```bash
pip install torch torchaudio pandas transformers
```

## Uso

1. Clone este repositório:

```bash
git clone https://github.com/beatrizalmeidaf/transcription-wav2vec2.git
cd wav2vec2
```

2. Execute o script de transcrição:

```bash
python wav2vec2.py --audio_folder "caminho/para/pasta/de/audios" --output_csv "caminho/para/saida/transcricoes.csv"
```

### Parâmetros:
- `--audio_folder`: Caminho para a pasta contendo os arquivos `.wav` que serão transcritos.
- `--output_csv`: Caminho onde será salvo o arquivo CSV com as transcrições.

## Exemplo de Uso

```bash
python wav2vec2.py --audio_folder "./audios" --output_csv "./transcricoes.csv"
```

## Estrutura do CSV
O arquivo CSV gerado terá duas colunas:
- **Nome**: Nome do arquivo de áudio transcrito.
- **Transcrição Wav2Vec2**: Texto transcrito a partir do áudio.

## Observações
- O script segmenta áudios longos em partes de 30 segundos para evitar problemas de memória.
- O modelo utilizado é `alefiury/wav2vec2-large-xlsr-53-coraa-brazilian-portuguese-gain-normalization-sna`, treinado para transcrição de português brasileiro.

import os
import argparse
import torch
import torchaudio
import pandas as pd
from transformers import AutoTokenizer, Wav2Vec2ForCTC, Wav2Vec2Processor

def transcribe_audio(waveform, sample_rate, processor, model):
    """Transcreve um áudio usando Wav2Vec2."""
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = resampler(waveform)
    
    input_values = processor(waveform, return_tensors="pt", sampling_rate=16000).input_values
    logits = model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    return processor.decode(predicted_ids[0])

def process_audio_file(file_path, processor, model):
    """Processa um único arquivo de áudio e retorna a transcrição."""
    try:
        waveform, sample_rate = torchaudio.load(file_path)
        waveform = waveform.squeeze()

        if len(waveform.shape) > 1:
            waveform = waveform[0]

        # Segmentar áudio para evitar estouro de memória
        segment_duration = 30  # segundos
        segment_length = segment_duration * sample_rate

        transcripts = []
        for start in range(0, waveform.size(0), segment_length):
            end = min(start + segment_length, waveform.size(0))
            transcript = transcribe_audio(waveform[start:end], sample_rate, processor, model)
            transcripts.append(transcript)

        return " ".join(transcripts)
    except Exception as e:
        print(f"Erro ao processar {file_path}: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Transcrição de áudio usando Wav2Vec2")
    parser.add_argument("--audio_folder", type=str, required=True, help="Caminho para a pasta contendo os arquivos de áudio")
    parser.add_argument("--output_csv", type=str, required=True, help="Caminho para o arquivo CSV de saída")
    args = parser.parse_args()
    
    # Carregar modelo e processador
    MODEL_ID = "alefiury/wav2vec2-large-xlsr-53-coraa-brazilian-portuguese-gain-normalization-sna"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = Wav2Vec2ForCTC.from_pretrained(MODEL_ID)
    processor = Wav2Vec2Processor.from_pretrained(MODEL_ID)

    transcriptions = []
    
    if not os.path.exists(args.audio_folder):
        print(f"Pasta de áudios não encontrada: {args.audio_folder}")
        return

    for filename in os.listdir(args.audio_folder):
        if filename.endswith(".wav"):
            file_path = os.path.join(args.audio_folder, filename)
            print(f"Processando: {filename}")
            transcription = process_audio_file(file_path, processor, model)

            if transcription:
                transcriptions.append([filename, transcription])
    
    df = pd.DataFrame(transcriptions, columns=["Nome", "Transcrição Wav2Vec2"])
    df.to_csv(args.output_csv, index=False, encoding="utf-8")
    
    print(f"Transcrições salvas em {args.output_csv}")

if __name__ == "__main__":
    main()

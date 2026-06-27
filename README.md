# Visão Computacional

Captura de imagem e detecção de pontos faciais (linhas de expressão) com
[OpenCV](https://opencv.org/) e [MediaPipe](https://developers.google.com/mediapipe).

## Requisitos

- Python 3.9+
- Uma câmera conectada ao sistema

## Instalação

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Uso

### Capturar uma imagem

```bash
python capture_image.py
```

Opções:

| Opção              | Descrição                                             | Padrão                |
| ------------------ | ----------------------------------------------------- | --------------------- |
| `-o`, `--output`   | Caminho do arquivo de saída                           | `captured_image.jpg`  |
| `-c`, `--camera`   | Índice do dispositivo de câmera                       | `0`                   |
| `-w`, `--warmup`   | Frames descartados antes da captura                   | `5`                   |
| `-a`, `--annotate` | Desenha as linhas de expressão na imagem capturada    | desativado            |
| `-v`, `--verbose`  | Ativa logs detalhados                                 | desativado            |

Exemplo — capturar já com as linhas de expressão desenhadas:

```bash
python capture_image.py --annotate --output fotos/rosto.jpg -v
```

### Detectar linhas de expressão em uma imagem existente

```bash
python detect_landmarks.py caminho/para/foto.jpg
```

São desenhados os contornos das regiões que caracterizam as expressões
faciais: olhos, sobrancelhas, boca, nariz e oval do rosto. Por padrão, a
saída é `<nome>_landmarks.<ext>` no mesmo diretório do arquivo de entrada.

Opções:

| Opção            | Descrição                                              | Padrão |
| ---------------- | ------------------------------------------------------ | ------ |
| `-o`, `--output` | Caminho do arquivo de saída                            | auto   |
| `--max-faces`    | Número máximo de rostos a detectar                     | `1`    |
| `--confidence`   | Confiança mínima da detecção (0 a 1)                   | `0.5`  |
| `--tesselation`  | Também desenha a malha facial completa                 | desativado |
| `-v`, `--verbose`| Ativa logs detalhados                                  | desativado |

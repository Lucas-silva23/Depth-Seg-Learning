# Depth-Seg-Learning

**Segmentação Semântica de Imagens Aéreas de Drone Utilizando Aprendizado Multitarefa**

Este repositório contém o código e experimentos do projeto de segmentação semântica de imagens aéreas utilizando uma abordagem de aprendizado multitarefa. O objetivo é melhorar a performance da segmentação ao incorporar uma tarefa auxiliar de estimativa de profundidade durante o treinamento da rede.

## 📌 Objetivo

Desenvolver e avaliar um modelo multitarefa baseado em redes neurais convolucionais (CNNs), capaz de realizar segmentação semântica e estimativa de profundidade simultaneamente, visando melhorar a precisão da segmentação de cenas aéreas complexas capturadas por drones.

## 🧠 Abordagem

- Utilização do dataset sintético **Swisstopo** com imagens RGB, máscaras semânticas e mapas de profundidade gerados via **Depth Anything**.
- Arquitetura **encoder-decoder** com backbone **ResNet50** e **ResNet101**.
- Dois decoders: um para segmentação semântica e outro para estimativa de profundidade.
- Função de perda ponderada combinando:
  - Cross-Entropy para segmentação.
  - BerHu Loss para profundidade.


# Mapeamento Visão-Ação: Controle de Manipulador por Percepção 3D (RAISE-BOT)

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9+-blue.svg" alt="Python Version">
  <img src="https://img.shields.io/badge/Licença-MIT-green.svg" alt="License">
  <img src="https://img.shields.io/badge/Status-Concluído-brightgreen.svg" alt="Project Status">
</p>

<p align="center">
  Um sistema de robótica e visão computacional que traduz a seleção de um ponto em uma imagem 2D para um movimento preciso de um braço robótico no espaço 3D.
</p>

<p align="center">
  <img src="assets/gif.gif" alt="Demonstração do Projeto RAISE-BOT">
</p>

## 📜 Tabela de Conteúdos

- [Sobre o Projeto](#-sobre-o-projeto)
- [✨ Features](#-features)
- [🛠️ Tecnologias Utilizadas](#️-tecnologias-utilizadas)
- [⚙️ Configuração do Ambiente](#️-configuração-do-ambiente)
- [🚀 Como Executar](#-como-executar)
- [📂 Estrutura do Projeto](#-estrutura-do-projeto)
- [📈 Simulação](#-simulação)
- [🎥 Vídeos de Demonstração](#-vídeos-de-demonstração)
- [💡 Trabalhos Futuros](#-trabalhos-futuros)

## Sobre o Projeto

Este projeto, intitulado **RAISE-BOT (Robotic Acoustic Inspection with Surface Estimation)**, foi desenvolvido como parte do curso de Engenharia de Controle e Automação da PUC-Rio. O sistema implementado constitui a fundação para tarefas de interação robótica com o ambiente, focando na problemática do mapeamento visão-ação.

O principal desafio resolvido foi criar um pipeline robusto que permite a um operador selecionar um ponto de interesse em um stream de vídeo e comandar um manipulador robótico para alcançar as coordenadas 3D correspondentes àquele ponto. Para isso, o sistema integra uma câmera de profundidade 3D, um braço robótico de 4 graus de liberdade e uma cadeia de software que gerencia desde a percepção sensorial até a atuação motora, passando por complexas transformações de coordenadas e cálculos de cinemática inversa.

## Features

- **Percepção 3D em Tempo Real:** Visualização de streams de vídeo RGB e mapas de profundidade alinhados.
- **Interface Interativa:** Permite selecionar pontos de interesse com um clique do mouse.
- **Cadeia de Transformação Completa:** Calcula coordenadas 3D no referencial da câmera e as transforma para o referencial do robô.
- **Controle Robótico Ponto-a-Ponto:** Move o braço robótico para o ponto 3D selecionado.
- **Validação de Workspace:** Verifica se o ponto alvo está dentro do espaço de trabalho seguro do robô antes do movimento.
- **Simulador Integrado:** Um simulador em Matplotlib para validar a lógica de transformação sem a necessidade do hardware físico.

## Tecnologias Utilizadas

### Hardware

| Componente | Descrição |
| :--- | :--- |
| **Câmera 3D** | Luxonis OAK-D |
| **Braço Robótico** | Interbotix Pincher (4-DoF) |
| **Servomotores** | Dynamixel AX-12A |
| **Controladora** | Arbotix-M |

### Software

| Tecnologia | Descrição |
| :--- | :--- |
| **Python 3.9+** | Linguagem principal de desenvolvimento. |
| **OpenCV** | Para a criação da GUI e processamento de imagem. |
| **NumPy** | Para todas as operações numéricas e de álgebra linear. |
| **DepthAI** | API oficial para interface com a câmera OAK-D. |
| **PySerial** | Para comunicação serial com a placa Arbotix-M. |
| **Matplotlib**| Para a construção do simulador gráfico. |

## ⚙️ Configuração do Ambiente

Siga os passos abaixo para configurar o ambiente de desenvolvimento.

1. **Pré-requisitos:**
    - Python 3.9 ou superior instalado.

2. **Clone o Repositório:**

    ```bash
    git clone [https://github.com/palomaflsette/raise-bot.git](https://github.com/palomaflsette/raise-bot.git)
    cd raise-bot
    ```

3. **Crie e Ative um Ambiente Virtual (Recomendado):**

    ```bash
    python -m venv venv
    # Windows
    .\venv\Scripts\activate
    # macOS/Linux
    source venv/bin/activate
    ```

4. **Instale as Dependências:**
    Crie um arquivo chamado `requirements.txt` na raiz do projeto com o seguinte conteúdo:

    ```txt
    numpy
    opencv-python
    depthai
    pyserial
    matplotlib
    ```

    Em seguida, instale as bibliotecas com o pip:

    ```bash
    pip install -r requirements.txt
    ```

## Como Executar

1. **Conecte o Hardware:**
    - Conecte a câmera OAK-D ao computador via USB-C.
    - Conecte a placa Arbotix-M ao computador via USB.
    - Ligue a fonte de alimentação de 12V para os servos Dynamixel.

2. **Verifique a Porta COM:**
    - No arquivo `monolito.py`, verifique se a variável `ROBOT_PORT` corresponde à porta serial correta da sua placa Arbotix-M (ex: `'COM7'` no Windows, `'/dev/ttyUSB0'` no Linux).

3. **Execute o Programa Principal:**

    ```bash
    python main.py
    ```

4. **Interagindo com a Interface:**
    - **Clique** na imagem da esquerda para selecionar um ponto e ver as coordenadas calculadas.
    - Pressione a tecla **`M`** para mover o robô para o último ponto clicado.
    - Pressione a tecla **`H`** para enviar o robô à posição "Home".
    - Pressione a tecla **`I`** para ver a lista completa de comandos no console.
    - Pressione **`Q`** ou **`ESC`** para fechar.

## Simulação

O projeto inclui um simulador (`simulator.py`) que foi utilizado para desenvolver e validar a lógica de transformação de coordenadas antes da implementação no hardware. Ele permite clicar em uma cena 2D virtual e visualizar a transformação 3D correspondente.

Para executá-lo:

```bash
python simulator.py
```

## 🎥 Vídeos de Demonstração
O funcionamento do sistema foi gravado e está disponível no YouTube:

Vídeo 1: Demonstração Geral do Sistema

Vídeo 2: Teste de Precisão e Repetibilidade

## Trabalhos Futuros
- Controle de Orientação Normal: Implementar o controle da orientação do efetuador para que ele aborde superfícies de forma perpendicular, utilizando o módulo de análise de perfil de superfície já desenvolvido.

- Calibração Automática: Criar uma rotina que utilize marcadores ArUco para calcular a matriz T_cam_to_robot automaticamente.

- Aplicação em Inspeção Acústica: Integrar um sensor de ultrassom para realizar inspeções de materiais, aproveitando o controle de pose.

- Arquitetura Modular: Refatorar o código para uma arquitetura baseada em classes ou migrar para o framework ROS.




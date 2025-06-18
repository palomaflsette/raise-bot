
TITLE = "RAISE: Robotic Acoustic Inspection with Surface Estimation"
README_URL = "https://github.com/palomaflsette/raise-bot/blob/main/README.md"
ABOUT_TEXT = (
    "RAISE: Robotic Acoustic Inspection with Surface Estimation\n\n"
    "Desenvolvido por Paloma Sette sob orientação de Wouter Caarls na PUC-Rio, 2025."
)

""" 
Recapitulando como será nosso sistema:

agora que terminamos de configurar a camera , pegar a depth image, as profundidades e colocar num grafico, pegando as normais etc..
agora precisaremos ponderar:

- a camera estará enxergando de cima tanto oobjeto (mais ao centro) e o robo. O objeto estará parado e terá uma superficie irregular, com diversas
ondulações... então, o objetivo é definirmos pontos nessa superficie, e desses pontos definidos, queremos pegar as NORMAIS, e essa informação será mandada
para o robo, que irá encostar num ponto determinado de forma que o efetuador e o ultimo elo fiquem perpendiculares, ou seja, paralelos à normal
daquele ponto na superficie. 

- sei que precisaremos fazer a transformação entre camera e robo tbm...

o maninpulador q usaremos será um desses que usei nos projetos de laboratorio da disciplina, que teve projeto 1 ao 6, vc se lembra.

então, precisamos agora organizar todos os proximos passos, tanto em termos matematicos quanto em termos de codigo. e precisamos que dê tempo de implementaros
a parte de captação acustica com um piezo , mas isso pensamos mais pra frete, vamos tentar construir ess parte q falei agora pq ela é a parte
obrigatoria.
"""
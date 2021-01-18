# Sistema recomanador per la formació d'equips
Treball de Final de Grau - Realitzat per David Valero Masachs i dirigit per Prof. Dr. Cecilio Angulo Bahón.
Trobaràs tota la memòria del projecte a `./documentation/memoria.pdf`.

## Resum ##

En el món universitari cada vegada és més freqüent que els estudiants hagin de fer pràctiques en una empresa durant el 
seu grau universitari, així i tot no hi ha sistemes que simplifiquin la feina d'assignació dels estudiants als diferents 
projectes oferts per les empreses.

Aquest treball final de grau presenta, mitjançant l'ús d'algorismes de reinforcement learning (aprenentatge per 
reforç), una eina que dóna als estudiants tantes alternatives com es vulgui segons les seves preferències, els projectes
que s'ofereixen i els perfils d'estudiants que volen cadascuna de les empreses per cada projecte. A més, també permet 
que cada estudiant pugui ordenar-les al seu gust i el sistema, intentant satisfer al màxim possible als estudiants, 
dóna una assignació final.

Per dur a terme aquest projecte s'ha utilitzat Python com a llenguatge de programació i la llibreria OpenAI per 
facilitar la creació de l'entorn. A causa de les característiques del problema, per l'agent s'ha utilitzat la llibreria 
Stable Baselines i l'algorisme d'aprenentatge per reforç A2C per realitzar la primera assignació.

L'assignació final també utilitza la mateixa estructura i algorisme, però per facilitar l'entrenament, un bucle assigna
prèviament tots els estudiants que pot assignar directament i l'agent A2C és l'encarregat d'assignar els que no se'ls hi
ha pogut assignar cap projecte de la seva llista inicial.


## Arxius i la seva funcionalitat ##
### Carpeta ./samples ###
Trobarà tots els executables del projecte. S'han creat els següents scripts:

- **generateJSON.py**
  
  Script que genera un JSON aleatori d'alumnes i projectes segons questions4students.md i questions4companies.md.
  Per executar-lo cal posar:
  
  `python generateJSON.py numeroAlumnes numeroProjectes`
  
  On:
  - `numeroAlumnes` és el nombre d'alumnes a generar, si no es posa valor, s'assignarà el valor per defecte, 100.
  - `numeroProjectes` és el nombre de projectes a generar, si no es posa valor, s'assignarà el valor per defecte, 20.
  
  Una vegada acaba d'executar, s'exporta els resultats a 'studentsProjectsData.json',
  on estan tots els estudiants i projectes creats.


- **recommender.py**
  
  Script que utilitza Reinforcement Learning per assignar tots els estudiants als projectes,
  per executar cal disposar del json 'studentsProjectsData.json' generat per l'anterior script i executar:
  
  `python recommender.py modo numeroOpcions tipusAgent`
  
  On:
  - `modo` és un bool que indica si es vol que s'executi en mode assignació (modo = 0) o reassignació (modo = 1). 
    El valor per defecte és 0, assignació.
  - `numeroOpcions` és el nombre d'opcions que es vol generar per estudiant, 
    si no es posa valor, s'assignarà el valor per defecte, 3.
  - `tipusAgent` és el número de l'agent que es vol entrenar, hi ha les següents opcions,
  si no es posa valor, s'assignarà el valor per defecte, 2, AC2:
    - 0: Random, Assignació aleatòria complint requisits.
    - 1: Q Learning
    - 2: AC2
  
  Una vegada acaba d'executar, exporta els resultats en un arxiu JSON anomenat `optionsData.json`, on estan les opcions
  que té cada estudiant amb la seva mitjana. A més, si és QLearning exporta QTable a `qTableData.csv`.


- **selector.py**
  
  Script que selecciona una de les opcions disponibles per cada estudiant aleatòriament. 
  Per executar cal disposar del json `optionsData.json` generat per l'anterior script i:
  
  `python selector.py`
  
  Una vegada acaba d'executar, s'exporta els resultats a `optionsSelectedData.json`, 
  on estan els estudiants ordenats per nota i amb l'opció triada.


- **runItAll.py**
  
  Script que executa tots els scripts anteriors seguits.
  
  `python runItAll.py numeroAlumnes numeroProjectes numeroOpcions tipusAgent`
  
  On:
  - `numeroAlumnes` és el nombre d'alumnes a generar, si no es posa valor, s'assignarà el valor per defecte, 100.
  - `numeroProjectes` és el nombre de projectes a generar, si no es posa valor, s'assignarà el valor per defecte, 20.
  - `numeroOpcions` és el nombre d'opcions que vols que generi per estudiant, si no es posa valor, s'assignarà el 
  valor per defecte, 3.
  - `tipusAgent` és el tipus d'agent que es vol utilitzar, funciona igual que l'script `recommender.py`, 
  si no es posa valor, s'assignarà el valor per defecte, AC2.
  
  Una vegada acaba d'executar, s'exporta els resultats a `studentsProjectsData.json`, 
  on estan tots els estudiants i projectes creats.

### Carpeta ./documentation ###
Trobarà tota la documentació del projecte.

- **momoria.pdf**:
Memòria del projecte.

- **questions4students.md**:
Recull de preguntes que se li faran als estudiants.

- **questions4companies.md**:
Recull de preguntes que se li faran a les empreses.


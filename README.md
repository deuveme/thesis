# Sistema recomanador per la formació d'equips
Treball de Final de Grau - Realitzat per David Valero Masachs i dirigit per Prof. Dr. Cecilio Angulo Bahón.
Trobaràs tota la memòria del projecte a `./documentation/memoria.pdf`.

## Arxius i la seva funcionalitat: ##
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


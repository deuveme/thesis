# Sistema recomanador per la formació d'equips
Treball de Final de Grau - Realitzat per David Valero.

### Arxius i la seva funcionalitat: ###
#### Carpeta /Samples ####
Trobarà tots els executables del projecte.

**generateJSON.py**:
Script que genera un JSON aleatori d'alumnes i projectes segons questions4students.md i questions4companies.md.
Per executar-lo cal posar:

`python generateJSON.py numeroAlumnes numeroProjectes`

On:
- `numeroAlumnes` es el número d'alumenes a generar, si no es posa valor s'assignarà el valor per defecte, 100.
- `numeroProjectes` es el número de projectes a generar, si no es posa valor s'assignarà el valor per defecte, 20.

Una vegada acaba d'executar, s'exporta els resultats a `studentsProjectsData.json`, 
on estan tots els estudiants i projectes creats.

**recommender.py**:
Script que utilitza Reinforcement Learning per asignar tots els estudiants als projects, 
per executar cal disposar del json `studentsProjectsData.json` generat per l'anterior script i:

`python recommender.py siEntrena tipusAgent numeroOpcions dataImportada`

On:
- `siEntrena` es un bool que indica si es vol que entreni abans o no, 
si no es posa valor s'assignarà el valor per defecte, 1, entrenar.
- `tipusAgent` es el número del agent que es vol entrenar, hi ha les següents opcions, 
si no es posa valor s'assignarà el valor per defecte, 2, AC2:
    - 0: Random, Asignació aleatoria complint requisits.
    - 1: Q Learning 
    - 2: AC2
    - 4: PPO2
    
- `numeroOpcions` es el número d'opcions que es vol generar per estudiant, 
si no es posa valor s'assignarà el valor per defecte, 3.
- `dataImportada` es el bool que indica si s'importa data externa o no (exemple QTable de QLearning), default 0, no.

Una vegada acaba d'executar, si es QLearning exporta QTable a qTableData.csv i els resultats a `optionsData.json`, 
on estan les opcions que te cada estudiant amb la seva mitjana.


**selector.py**:
Script que selecciona una de les opcions disponibles per cada estudiant aleatoriament. 
Per executar cal disposar del json `optionsData.json` generat per l'anterior script i:

`python selector.py`

Una vegada acaba d'executar, s'exporta els resultats a `optionsSelectedData.json`, 
on estan els estudiants ordenats per nota i amb l'opció triada.

**generateJSON.py**:
Script que genera un JSON aleatori d'alumnes i projectes segons questions4students.md i questions4companies.md.
Per executar-lo cal posar:

`python generateJSON.py numeroAlumnes numeroProjectes`

On:
- `numeroAlumnes` es el número d'alumenes a generar, si no es posa valor s'assignarà el valor per defecte, 100.
- `numeroProjectes` es el número de projectes a generar, si no es posa valor s'assignarà el valor per defecte, 20.

Una vegada acaba d'executar, s'exporta els resultats a `studentsProjectsData.json`, 
on estan tots els estudiants i projectes creats.

**runItAll.py**:
Script que executa tots els scripts anteriors seguits.

`python runItAll.py numeroAlumnes numeroProjectes numeroOpcions tipusAgent`

On:
- `numeroAlumnes` es el número d'alumenes a generar, si no es posa valor s'assignarà el valor per defecte, 100.
- `numeroProjectes` es el número de projectes a generar, si no es posa valor s'assignarà el valor per defecte, 20.
- `numeroOpcions` es el número d'opcions que vols que generi per estudiant, si no es posa valor s'assignarà el 
valor per defecte, 3.
- `tipusAgent` es el tipus d'agent que es vol utilitzar, funciona igual que l'script `recommender.py`, 
si no es posa valor s'assignarà el valor per defecte, AC2.

Una vegada acaba d'executar, s'exporta els resultats a `studentsProjectsData.json`, 
on estan tots els estudiants i projectes creats.

#### Carpeta /Documentation ####
Trobarà tota la documentació del projecte.

**questions4students.md**:
Recull de preguntes que se li faran als estudiants.


**questions4companies.md**:
Recull de preguntes que se li faran a les empreses.


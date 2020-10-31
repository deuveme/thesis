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

**agent.py**:
Script que utilitza Reinforcement Learning per asignar tots els estudiants als projects, per executar-ho cal posar:

`python agent.py numeroOpcions`

On `numeroOpcions` es el número d'opcions que es vol generar per estudiant, 
si no es posa valor s'assignarà el valor per defecte, 3.

#### Carpeta /Documentation ####
Trobarà tota la documentació del projecte.

**questions4students.md**:
Recull de preguntes que se li faran als estudiants.


**questions4companies.md**:
Recull de preguntes que se li faran a les empreses.


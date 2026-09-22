# L08 - Lösningsförslag: Dense-lager i C++ (del I)
Lösningsförslag till övningsuppgiften i [bilaga B](../appendix/b_exercises.md).
Katalogen innehåller den del av `ml`-kodbasen som tillkommer under denna lektion:
* Enumerationsklassen `ActFunc`, som anger ett lagers aktiveringsfunktion.
* Klassen `Dense`, en skarp implementation av dense-lagrets interface.
* Ett testprogram som bygger nätverket från **L07** av två `Dense`-lager i stället för två stubbar.

---

## Vad L08 bygger
**En utbyggnad av L06–L07.** Dense-lagrets interface, stubben samt hela `ml::neural_network`
följer med oförändrade från [lösningsförslaget från **L07**](../../L07/exercises/README.md), där de
beskrivs. Det som tillkommer är:
* `ActFunc` i [types.h](./include/ml/types.h).
* Klassen `Dense`, deklarerad i [dense.h](./include/ml/dense_layer/dense.h) och definierad i
  [dense.cpp](./source/ml/dense_layer/dense.cpp).
* En rad i [makefilen](./Makefile), samt två lagerdeklarationer i
  [main.cpp](./source/main.cpp).

**Ett lager som äger sina parametrar.** Stubben från **L06** höll enbart utdata, fel och vikter,
eftersom den aldrig beräknade någonting. `Dense` har dessutom en biasvektor och en vektor för den
viktade summan innan aktiveringsfunktionen, vilket är precis vad feedforward, backpropagation och
optimering behöver under **L09**.

**Aktiveringsfunktionen väljs per lager.** Vilken aktiveringsfunktion ett lager använder anges vid
konstruktion och sparas som ett värde i lagret. Själva matematiken finns ännu inte i kodbasen; den
läggs till under **L09**, i hjälpfunktioner som väljer beräkning utifrån just detta värde.

**Beräkningarna kommer i L09.** Klassen är i övrigt färdig: konstruktorn, samtliga getters samt
argumentkontrollerna i de fyra beräkningsmetoderna. Metoderna kontrollerar sina argument och
returnerar resultatet utan att beräkna någonting, vilket räcker för att nätverket från **L07** ska
gå att bygga och köra av `Dense`-lager redan nu.

---

## Filer

| Fil | Innehåll |
|---|---|
| [include/ml/types.h](./include/ml/types.h) | Matrisaliasen från **L06**, samt den nytillkomna enumerationsklassen `ActFunc` med `Relu`, `Tanh` och `None`, med `std::uint8_t` som underliggande typ. |
| [include/ml/dense_layer/interface.h](./include/ml/dense_layer/interface.h) | Oförändrad från **L06**: `ml::dense_layer::Interface`, det kontrakt som `Dense` nu uppfyller. |
| [include/ml/dense_layer/stub.h](./include/ml/dense_layer/stub.h) | Oförändrad från **L06**: `ml::dense_layer::Stub`, som `Dense` inte ersätter, utan kompletterar. |
| [include/ml/dense_layer/dense.h](./include/ml/dense_layer/dense.h) | Deklarationen av `ml::dense_layer::Dense`, markerad `final`: en `explicit` konstruktor med aktiveringsfunktionen som tredje argument, samtliga överlagringar från interfacet, raderade default-, copy- och move-operationer, samt sex medlemsvariabler. |
| [source/ml/dense_layer/dense.cpp](./source/ml/dense_layer/dense.cpp) | Implementationen av `Dense`: konstruktorns två kontroller och allokeringar, samtliga getters, samt de fyra beräkningsmetodernas argumentkontroller. |
| [include/ml/neural_network/](./include/ml/neural_network/) samt [source/ml/neural_network/](./source/ml/neural_network/) | Oförändrade från **L07**: nätverkets interface samt klassen `Shallow`. |
| [source/main.cpp](./source/main.cpp) | Testprogrammet enligt avsnitt 10 i bilaga B: samma program som i **L07**, men lagren är `Dense` i stället för `Stub`, och utgångslagret skapas med `ActFunc::Tanh`. |
| [Makefile](./Makefile) | Bygger `source/main.cpp`, `source/ml/dense_layer/dense.cpp` och `source/ml/neural_network/shallow.cpp` till `neural_network` via `g++ -Wall -Werror -std=c++17 -Iinclude`. Targets: `build`, `run` och `clean`, med `build run` som default. |

Katalogen [test](./test/) innehåller lektionens testsvit, som beskrivs i sin
[egen README](./test/README.md). Testsviten innehåller även testerna från **L06–L07**.

---

## Bygg och kör
Bygg och kör programmet via följande kommando i denna katalog:

```bash
make
```

Programmet skriver ut följande, vilket är exakt den utskrift som avsnitt 10 i bilaga B anger:

```
--------------------------------------------------------------------------------
Predictions before training:
Input: 0 0, predicted output: 0
Input: 0 1, predicted output: 0
Input: 1 0, predicted output: 0
Input: 1 1, predicted output: 0
--------------------------------------------------------------------------------
Predictions after training:
Input: 0 0, predicted output: 0
Input: 0 1, predicted output: 0
Input: 1 0, predicted output: 0
Input: 1 1, predicted output: 0
--------------------------------------------------------------------------------
```

Tre saker i utskriften är värda att notera:
* **Samtliga prediktioner är 0.** `feedforward()` kontrollerar enbart sin indata, så utdatan
  behåller startvärdet `0.0`. Vikterna och biasvärdena är dessutom noll tills de randomiseras
  under **L09**.
* **Prediktionerna är desamma före och efter träningen.** Träningen genomförs, men eftersom
  varken backpropagation eller optimering räknar ännu finns det ingenting som kan ändra dem.
* **Utskriften saknar felmeddelanden och `Training failed!`.** Samtliga argumentkontroller
  godkände alltså samtliga anrop under träningens 400 steg, vilket visar att lagren är rätt
  kopplade till varandra och till träningsdatan: 2 insignaler in i det dolda lagrets 3 noder, och
  vidare in i utgångslagrets enda nod.

Övriga targets i [makefilen](./Makefile):

```bash
make build  # Bygger programmet utan att köra det.
make run    # Kör programmet utan att bygga om det.
make clean  # Tar bort den byggda binären.
```

## Kör testerna

```bash
make -C test
```

Samtliga 41 testfall går igenom: 15 för stubben från **L06**, 15 för nätverket från **L07** samt
11 för `Dense`.

---

## Noteringar om implementationen

### Aktiveringsfunktionen är ett värde, inte en funktion
`ActFunc` ligger i [types.h](./include/ml/types.h), bredvid matrisaliasen, eftersom den hör till
kodbasens gemensamma typer: både lagret och den som skapar lagret använder den. Lagret sparar det
givna värdet i `myActFunc` och gör ingenting mer med det under denna lektion. Under **L09** blir
värdet det som hjälpfunktionerna `actFuncOutput()` och `actFuncDelta()` väljer beräkning utifrån.

Tre enumeratorer finns:
* `Relu` returnerar indatan om den är positiv, annars noll.
* `Tanh` ger utdata i intervallet (-1, 1).
* `None` är ingen aktivering alls, dvs. identitet: utdatan är den viktade summan oförändrad, och
  derivatan är `1.0`.

`myActFunc` är `const`, så ett lagers aktiveringsfunktion kan inte bytas efter konstruktionen. Ett
lager som byter aktiveringsfunktion mitt i träningen skulle ha vikter som är upplärda för en annan
funktion, och den `const`-markerade medlemsvariabeln gör det omöjligt i stället för olämpligt.

### Medlemsvariablerna och vad de ska användas till
Utöver stubbens tre vektorer har `Dense` en biasvektor samt en vektor för den viktade summan:

| Medlemsvariabel | Storlek | Roll |
|---|---|---|
| `myWeights` | `nodeCount` x `weightCount` | En vikt per nod och insignal. Randomiseras under **L09**. |
| `myOutput` | `nodeCount` | Nodernas utdata, efter aktiveringsfunktionen. |
| `myPreActOutput` | `nodeCount` | Nodernas viktade summa, innan aktiveringsfunktionen. |
| `myBias` | `nodeCount` | En bias per nod. Randomiseras under **L09**. |
| `myError` | `nodeCount` | Nodernas beräknade fel. |
| `myActFunc` | - | Lagrets aktiveringsfunktion. |

Ingenting skriver till `myPreActOutput` ännu, men den finns med redan nu eftersom `backpropagate()`
under **L09** behöver aktiveringsfunktionens derivata beräknad utifrån den viktade summan, inte
utifrån utdatan. Utdatan går inte att räkna tillbaka till summan för samtliga aktiveringsfunktioner,
så värdet måste sparas när det beräknas.

### Konstruktorn kontrollerar innan den allokerar
Medlemsinitieringslistan lämnar samtliga fem vektorer tomma. Antalet noder och antalet vikter
kontrolleras först, med var sitt felmeddelande till `stderr` följt av `std::terminate()`, och först
när båda kontrollerna har passerats tilldelas vektorerna sina storlekar. Ett lager som skapas med
ett antal satt till noll avslutar därmed programmet utan att ha allokerat någonting.

Det är samma upplägg, och samma två felmeddelanden, som i stubben från **L06**. Lagret har inte
heller några egna medlemsvariabler för antalen: `nodeCount()` returnerar utdatavektorns storlek och
`weightCount()` storleken på viktmatrisens första rad. Att läsa rad 0 är säkert just tack vare
nollkontrollerna, eftersom lagret garanterat har minst en nod.

Vektorerna fylls med `0.0`, biasvektorn och vikterna inkluderade. Randomiseringen hör till **L09**,
där hjälpfunktionerna för slumptal skapas.

### Beräkningsmetoderna kontrollerar enbart sina argument
Var och en av de fyra metoderna jämför en storlek med ett antal, och skriver vid fel ut ett
felmeddelande till `stderr` innan den returnerar `false`:

| Metod | Godkänner då | Felmeddelande vid fel |
|---|---|---|
| `feedforward(input)` | `input.size() == weightCount()` | `Input dimension mismatch: expected X, actual: Y!` |
| `backpropagate(reference)` | `reference.size() == nodeCount()` | `Output dimension mismatch: expected X, actual: Y!` |
| `backpropagate(nextLayer)` | `nextLayer.weightCount() == nodeCount()` | `Layer dimension mismatch: expected X, actual: Y!` |
| `optimize(input, learningRate)` | lärhastigheten ligger inom `(0.0, 1.0)`, samt `input.size() == weightCount()` | `Invalid learning rate X!` respektive `Input dimension mismatch: expected X, actual: Y!` |

Villkoren är desamma som stubbens, med undantag för stubbens feedforward-räknare, som enbart fanns
för att göra nätverket testbart. Felmeddelandena är däremot nya: stubben avvisar tyst, medan
`Dense` säger vad som gick fel, och det är samma meddelanden som lagret behåller när beräkningarna
läggs till under **L09**. Kontrollen av lärhastigheten formuleras som ett positivt villkor,
`(0.0 < learningRate) && (1.0 > learningRate)`, vilket också avvisar NaN; samma resonemang som för
`train()` i [lösningsförslaget från **L07**](../../L07/exercises/README.md).

Testsviten kontrollerar samtliga fyra metoder redan nu, i testfallen `FeedforwardChecksInputSize`,
`BackpropagateChecksReferenceSize`, `BackpropagateChecksNextLayerWeightCount`,
`OptimizeChecksInputSize` samt `OptimizeChecksLearningRate`. Under **L09** behålls kontrollerna
oförändrade, och beräkningarna läggs till efter dem.

### `Dense` ersätter inte stubben
Båda klasserna finns kvar i kodbasen, och båda uppfyller `dense_layer::Interface`. Stubben används
fortfarande av nätverkstesterna från **L07**, som behöver lager med förutsägbar utdata och en
räknare som visar hur ofta nätverket matade respektive lager. Ett `Dense`-lager kan inte ge dem
det, i synnerhet inte från och med **L09**, när dess startvärden slumpas.

### Testprogrammet bygger nätverket av `Dense`-lager
Skillnaden mot **L07** är två rader i [main.cpp](./source/main.cpp):

```cpp
ml::dense_layer::Dense hiddenLayer{hiddenCount, inputCount};
ml::dense_layer::Dense outputLayer{outputCount, hiddenCount, ml::ActFunc::Tanh};
```

Det dolda lagret använder defaultvärdet `ActFunc::Relu`, medan utgångslagret använder
`ActFunc::Tanh`, som kan ge både positiva och negativa utsignaler och därmed röra sig åt båda håll
under träningen. `Shallow` behövde inte ändras en enda rad, vilket är hela poängen med att
nätverket håller sina lager via `dense_layer::Interface&`.

---

## Nästa steg
* **L09:** Randomisering av bias och vikter, samt beräkningarna i `feedforward()`,
  `backpropagate()` och `optimize()`. Först då ändrar träningen prediktionerna, och först då kan
  nätverket lära sig XOR-mönstret.

---

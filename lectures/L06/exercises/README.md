# L06 - Lösningsförslag: Enkelt neuralt nätverk i C++ (del I)
Lösningsförslag till övningsuppgiften i [bilaga B](../appendix/b_exercises.md).
Katalogen innehåller den del av `ml`-kodbasen som tillkommer under denna lektion:
* Ett interface för dense-lager.
* En stubbklass som implementerar interfacet utan att beräkna någonting.
* Ett litet program som kör stubbens metoder och skriver ut resultatet.

---

## Vad L06 bygger
**En ny del av `ml`-kodbasen.** Den linjära regressionsmodellen finns kvar i
[lösningsförslaget från **L04**](../../L04/exercises/README.md); denna katalog innehåller enbart
det som tillkommer. Det som följer med är kodens form snarare än själva koden:
* Ett interface i en egen namnrymd.
* En underklass markerad `final`.
* `std::terminate()` reserverad för konstruktorn.

Endast [types.h](./include/ml/types.h) följer med i stort sett oförändrad.

**Ett interface för dense-lager.** `ml::dense_layer::Interface` är det kontrakt som varje
dense-lager i kursen uppfyller: fem getters samt fyra beräkningsmetoder som rapporterar fel via
sitt returvärde. Samtliga metoder förutom destruktorn är rent virtuella, så interfacet anger vad
ett lager ska kunna göra men ingenting om hur.

**En stubbe i stället för ett skarpt lager.** `ml::dense_layer::Stub` uppfyller kontraktet med
range-checkar och ingenting annat. Den finns för att nätverket i **L07** ska gå att skriva, bygga
och testa mot ett riktigt `dense_layer::Interface` innan det skarpa `Dense`-lagret skapas under
**L08–L09**, och den har tre extra metoder som enbart finns för att göra nätverket testbart.

---

## Filer

| Fil | Innehåll |
|---|---|
| [include/ml/types.h](./include/ml/types.h) | Samma matrisalias som i **L04**: `Matrix1d`, `Matrix2d` och `Matrix3d` av `double`, samt `MatrixU32`. Endast `Matrix1d` och `Matrix2d` används under denna lektion; övriga alias finns för kod som byggs senare i kursen. |
| [include/ml/dense_layer/interface.h](./include/ml/dense_layer/interface.h) | `ml::dense_layer::Interface`: en virtuell destruktor, getters `output()`, `error()`, `weights()`, `nodeCount()` och `weightCount()`, samtliga `[[nodiscard]]`, `const` och `noexcept`, samt beräkningsmetoderna `feedforward()`, båda varianterna av `backpropagate()` och `optimize()`, samtliga med returtypen `bool`. Samtliga metoder förutom destruktorn är rent virtuella. |
| [include/ml/dense_layer/stub.h](./include/ml/dense_layer/stub.h) | `ml::dense_layer::Stub`, implementerad direkt i headerfilen: konstruktorn med sina två nollkontroller, samtliga överlagringar från interfacet, de stubbspecifika metoderna `setOutput()`, `feedforwardCount()` och `clearFeedforwardCount()`, raderade default-, copy- och move-operationer, samt fyra medlemsvariabler. |
| [source/main.cpp](./source/main.cpp) | Demo enligt avsnitt 4 i bilaga B: skapar ett dolt lager med 3 noder och 2 vikter per nod på defaultvärdet `0.5`, samt ett utgångslager med 1 nod och 3 vikter per nod med utdatan `0.8`, och kör sedan feedforward, backpropagation, optimering, `setOutput()` och `clearFeedforwardCount()` på lagren, med samtliga returvärden utskrivna. |
| [Makefile](./Makefile) | Bygger `source/main.cpp` till `dense_layer` via `g++ -Wall -Werror -std=c++17 -Iinclude`. Targets: `build`, `run` och `clean`, med `build run` som default. |

Katalogen [test](./test/) innehåller lektionens testsvit, som beskrivs i sin
[egen README](./test/README.md).

---

## Bygg och kör
Bygg och kör programmet via följande kommando i denna katalog:

```bash
make
```

Programmet skriver ut följande, vilket är exakt den utskrift som avsnitt 4 i bilaga B anger:

```
--------------------------------------------------------------------------------
Hidden layer: node count: 3, weight count: 2, output: 0.5 0.5 0.5
Output layer: node count: 1, weight count: 3, output: 0.8
--------------------------------------------------------------------------------
Feedforward with 2 inputs: true
Feedforward with 3 inputs: false
Output after feedforward: 0.5 0.5 0.5
Feedforward count: 2
--------------------------------------------------------------------------------
Backpropagate output layer: true
Backpropagate hidden layer: true
Optimize with learning rate 0.01: true
Optimize with learning rate 1.5: false
--------------------------------------------------------------------------------
Output after setOutput(0.1): 0.1 0.1 0.1
Feedforward count after clearFeedforwardCount(): 0
--------------------------------------------------------------------------------
```

Tre saker i utskriften är värda att notera:
* **Utdatan är fortfarande `0.5` efter feedforward.** Stubben beräknar ingenting, utan rapporterar
  det värde den skapades med. Att de två lagren har olika värden är det som låter ett nätverk byggt
  av dem visa vilket lager en prediktion kom från.
* **Feedforward-räknaren är `2`, trots att det andra anropet avvisades.** Räknaren mäter hur ofta
  lagret ombads göra feedforward, inte hur ofta det gick med på det.
* **`hiddenLayer.backpropagate(outputLayer)` lyckas.** Utgångslagrets 3 vikter per nod matchar det
  dolda lagrets 3 noder, vilket är samma koppling som nätverket i **L07** bygger på. Formerna är
  alltså inte godtyckliga.

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

Samtliga 15 testfall går igenom.

---

## Noteringar om implementationen

### Stubben är implementerad direkt i headerfilen
Samtliga metoder i `Stub` definieras inuti klasskroppen i
[stub.h](./include/ml/dense_layer/stub.h), så det finns ingen `stub.cpp`, och `main.cpp` är den
enda fil som [makefilen](./Makefile) kompilerar. Testsvitens lista `SOURCE_FILES` är tom av samma
anledning. Båda listorna växer under **L07**, när nätverket tillför de första `.cpp`-filerna.

En så liten klass vinner ingenting på en separat källkodsfil, och när samtliga metoder syns på ett
och samma ställe är det lätt att se att ingen av dem beräknar någonting.

### Antalen läses ur vektorerna
Stubben har varken `myNodeCount` eller `myWeightCount`. `nodeCount()` returnerar utdatavektorns
storlek, och `weightCount()` returnerar storleken på viktmatrisens första rad. Att även lagra
antalen skulle ge vart och ett av dem två sanningskällor, samma resonemang som gör att
`myTrainOrder.size()` anger antalet träningsuppsättningar i den linjära regressionsmodellen.

Att läsa rad 0 är bara säkert eftersom lagret garanterat har minst en nod, vilket är
nollkontrollernas andra uppgift i konstruktorn: en stubbe utan noder skulle inte ha någon rad att
läsa.

### Konstruktorn kontrollerar innan den allokerar
Medlemsinitieringslistan lämnar samtliga tre vektorer tomma. Antalet noder och antalet vikter
kontrolleras först, med var sitt felmeddelande, och först när båda kontrollerna har passerats
tilldelas vektorerna sina storlekar: utdatan fylls med `outputValue`, felet och vikterna med
`0.0`. En stubbe som skapas med ett antal satt till noll avslutar därmed programmet utan att ha
allokerat någonting.

Någon biasvektor finns inte alls. Bias ingår inte i `Interface`, och ett lager som aldrig optimerar
har ingenting att lägga i den. Det är en förenkling som enbart gäller stubben; det skarpa
`Dense`-lagret i **L08–L09** behöver en biasvektor lika mycket som det behöver sina vikter.

### Range-checkarna är stubbens enda logik
Varje beräkningsmetod jämför en storlek med ett antal och returnerar resultatet:

| Metod | Godkänner då |
|---|---|
| `feedforward(input)` | `input.size() == weightCount()` |
| `backpropagate(reference)` | `reference.size() == nodeCount()` |
| `backpropagate(nextLayer)` | `nextLayer.weightCount() == nodeCount()` |
| `optimize(input, learningRate)` | lärhastigheten ligger inom `(0.0, 1.0)`, samt `input.size() == weightCount()` |

Ingenting annat händer. Utdatan förblir `outputValue`, och felet samt vikterna förblir noll, oavsett
om anropet godkänns eller avvisas. Kontrollerna är ändå viktiga, eftersom nätverket i **L07** testas
mot ingenting annat än denna stubbe: en `feedforward()` som godkände indata av vilken storlek som
helst skulle låta ett nätverk som skickar en vektor till fel lager klara samtliga sina tester.

### Lärhastigheten kontrolleras som "inom", inte som "inte utanför"
`optimize()` formulerar kontrollen som ett positivt villkor:

```cpp
const bool lrValid{(0.0 < learningRate) && (1.0 > learningRate)};
return lrValid && (input.size() == weightCount());
```

För varje vanligt tal är det likvärdigt med att avvisa lärhastigheten då
`(0.0 >= learningRate) || (1.0 <= learningRate)`. De två formerna skiljer sig åt för NaN: varje
jämförelse som involverar NaN är falsk, så NaN klarar inte `lrValid` och avvisas här, medan den
omvända formen inte skulle hitta någon överskriden gräns och därmed släppa igenom värdet.
Testsviten matar enbart `optimize()` med vanliga tal, så den kan inte skilja de två formerna åt.

### `feedforward()` räknar innan den kontrollerar
`++myFeedforwardCount` är den första satsen i `feedforward()`, före storlekskontrollen, så ett
avvisat anrop höjer räknaren precis som ett godkänt. Det är just den frågan testerna i **L07**
behöver få besvarad: en träningsloop som kör ett enda pass i stället för ett per epok stämmer
fortfarande dimensionsmässigt och returnerar fortfarande `true`, och räknaren är det enda som
avslöjar den.

### Testhjälpmedlen ingår inte i interfacet
`setOutput()`, `feedforwardCount()` och `clearFeedforwardCount()` finns enbart på `Stub`. Kod som
håller ett lager via en `dense_layer::Interface&`, som nätverket i **L07** gör, kan inte anropa dem,
utan det kan bara ett test som håller den konkreta `Stub`-instansen. Det är en avsiktlig
uppdelning: nätverket kan inte börja bero på ett testhjälpmedel, och testerna får två saker som
interfacet ensamt inte kan ge dem. `setOutput()` ändrar ett lagers utdata bakom nätverkets rygg,
vilket visar om nätverket läser sitt utgångslager direkt eller har sparat en egen kopia, och
feedforward-räknaren visar hur ofta nätverket anropade respektive lager.

### Överlagringarna behåller `[[nodiscard]]`
Samtliga getters samt `feedforwardCount()` är markerade `[[nodiscard]]` i stubben lika väl som i
interfacet. Attribut ärvs inte, så utan det skulle ett bortkastat anrop på den konkreta typen, som
`stub.nodeCount();`, kompilera utan varning. Med det avvisar `g++` raden under `-Werror`
("ignoring return value of ... declared with attribute 'nodiscard'"), oavsett om anropet görs på
en `Stub` eller via en `Interface&`. Beräkningsmetoderna har inte attributet i interfacet, och deras
överlagringar lägger inte till det.

### `main.cpp` använder både interfacet och den konkreta typen
`printLayerInfo()` tar emot en `const dense_layer::Interface&` och vet därmed inte att den fick en
stubbe; samma funktion fungerar oförändrad för `Dense`-lagret i **L08–L09**. Resten av `main()`
anropar metoderna direkt på de två `Stub`-instanserna, eftersom `setOutput()`,
`feedforwardCount()` och `clearFeedforwardCount()` inte går att nå via interfacet.

Returvärdena skrivs ut via `std::cout` med `std::boolalpha`, som får `bool` att skrivas ut som
`true`/`false` i stället för `1`/`0`. Det är också anledningen till att programmet använder
`<iostream>` i stället för `std::printf()`, som saknar ett formatspecifikationstecken för `bool`.
Samtliga värden som skickas till lagren, som lärhastigheterna `0.01` och `1.5`, är namngivna
konstanter överst i `main()`, och antalet insignaler i utskriften hämtas från själva vektorerna i
stället för att skrivas in som text.

---

## Nästa steg
* **L07:** `ml::neural_network::Shallow`, ett neuralt nätverk vars dolda lager och utgångslager
  hålls via `dense_layer::Interface&` och testas mot två stubbar.
* **L08–L09:** `ml::dense_layer::Dense`, en skarp implementation av samma interface som ersätter
  stubbarna i nätverket, utan att nätverket behöver ändras.

---

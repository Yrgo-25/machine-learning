# L07 - Lösningsförslag: Enkelt neuralt nätverk i C++ (del II)
Lösningsförslag till övningsuppgiften i [bilaga B](../appendix/b_exercises.md).
Katalogen innehåller den del av `ml`-kodbasen som tillkommer under denna lektion:
* Ett interface för neurala nätverk.
* Klassen `Shallow`, ett nätverk med ett dolt lager och ett utgångslager som det håller via
  dense-lagrets interface.
* Ett testprogram som tränar nätverket på ett XOR-mönster, med två stubbar som lager.

---

## Vad L07 bygger
**En utbyggnad av L06.** [types.h](./include/ml/types.h) samt dense-lagrets interface och stubbe
följer med oförändrade från [lösningsförslaget från **L06**](../../L06/exercises/README.md), där de
beskrivs. Det som tillkommer är namnrymden `ml::neural_network`, med ett interface i en headerfil
och en implementation fördelad på en header- och en källkodsfil, samt ett nytt testprogram i
`main.cpp`.

**Ett interface för neurala nätverk.** `ml::neural_network::Interface` har en enda metod förutom
destruktorn: den rent virtuella `predict()`. Kod som enbart använder ett nätverk behöver varken
veta hur det är uppbyggt eller hur det tränas.

**Ett nätverk som enbart kopplar ihop lager.** `ml::neural_network::Shallow` beräknar ingenting
själv. Varje steg i feedforward, backpropagation och optimering delegeras till de två lagren, som
nätverket enbart känner till som `dense_layer::Interface&`. Nätverkets uppgift är ordningen:
* Vilket lager som anropas när.
* Vilken indata respektive lager får.
* Att träningen avbryts så snart ett lageranrop misslyckas.

Det är därför nätverket går att testa fullt ut mot två stubbar, och därför det inte behöver ändras
när det skarpa `Dense`-lagret ersätter stubbarna under **L08–L09**.

---

## Filer

| Fil | Innehåll |
|---|---|
| [include/ml/types.h](./include/ml/types.h) | Oförändrad från **L06**: matrisaliasen `Matrix1d`, `Matrix2d`, `Matrix3d` och `MatrixU32`. Endast `Matrix1d` och `Matrix2d` används under denna lektion. |
| [include/ml/dense_layer/interface.h](./include/ml/dense_layer/interface.h) | Oförändrad från **L06**: `ml::dense_layer::Interface`, det kontrakt som nätverkets båda lager uppfyller. |
| [include/ml/dense_layer/stub.h](./include/ml/dense_layer/stub.h) | Oförändrad från **L06**: `ml::dense_layer::Stub`, som utgör båda lagren i testprogrammet och i testsviten. |
| [include/ml/neural_network/interface.h](./include/ml/neural_network/interface.h) | `ml::neural_network::Interface`: en virtuell destruktor samt den rent virtuella `predict()`, markerad `[[nodiscard]]` och `noexcept`, som returnerar en `const Matrix1d&`. |
| [include/ml/neural_network/shallow.h](./include/ml/neural_network/shallow.h) | Deklarationen av `ml::neural_network::Shallow`, markerad `final`: en `explicit` konstruktor, `predict()`, `train()` med lärhastigheten `0.01` som defaultvärde, raderade default-, copy- och move-operationer, tre privata hjälpmetoder samt fem medlemsvariabler. Dense-lagrets interface forward-deklareras i stället för att inkluderas. |
| [source/ml/neural_network/shallow.cpp](./source/ml/neural_network/shallow.cpp) | Implementationen av `Shallow`: konstruktorns två kontroller, `predict()`, träningsloopen i `train()`, samt hjälpmetoderna `feedforward()`, `backpropagate()` och `optimize()`, som vardera anropar båda lagren i rätt ordning. |
| [source/main.cpp](./source/main.cpp) | Testprogrammet enligt avsnitt 5 i bilaga B: skapar ett dolt lager med 3 noder och 2 vikter per nod samt ett utgångslager med 1 nod och 3 vikter per nod, kopplar ihop dem till ett nätverk med ett 2-bitars XOR-mönster som träningsdata, och predikterar före och efter 100 epokers träning med lärhastigheten `0.01`. Returnerar `-1` om träningen misslyckas. |
| [Makefile](./Makefile) | Bygger `source/main.cpp` och `source/ml/neural_network/shallow.cpp` till `neural_network` via `g++ -Wall -Werror -std=c++17 -Iinclude`. Targets: `build`, `run` och `clean`, med `build run` som default. |

Katalogen [test](./test/) innehåller lektionens testsvit, som beskrivs i sin
[egen README](./test/README.md). Testsviten innehåller även stubbtesterna från **L06**.

---

## Bygg och kör
Bygg och kör programmet via följande kommando i denna katalog:

```bash
make
```

Programmet skriver ut följande, vilket är exakt den utskrift som avsnitt 7 i bilaga B anger:

```
--------------------------------------------------------------------------------
Predictions before training:
Input: 0 0, predicted output: 0.5
Input: 0 1, predicted output: 0.5
Input: 1 0, predicted output: 0.5
Input: 1 1, predicted output: 0.5
--------------------------------------------------------------------------------
Predictions after training:
Input: 0 0, predicted output: 0.5
Input: 0 1, predicted output: 0.5
Input: 1 0, predicted output: 0.5
Input: 1 1, predicted output: 0.5
--------------------------------------------------------------------------------
```

Tre saker i utskriften är värda att notera:
* **Prediktionerna är desamma före och efter träningen.** Stubbarna skapas med defaultvärdet `0.5`
  och beräknar ingenting, så varje prediktion är utgångslagrets oförändrade utdata.
* **Utskriften saknar `Training failed!`.** `train()` returnerade alltså `true`, vilket betyder att
  samtliga 400 träningssteg (100 epoker med 4 träningsuppsättningar vardera) genomfördes utan att
  något lageranrop avvisades. Att träningsloopen verkligen matade lagren en gång per steg går inte
  att se här, utan kontrolleras av testsviten via stubbarnas feedforward-räknare.
* **Varje prediktion består av ett enda värde.** Båda lagren rapporterar `0.5`, så värdet i sig
  avslöjar inte vilket lager prediktionen kommer från, men antalet gör det: det dolda lagrets utdata
  har tre värden. Testsviten går längre och ger lagren olika utdata, `0.25` respektive `0.75`.

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

Samtliga 30 testfall går igenom: 15 för stubben från **L06** och 15 för nätverket.

---

## Noteringar om implementationen

### Interfacet innehåller enbart `predict()`
`train()` finns enbart i `Shallow`. Träning kräver träningsdata, ett antal epoker och en
lärhastighet, vilket är detaljer som hör till hur ett visst nätverk tränas; den som enbart använder
ett tränat nätverk behöver bara kunna predicera. [main.cpp](./source/main.cpp) visar uppdelningen:
* `printPredictions()` tar emot en `neural_network::Interface&` och vet därmed inte vilket slags
  nätverk den fick.
* `trainAndTest()` tar emot en `Shallow&`, eftersom den anropar `train()`.

`predict()` är inte `const`, eftersom varje prediktion uppdaterar lagrens utdata, så även
`printPredictions()` tar emot interfacet som en icke-konstant referens. Utöver det bilaga B kräver
är `predict()` markerad `[[nodiscard]]`, både i interfacet och i `Shallow`: ett bortkastat anrop som
`network.predict(input);` avvisas under `-Werror`, eftersom en prediktion som ingen läser nästan
alltid är ett misstag.

### Dense-lagrets interface forward-deklareras
[shallow.h](./include/ml/neural_network/shallow.h) inkluderar `ml/neural_network/interface.h`,
eftersom `Shallow` ärver den klassen och kompilatorn då behöver hela definitionen. Dense-lagrets
interface forward-deklareras däremot, eftersom headerfilen enbart använder det i form av referenser:

```cpp
// clang-format off
namespace ml::dense_layer { class Interface; }
// clang-format on
```

Kommentarerna hindrar `clang-format` från att bryta upp deklarationen på flera rader.
[shallow.cpp](./source/ml/neural_network/shallow.cpp) anropar lagrens metoder och inkluderar därför
`ml/dense_layer/interface.h` fullt ut. Den som inkluderar `shallow.h` drar alltså inte med sig
dense-lagrets interface, och en ändring i det tvingar inte fram en omkompilering av allt som
använder nätverket.

`main.cpp` inkluderar för övrigt inte `ml/neural_network/interface.h` separat, trots att
`printPredictions()` använder klassen. Den följer med via `shallow.h`, och det är ingen
tillfällighet: `shallow.h` måste inkludera den för att `Shallow` ska kunna ärva klassen.

### Konstruktorns två kontroller
Medlemsinitieringslistan initierar samtliga fem medlemsvariabler, och `myTrainSetCount` sätts till
`std::min(trainInput.size(), trainOutput.size())`. Har den ena matrisen fler rader än den andra
ignoreras de överblivna raderna, i stället för att `train()` läser utanför den kortare matrisen.

Därefter genomförs två kontroller, var och en med ett eget felmeddelande till `stderr` följt av
`std::terminate()`:
1. `myTrainSetCount` är noll, dvs. det finns inte en enda fullständig träningsuppsättning. Utskrift:
   `Cannot train network without training sets!`
2. Det dolda lagrets antal noder skiljer sig från utgångslagrets antal vikter per nod, dvs. lagren
   är felkopplade. Utskrift: `Dimension mismatch between the dense layers!`

Ingen av kontrollerna testas av testsviten, eftersom `std::terminate()` avslutar hela
testprogrammet. Båda går att prova för hand i `main.cpp`: töm `trainOut`, eller skapa
utgångslagret med 2 vikter per nod i stället för 3.

Konstruktorn kontrollerar däremot inte att träningsdatans rader har rätt bredd, dvs. att varje
indatarad har en insignal per vikt i det dolda lagret och varje utdatarad ett värde per nod i
utgångslagret. Den kontrollen görs redan av lagren själva, och en felaktig bredd leder till att
`train()` returnerar `false` vid det första lageranropet som avvisas, se nedan.

### Referenser som medlemsvariabler
Nätverket håller lagren och träningsdatan via referenser och kopierar ingenting. Det får tre
följder:
* **Livslängden.** Lagren och träningsdatan måste leva minst lika länge som nätverket. I `main()`
  deklareras de före nätverket, så de förstörs efter det.
* **Temporära objekt.** Satsen `Shallow network{hidden, output, {{0.0, 0.0}}, {{0.0}}};`
  kompilerar utan varning, även med `-Wextra`, men de två temporära matriserna förstörs i slutet av
  satsen.
  Nätverket står då kvar med referenser till minne som inte längre finns, och ett senare anrop till
  `train()` har odefinierat beteende. Träningsdatan måste därför alltid vara namngivna variabler.
* **Delat tillstånd.** Ändras ett lager via en annan väg syns ändringen direkt i nätverket. Testet
  `PredictReadsOutputLayerLive` utnyttjar detta genom att anropa `setOutput()` på utgångslagret
  bakom nätverkets rygg.

Default-konstruktorn samt copy- och move-operationerna är raderade. En kopia av nätverket skulle
ändå dela lager med originalet, så två "olika" nätverk skulle i själva verket träna samma vikter.

### `predict()` returnerar utgångslagrets egen vektor
`predict()` returnerar `myOutputLayer.output()` direkt, utan att spara någon egen kopia i
`Shallow`. Testet `PredictReturnsReferenceToOutputLayer` kontrollerar det genom att jämföra
adresserna. En egen kopia hade kostat en kopiering per prediktion, och hade dessutom blivit
inaktuell så snart lagret ändrats på annat sätt.

Priset är att referensen enbart gäller tills nästa anrop som ändrar utgångslagrets utdata, dvs.
nästa prediktion eller träningssteg. I `main.cpp` skrivs varje prediktion ut direkt; den som vill
spara en prediktion får kopiera den själv, t.ex. via
`const ml::Matrix1d saved{network.predict(input)};`.

### `train()` anropar inte `predict()`
Feedforward-steget i träningsloopen görs av den privata hjälpmetoden `feedforward()`, inte av
`predict()`. Båda gör samma två lageranrop, men enbart `feedforward()` returnerar `false` så snart
något av dem misslyckas.

`predict()` kan nämligen inte rapportera fel. Den returnerar en `const Matrix1d&`, så
returvärdena från lagrens `feedforward()` kastas bort. Att det kompilerar under `-Werror` beror på
att beräkningsmetoderna i `dense_layer::Interface` saknar `[[nodiscard]]`. Ett `predict()`-anrop
med fel antal insignaler avvisas av det dolda lagret utan att någon får veta det: utgångslagrets
utdata returneras ändå, och den bygger då inte på den givna indatan. För en prediktion får
anroparen stå för att indatan har rätt storlek; under träningen kan nätverket däremot själv
upptäcka felet och avbryta.

Ett nätverk som ändå anropar `predict()` i träningsloopen klarar testsviten, men upptäcker felet
senare. I det första fallet av `TrainStopsAtFirstFailedLayerCall`, med för breda indatarader,
stoppar lösningsförslaget redan vid det dolda lagrets `feedforward()`, medan en variant som
anropar `predict()` hinner genomföra backpropagation innan det dolda lagrets `optimize()` avvisar
indatan. I båda fallen har det dolda lagret matats exakt en gång, vilket är det testet
kontrollerar.

### En hjälpmetod per steg i träningen
Den inre loopen i `train()` består av tre rader, en per steg i
[bilaga A](../appendix/a_training_loop.md):

```cpp
if (!feedforward(input)) { return false; }
if (!backpropagate(output)) { return false; }
if (!optimize(input, learningRate)) { return false; }
```

`input` och `output` är referenser till den aktuella träningsuppsättningens rader, så ingen rad
kopieras. Varje hjälpmetod anropar båda lagren i den ordning steget kräver:

| Hjälpmetod | Första anropet | Andra anropet |
|---|---|---|
| `feedforward(input)` | `myHiddenLayer.feedforward(input)` | `myOutputLayer.feedforward(myHiddenLayer.output())` |
| `backpropagate(output)` | `myOutputLayer.backpropagate(output)` | `myHiddenLayer.backpropagate(myOutputLayer)` |
| `optimize(input, learningRate)` | `myHiddenLayer.optimize(input, learningRate)` | `myOutputLayer.optimize(myHiddenLayer.output(), learningRate)` |

Ordningen är inte godtycklig:
* **Backpropagation går baklänges.** Utgångslagrets fel beräknas utifrån referensvärdena, och det
  dolda lagrets fel utifrån utgångslagrets fel och vikter. Utgångslagret måste alltså vara klart
  först.
* **Utgångslagret optimeras med samma indata som det matades med.** `optimize()` ändrar lagrets
  vikter men inte dess utdata, så `myHiddenLayer.output()` är samma vektor i feedforward och i
  optimeringen.
* **Det andra anropet görs aldrig efter ett misslyckat första.** Varje hjälpmetod returnerar
  `false` direkt om det första lageranropet misslyckas, och `train()` returnerar i sin tur vid det
  första `false`. Träningen avbryts därmed mitt i epoken, vilket testet
  `TrainStopsAtFirstFailedLayerCall` kontrollerar via det dolda lagrets feedforward-räknare.

Hjälpmetoderna är privata. Bilaga B tillåter extra privata metoder, och de hör inte hemma i
interfacet: den som använder nätverket ska inte kunna genomföra ett halvt träningssteg.

### Argumenten kontrolleras innan träningen börjar
`train()` returnerar `false` för ett antal epoker satt till noll och för en lärhastighet utanför
`(0.0, 1.0)`, innan loopen startar. Ett avvisat anrop rör därmed inte lagren, vilket testerna
`TrainRejectsZeroEpochCount` och `TrainRejectsInvalidLearningRate` kontrollerar genom att
feedforward-räknaren fortfarande är noll. Utan kontrollen av lärhastigheten skulle stubbens egen
range-check i `optimize()` visserligen ändå få `train()` att returnera `false`, men först efter att
den första träningsuppsättningen matats genom nätverket, och med `Dense` efter att lagrens utdata
och fel redan skrivits över.

Kontrollen av lärhastigheten formuleras, som avsnitt 6 i bilaga B anger, som ett positivt villkor
på samma sätt som i stubbens `optimize()`:

```cpp
const auto lrValid = ((0.0 < learningRate) && (1.0 > learningRate));
if (!lrValid) { return false; }
```

Den omvända formen, `(0.0 >= learningRate) || (1.0 <= learningRate)`, är likvärdig för vanliga
tal, men som [lösningsförslaget från **L06**](../../L06/exercises/README.md) beskriver skiljer
formerna sig åt för NaN. Varje jämförelse som involverar NaN är falsk, så NaN klarar inte `lrValid`
och avvisas direkt, medan den omvända formen skulle släppa igenom värdet. Med stubbarna hade NaN
då ändå avvisats, men sent: det dolda lagrets `optimize()` hade returnerat `false` under den
första träningsuppsättningen, efter en feedforward per lager. Testsviten matar enbart `train()`
med vanliga tal och kan därför inte skilja formerna åt.

---

## Nästa steg
* **L08:** `ml::dense_layer::Dense`, en skarp implementation av dense-lagrets interface med ett
  fåtal aktiveringsfunktioner. Samtliga metoder förutom `feedforward()`, `backpropagate()` och
  `optimize()` implementeras.
* **L09:** De tre återstående beräkningsmetoderna implementeras, och `Dense` ersätter stubbarna i
  nätverket utan att `Shallow` behöver ändras. Först då påverkar träningen prediktionerna.

---

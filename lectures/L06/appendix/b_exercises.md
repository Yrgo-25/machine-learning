# Bilaga B - Övningsuppgift: Enkelt neuralt nätverk i C++ (del I)
Ni ska bygga vidare på er `ml`-kodbas med ett interface för dense-lager samt en stubbklass som 
implementerar detta interface. Nätverket som använder lagren byggs under **L07**.

---

### 1. Kom igång
Hämta först testramverket, som ligger i git-submodulen `libs/test`. Kör följande kommando en gång,
i repots rotkatalog:

```bash
git submodule update --init --recursive
```

Katalogen [`exercises`](../exercises) innehåller en färdig katalogstruktur med tomma headerfiler
för interfacet och stubbklassen, samt en testsvit i `exercises/test` (se avsnitt 5). Skriv er kod
där, eller bygg ut katalogstrukturen i er befintliga `ml`-kodbas enligt nedan:

```
ml/
├── include/
│   └── ml/
│       ├── dense_layer/
│       │   ├── interface.h
│       │   └── stub.h
│       └── types.h
├── source/
│   └── main.cpp
└── Makefile
```

Både interfacet och stubbklassen implementeras direkt i sina headerfiler, så makefilen behöver
inte uppdateras under denna lektion.

---

### 2. Dense-lagrets interface
Det dolda lagret samt utgångslagret representeras av interfacet `ml::dense_layer::Interface`.
En skarp implementation skapas först under **L08–L09**. Fram tills dess implementerar ni en enkel
stubbklass `ml::dense_layer::Stub` som placeholder, se avsnitt 3 nedan.

I headerfilen `ml/dense_layer/interface.h`, implementera klassen `Interface` i namnrymden
`ml::dense_layer`. Tabellerna nedan använder följande typer från `ml/types.h`. Inuti namnrymden
`ml::dense_layer` räcker det att skriva `Matrix1d` respektive `Matrix2d`:

| Typ | Definition | Innehåll |
|---|---|---|
| `ml::Matrix1d` | `std::vector<double>` | En vektor av flyttal, t.ex. ett lagers utdata, fel eller indata. |
| `ml::Matrix2d` | `std::vector<ml::Matrix1d>` | Ett lagers vikter: en rad per nod och en kolumn per vikt. |

Samtliga metoder utom destruktorn ska deklareras som rent virtuella (`= 0`):

| Metod | Ingående argument | Returtyp | Markeras | Beskrivning |
|---|---|---|---|---|
| `~Interface()` | - | - | `virtual`, `noexcept`, `= default` | Destruktor. |
| `nodeCount()` | - | `std::size_t` | `[[nodiscard]]`, `const`, `noexcept` | Antal noder i lagret. |
| `weightCount()` | - | `std::size_t` | `[[nodiscard]]`, `const`, `noexcept` | Antal vikter per nod. |
| `output()` | - | `const Matrix1d&` | `[[nodiscard]]`, `const`, `noexcept` | Lagrets utdata. |
| `error()` | - | `const Matrix1d&` | `[[nodiscard]]`, `const`, `noexcept` | Lagrets fel. |
| `weights()` | - | `const Matrix2d&` | `[[nodiscard]]`, `const`, `noexcept` | Lagrets vikter. |
| `feedforward()` | `const Matrix1d& input` | `bool` | `noexcept` | Genomför feedforward med indatan `input`. |
| `backpropagate()` | `const Matrix1d& reference` | `bool` | `noexcept` | Utgångslager: beräknar felet utifrån referensvärdena `reference`. |
| `backpropagate()` | `const Interface& nextLayer` | `bool` | `noexcept` | Dolt lager: beräknar felet utifrån nästa lager, `nextLayer`. |
| `optimize()` | `const Matrix1d& input`, `double learningRate` | `bool` | `noexcept` | Uppdaterar bias och vikter utifrån indatan `input` och lärhastigheten `learningRate`. |

Beräkningsmetoderna (`feedforward()`, `backpropagate()` och `optimize()`) returnerar `false` vid
ogiltig indata, dvs. felaktig dimension eller ogiltig lärhastighet, och `true` annars. Anroparen
kan då själv avgöra vad som ska göras åt felet. Som i **L02** och **L04** är `std::terminate()`
reserverad för konstruktorn, som inte kan returnera någon felkod till anroparen.

---

### 3. Stubbklassen
Stubben genomför ingen riktig beräkning. Den finns enbart för att annan kod ska gå att kompilera,
testköra och enhetstesta mot ett riktigt `dense_layer::Interface` innan en skarp
`Dense`-implementation finns (se **L08–L09**). Nätverket ni bygger under **L07** testas i sin
helhet mot denna stubb, så det är värt att få den rätt.

I headerfilen `ml/dense_layer/stub.h`, implementera klassen `Stub` i namnrymden
`ml::dense_layer`. Klassen ärver `Interface` via publikt arv och markeras `final`.

Klassen ska inneha följande privata medlemsvariabler:

| Medlemsvariabel | Typ | Innehåll |
|---|---|---|
| `myOutput` | `Matrix1d` | Lagrets utdata, ett element per nod. |
| `myError` | `Matrix1d` | Lagrets fel, ett element per nod. |
| `myWeights` | `Matrix2d` | Lagrets vikter, en rad per nod och en kolumn per vikt. |
| `myFeedforwardCount` | `std::size_t` | Antal anrop till `feedforward()`. |

Klassen ska inneha följande publika metoder:

| Metod | Ingående argument | Returtyp | Markeras | Beskrivning |
|---|---|---|---|---|
| `Stub()` | `std::size_t nodeCount`, `std::size_t weightCount`, `double outputValue = 0.5` | - | `explicit`, `noexcept` | Konstruktor. Sätter varje element i `myOutput` till `outputValue`, samt `myError` och `myWeights` till nollor. Skriver ut ett felmeddelande och anropar `std::terminate()` om `nodeCount` eller `weightCount` är 0. |
| `~Stub()` | - | - | `noexcept`, `override`, `= default` | Destruktor. |
| `nodeCount()` | - | `std::size_t` | `[[nodiscard]]`, `const`, `noexcept`, `override` | Returnerar storleken på `myOutput`. |
| `weightCount()` | - | `std::size_t` | `[[nodiscard]]`, `const`, `noexcept`, `override` | Returnerar storleken på första raden i `myWeights`. |
| `output()` | - | `const Matrix1d&` | `[[nodiscard]]`, `const`, `noexcept`, `override` | Returnerar `myOutput`. |
| `error()` | - | `const Matrix1d&` | `[[nodiscard]]`, `const`, `noexcept`, `override` | Returnerar `myError`. |
| `weights()` | - | `const Matrix2d&` | `[[nodiscard]]`, `const`, `noexcept`, `override` | Returnerar `myWeights`. |
| `feedforward()` | `const Matrix1d& input` | `bool` | `noexcept`, `override` | Ökar först `myFeedforwardCount`, oavsett indata. Returnerar sedan `true` om `input.size() == weightCount()`, annars `false`. |
| `backpropagate()` | `const Matrix1d& reference` | `bool` | `noexcept`, `override` | Returnerar `true` om `reference.size() == nodeCount()`, annars `false`. |
| `backpropagate()` | `const Interface& nextLayer` | `bool` | `noexcept`, `override` | Returnerar `true` om `nextLayer.weightCount() == nodeCount()`, annars `false`. |
| `optimize()` | `const Matrix1d& input`, `double learningRate` | `bool` | `noexcept`, `override` | Returnerar `true` om `input.size() == weightCount()` samt `0.0 < learningRate < 1.0`, annars `false`. |
| `setOutput()` | `double outputValue` | `void` | `noexcept` | Sätter varje element i `myOutput` till `outputValue`. Ingår inte i `Interface`. |
| `feedforwardCount()` | - | `std::size_t` | `[[nodiscard]]`, `const`, `noexcept` | Returnerar `myFeedforwardCount`. Ingår inte i `Interface`. |
| `clearFeedforwardCount()` | - | `void` | `noexcept` | Nollställer `myFeedforwardCount`. Ingår inte i `Interface`. |

Radera default-konstruktorn, copy- och move-konstruktorerna samt tillhörande
tilldelningsoperatorer (`= delete`).

#### Att tänka på
* **Beräkningsmetoderna beräknar avsiktligt ingenting.** De genomför endast range-checkarna i
  tabellen ovan: utdatan förblir `outputValue` oavsett vad som matas in, och felet förblir noll.
* **Blanda inte ihop noder och vikter.** Indatan till ett lager har ett värde per vikt, medan
  referensvärdena har ett värde per nod.
* **Antalet noder och vikter behöver inga egna medlemsvariabler.** Båda går att läsa ur
  vektorerna ni redan har. Att konstruktorn vägrar 0 noder och 0 vikter är det som gör det säkert
  för `weightCount()` att läsa första raden i `myWeights`.
* **Stubben har ingen bias.** Den ingår inte i `Interface`, och stubben optimerar aldrig
  någonting, så det finns inget att lägga i den. **OBS!** Det gäller bara stubben: det skarpa
  `Dense`-lagret i **L08–L09** behöver en biasvektor, som justeras tillsammans med vikterna vid
  varje anrop till `optimize()`.
* **`outputValue` gör två stubbar möjliga att skilja åt.** Ett nätverk vars samtliga lager
  rapporterar samma utdata kan inte visa om en prediktion kom från utgångslagret eller från det
  dolda lagret. Ger man varje lager sitt eget värde blir det synligt.
* **Getters på stubben behöver ett eget `[[nodiscard]]`.** Attribut ärvs inte, så utan det
  kompilerar ett bortkastat anrop som `stub.nodeCount();` utan varning, så snart anropet görs på en
  `Stub` i stället för via en `Interface&`.
* **`setOutput()` styr utdatan för ett helt nätverk i ett test.** Nätverket i **L07** lagrar sina
  lager som referenser, så ett anrop till `setOutput()` på ett av lagren ändrar vad nätverket
  predikterar. Det avslöjar om nätverket har sparat en egen kopia av utdatan, vilket noteringen om
  `predict()` i **L07** är till för att förhindra.
* **`feedforwardCount()` räknar varje anrop, även avvisade.** Räknaren mäter hur ofta lagret
  *ombads* göra feedforward, inte hur ofta det gick med på det. `train()` i **L07** genomför en
  feedforward per träningsuppsättning och epok, och en loop som kör ett enda pass i stället för
  samtliga epoker stämmer fortfarande dimensionsmässigt. Utan räknaren går den inte att skilja från
  en korrekt loop.

---

### 4. Kompilering och test
Skriv en `main`-funktion i `main.cpp` som testar stubbklassen via följande steg:
1. Skapa en `ml::dense_layer::Stub`-instans döpt `hiddenLayer` med 3 noder och 2 vikter per nod 
   (defaultvärdet `0.5` som utdata).
2. Skapa en `ml::dense_layer::Stub`-instans döpt `outputLayer` med 1 nod, 3 vikter per nod samt 
   utdatan `0.8`. Antalet vikter i utgångslagret matchar antalet noder i det dolda lagret.
3. Skriv ut antalet noder, antalet vikter per nod samt utdatan för båda lagren.
4. Anropa `hiddenLayer.feedforward()` två gånger; först med 2 insignaler, sedan med 3. Skriv ut 
   returvärdena, lagrets utdata samt feedforward-räknaren.
5. Anropa `outputLayer.backpropagate()` med 1 referensvärde samt `hiddenLayer.backpropagate()` med 
   `outputLayer` som nästa lager. Skriv ut returvärdena.
6. Anropa `hiddenLayer.optimize()` med 2 insignaler två gånger; först med lärhastigheten `0.01`, 
   sedan med `1.5`. Skriv ut returvärdena.
7. Anropa `hiddenLayer.setOutput(0.1)` samt `hiddenLayer.clearFeedforwardCount()`. Skriv ut lagrets 
   utdata samt feedforward-räknaren.

Använd `std::boolalpha` för att skriva ut returvärdena som `true`/`false` i stället för `1`/`0`.

Kompilera och testkör programmet. Ni ska få följande utskrift:

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

Notera följande i utskriften:
* Utdatan är fortfarande `0.5` efter feedforward, eftersom stubben inte beräknar någonting.
* Feedforward-räknaren är `2` trots att det andra anropet avvisades.
* `hiddenLayer.backpropagate(outputLayer)` lyckas eftersom utgångslagrets antal vikter per nod (3) 
  matchar det dolda lagrets antal noder (3). Detta är samma koppling som nätverket i **L07** 
  bygger på.

---

### 5. Enhetstester
Kontrollera er implementation mot testsviten i `exercises/test`. Se testsvitens 
[README](../exercises/test/README.md) för detaljer.

1. Bygg och kör testsviten via kommandot `make` i katalogen `exercises/test`. Testramverket måste
   ha hämtats först, se avsnitt 1.
    * Skriver ni er kod i er egen `ml`-kodbas i stället för i `exercises`, ange sökvägen till den
      via `make ML_DIR=<sökväg till er ml-katalog>`.
2. Åtgärda eventuella fel och kör testsviten igen, tills samtliga testfall går igenom.

Testsviten kompilerar inte förrän båda headerfilerna finns och deklarerar samtliga metoder som 
testerna anropar. Läs det första kompileringsfelet; det anger oftast vilken metod som saknas eller 
har fel signatur.

Konstruktorns anrop till `std::terminate()` testas inte av testsviten, eftersom det avslutar hela 
testprogrammet. Kontrollera det för hand genom att skapa en stubb med 0 noder.

---

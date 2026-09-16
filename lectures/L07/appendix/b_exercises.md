# Bilaga B - Övningsuppgift: Enkelt neuralt nätverk i C++ (del II)
Ni ska bygga vidare på er `ml`-kodbas från **L06** med ett interface samt en klass för ett enkelt 
neuralt nätverk innehållande ett dolt lager samt ett utgångslager. Lagren utgörs av stubbklassen 
`ml::dense_layer::Stub` från **L06** tills en skarp dense-lagerimplementation finns (se 
**L08–L09**).

---

### 1. Kom igång
Har ni inte redan hämtat testramverket under **L06**, gör det först. Kör följande kommando en
gång, i repots rotkatalog:

```bash
git submodule update --init --recursive
```

Katalogen [`exercises`](../exercises) innehåller dense-lagrets interface och stubbklass från
**L06**, samt en testsvit i `exercises/test` (se avsnitt 8). Skriv er kod där, eller bygg ut
katalogstrukturen i er befintliga `ml`-kodbas enligt nedan:

```
ml/
├── include/
│   └── ml/
│       ├── dense_layer/
│       │   ├── interface.h
│       │   └── stub.h
│       ├── neural_network/
│       │   ├── interface.h
│       │   └── shallow.h
│       └── types.h
├── source/
│   ├── neural_network/
│   │   └── shallow.cpp
│   └── main.cpp
└── Makefile
```

Glöm inte att lägga till `source/neural_network/shallow.cpp` i er makefil.

---

### 2. Interface för neurala nätverk
I headerfilen `ml/neural_network/interface.h`, implementera klassen `Interface` i namnrymden
`ml::neural_network`. Typerna är desamma som i **L06**: `Matrix1d` är en `std::vector<double>`,
och `Matrix2d` en `std::vector<Matrix1d>`, båda från `ml/types.h`.

| Metod | Ingående argument | Returtyp | Markeras | Beskrivning |
|---|---|---|---|---|
| `~Interface()` | - | - | `virtual`, `noexcept`, `= default` | Destruktor. |
| `predict()` | `const Matrix1d& input` | `const Matrix1d&` | `virtual`, `noexcept`, `= 0` | Predikterar utdata utifrån indatan `input`. |

---

### 3. Klassen Shallow
I headerfilen `ml/neural_network/shallow.h`, deklarera klassen `Shallow` i namnrymden
`ml::neural_network`. Klassen ärver `Interface` via publikt arv och markeras `final`. Metoderna
implementeras i `source/neural_network/shallow.cpp`, se avsnitt 4–6.

Klassen ska inneha följande privata medlemsvariabler:

| Medlemsvariabel | Typ | Innehåll |
|---|---|---|
| `myHiddenLayer` | `dense_layer::Interface&` | Nätverkets dolda lager. |
| `myOutputLayer` | `dense_layer::Interface&` | Nätverkets utgångslager. |
| `myTrainInput` | `const Matrix2d&` | Träningsdatans indata, en rad per träningsuppsättning. |
| `myTrainOutput` | `const Matrix2d&` | Träningsdatans utdata, en rad per träningsuppsättning. |
| `myTrainSetCount` | `const std::size_t` | Antal fullständiga träningsuppsättningar, dvs. det minsta av `myTrainInput.size()` och `myTrainOutput.size()`. |

Klassen ska inneha följande publika metoder:

| Metod | Ingående argument | Returtyp | Markeras | Beskrivning |
|---|---|---|---|---|
| `Shallow()` | `dense_layer::Interface& hiddenLayer`, `dense_layer::Interface& outputLayer`, `const Matrix2d& trainInput`, `const Matrix2d& trainOutput` | - | `explicit`, `noexcept` | Konstruktor. Initierar samtliga medlemsvariabler. |
| `~Shallow()` | - | - | `noexcept`, `override`, `= default` | Destruktor. |
| `predict()` | `const Matrix1d& input` | `const Matrix1d&` | `noexcept`, `override` | Genomför feedforward genom hela nätverket och returnerar utgångslagrets utdata. |
| `train()` | `std::size_t epochCount`, `double learningRate = 0.01` | `bool` | `noexcept` | Tränar nätverket i `epochCount` epoker. Returnerar `true` efter genomförd träning, annars `false`. |

Radera default-konstruktorn, copy- och move-konstruktorerna samt tillhörande
tilldelningsoperatorer (`= delete`). Ni får gärna lägga till fler privata metoder vid behov.

#### Att tänka på
* **Utgå från interfacet.** Kopiera in innehållet i `ml/neural_network/interface.h` i
  `shallow.h`, döp om klassen till `Shallow`, låt den ärva `Interface`, samt ersätt `virtual`
  och `= 0` med `override`. Lägg därefter till det som saknas enligt tabellerna ovan.
* **Två olika `Interface`.** Inkludera både `ml/neural_network/interface.h` och
  `ml/dense_layer/interface.h` i `shallow.h`. Inuti `Shallow` syftar `Interface` på nätverkets
  interface, så lagren måste skrivas ut som `dense_layer::Interface&`.
* **`predict()` är inte `const`.** Varje prediktion uppdaterar lagrens utdata.
* **`predict()` sparar ingen egen kopia av prediktionen.** Den returnerar en referens direkt till
  utgångslagrets utdata. Det är detta som gör att `setOutput()` på stubben från **L06** slår
  igenom på nätverkets prediktion; en egen kopia i `Shallow` hade gjort den inaktuell.
* **Referenser som medlemsvariabler.** Lagren och träningsdatan måste leva minst lika länge som
  nätverket, eftersom `Shallow` enbart refererar till dem.

---

### 4. Konstruktor och prediktion
Implementera följande i `source/neural_network/shallow.cpp`:

**Konstruktorn:**
* Initiera samtliga medlemsvariabler enligt tabellen i avsnitt 3.

**Metoden `predict()`:**
* Genomför feedforward genom hela nätverket:
    1. Anropa `myHiddenLayer.feedforward(input)` med given indata.
    2. Anropa `myOutputLayer.feedforward(myHiddenLayer.output())` med det dolda lagrets output 
       som indata.
* Returnera `myOutputLayer.output()`, dvs. en referens till utgångslagrets utdata.

**Metoden `train()` (tillfällig):**
* Ge `train()` en tillfällig funktionskropp bestående av `return false;`, så att filen är komplett 
  och går att kompilera. Den fullständiga implementationen genomförs i avsnitt 6.
* `false` i stället för `true`, så att en metod ni glömmer att färdigställa rapporterar att den 
  misslyckades i stället för att tyst påstå att allt gick bra.

---

### 5. Kontrollpunkt: prediktion
Uppdatera `main`-funktionen i `main.cpp` så att den:
* Skapar en `ml::dense_layer::Stub`-instans för det dolda lagret samt en för utgångslagret, t.ex. 
  3 noder/2 vikter per nod respektive 1 nod/3 vikter per nod. Antalet vikter i utgångslagret ska 
  matcha antalet noder i det dolda lagret.
* Skapar en `ml::neural_network::Shallow`-instans utifrån dessa två lager samt valfri 
  träningsdata, t.ex. ett 2-bitars XOR-mönster.
* Genomför en prediktion för varje träningsuppsättnings indata, och skriver ut indatan samt den 
  predikterade utdatan i terminalen.

Kompilera och testkör programmet. Ni ska få följande utskrift (dense-lagren är stubbar, så 
prediktionen är alltid 0.5):

```
--------------------------------------------------------------------------------
Input: 0 0, predicted output: 0.5
Input: 0 1, predicted output: 0.5
Input: 1 0, predicted output: 0.5
Input: 1 1, predicted output: 0.5
--------------------------------------------------------------------------------
```

---

### 6. Träningsmetod
Ersätt den tillfälliga versionen av `train()` i `source/neural_network/shallow.cpp` med en 
fullständig implementation. Se [bilaga A](./a_training_loop.md) för en genomgång av träningsloopens 
struktur.

**Indatakontroll:**
* Returnera `false` om `myTrainSetCount == 0`, `epochCount == 0` eller 
  `learningRate <= 0.0 || learningRate >= 1.0`.
* Som i **L02** och **L04** rapporterar `train()` ogiltiga argument via sitt returvärde. Endast en 
  konstruktor anropar `std::terminate()`, eftersom den inte kan returnera någon felkod till 
  anroparen.

**Träning:**
* Iterera önskat antal epoker med en for-loop: 
  `for (std::size_t epoch{}; epoch < epochCount; ++epoch)`.
* För varje epok, iterera genom samtliga träningsuppsättningar: 
  `for (std::size_t x{}; x < myTrainSetCount; ++x)`.
* För varje träningsuppsättning `x`, genomför följande tre steg:
    1. **Feedforward:** Anropa `predict(myTrainInput[x])`. Detta genomför feedforward genom både 
       det dolda lagret och utgångslagret.
    2. **Backpropagation:**
        * Beräkna felet i utgångslagret: `myOutputLayer.backpropagate(myTrainOutput[x])`.
        * Beräkna felet i det dolda lagret utifrån utgångslagrets fel och vikter: 
          `myHiddenLayer.backpropagate(myOutputLayer)`.
    3. **Optimering:**
        * Optimera det dolda lagret: `myHiddenLayer.optimize(myTrainInput[x], learningRate)`.
        * Optimera utgångslagret utifrån det dolda lagrets output: 
          `myOutputLayer.optimize(myHiddenLayer.output(), learningRate)`.

Var och en av de fyra lageranropen ovan returnerar `bool` (se **L06**). Returnera `false` så snart 
något av dem misslyckas: en dimensionsmiss betyder att nätverket är felkopplat, och att fortsätta 
träna därifrån ger bara meningslösa siffror.

**Returvärde:** `false` vid ogiltiga argument eller ett misslyckat lageranrop, annars `true` efter 
genomförd träning.

---

### 7. Kompilering och test
Uppdatera `main.cpp` så att `train()` anropas innan prediktion genomförs:
* Träna nätverket under 100 epoker med lärhastigheten `0.01`.
* Kontrollera returvärdet från `train()`. Skriv ut ett felmeddelande och avsluta programmet med en 
  felkod om träningen misslyckas.
* Skriv ut det dolda lagrets feedforward-räknare (`feedforwardCount()`) efter träningen.

Kompilera och testkör programmet. Ni ska få följande utskrift:

```
--------------------------------------------------------------------------------
Feedforward count after training: 400
--------------------------------------------------------------------------------
Input: 0 0, predicted output: 0.5
Input: 0 1, predicted output: 0.5
Input: 1 0, predicted output: 0.5
Input: 1 1, predicted output: 0.5
--------------------------------------------------------------------------------
```

Notera följande i utskriften:
* Feedforward-räknaren ska vara exakt 400, dvs. en feedforward per träningsuppsättning (4) och 
  epok (100). Ett annat värde betyder att träningsloopen itererar fel.
* Bli inte oroliga över att nätverket inte predikterar korrekt; dense-lagren är fortfarande 
  stubbar och tränar därmed inte på riktigt. En skarp implementation av `Dense` läggs till under 
  **L08–L09**.

---

### 8. Enhetstester
Kontrollera er implementation mot testsviten i `exercises/test`. Se testsvitens
[README](../exercises/test/README.md) för detaljer. Testsviten innehåller även stubbtesterna från
**L06**, så det räcker att köra denna.

1. Bygg och kör testsviten via kommandot `make` i katalogen `exercises/test`. Testramverket måste
   ha hämtats först, se avsnitt 1.
    * Skriver ni er kod i er egen `ml`-kodbas i stället för i `exercises`, ange sökvägen till den
      via `make ML_DIR=<sökväg till er ml-katalog>`.
2. Åtgärda eventuella fel och kör testsviten igen, tills samtliga testfall går igenom.

Testsviten kompilerar inte förrän samtliga headerfiler finns och deklarerar samtliga metoder som
testerna anropar. Läs det första kompileringsfelet; det anger oftast vilken metod som saknas eller
har fel signatur.

---

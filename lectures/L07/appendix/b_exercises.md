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
**L06**, samt en testsvit i `exercises/test` (se avsnitt 8). Där ligger även tre tomma filer
märkta `@todo`, som ni fyller i under lektionen:
* `include/ml/neural_network/interface.h` (avsnitt 2).
* `include/ml/neural_network/shallow.h` (avsnitt 3).
* `source/ml/neural_network/shallow.cpp` (avsnitt 4 och 6).

Skriv er kod där, eller bygg ut katalogstrukturen i er befintliga `ml`-kodbas enligt nedan:

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
│   ├── ml/
│   │   └── neural_network/
│   │       └── shallow.cpp
│   └── main.cpp
└── Makefile
```

Glöm inte att lägga till `source/ml/neural_network/shallow.cpp` i er makefil.

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
implementeras i `source/ml/neural_network/shallow.cpp`, se avsnitt 4–6.

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
| `Shallow()` | `dense_layer::Interface& hiddenLayer`, `dense_layer::Interface& outputLayer`, `const Matrix2d& trainInput`, `const Matrix2d& trainOutput` | - | `explicit`, `noexcept` | Konstruktor. Initierar samtliga medlemsvariabler, och avslutar programmet vid ogiltig konfiguration (se avsnitt 4). |
| `~Shallow()` | - | - | `noexcept`, `override`, `= default` | Destruktor. |
| `predict()` | `const Matrix1d& input` | `const Matrix1d&` | `noexcept`, `override` | Genomför feedforward genom hela nätverket och returnerar utgångslagrets utdata. |
| `train()` | `std::size_t epochCount`, `double learningRate = 0.01` | `bool` | `noexcept` | Tränar nätverket i `epochCount` epoker. Returnerar `true` efter genomförd träning, annars `false`. |

Radera default-konstruktorn, copy- och move-konstruktorerna samt tillhörande
tilldelningsoperatorer (`= delete`). Ni får gärna lägga till fler privata metoder vid behov.

#### Att tänka på
* **Utgå från interfacet.** Kopiera in innehållet i `ml/neural_network/interface.h` i
  `shallow.h`, döp om klassen till `Shallow`, låt den ärva `Interface`, samt ersätt `virtual`
  och `= 0` med `override`. Lägg därefter till det som saknas enligt tabellerna ovan.
* **Två olika `Interface`.** Inuti `Shallow` syftar `Interface` på nätverkets interface, så lagren
  måste skrivas ut som `dense_layer::Interface&`.
* **Inkludera nätverkets interface, forward-deklarera dense-lagrets.** `shallow.h` inkluderar
  `ml/neural_network/interface.h`, eftersom `Shallow` ärver den klassen. Dense-lagrets interface
  ska däremot inte inkluderas här; en forward-deklaration räcker, eftersom medlemsvariablerna
  enbart är referenser:

  ```cpp
  // clang-format off
  namespace ml::dense_layer { class Interface; }
  // clang-format on
  ```

  Forward-deklarera hellre än inkludera i headerfiler när det går. Det håller nere beroendena: den
  som inkluderar `shallow.h` drar då inte med sig dense-lagrets interface i onödan, och en ändring
  i `dense_layer/interface.h` tvingar inte fram en omkompilering av allt som använder nätverket.
* **Inkludera dense-lagrets interface i `shallow.cpp`.** Där anropas lagrens metoder, och då krävs
  den fullständiga klassdefinitionen. En forward-deklaration räcker inte.
* **`predict()` är inte `const`.** Varje prediktion uppdaterar lagrens utdata.
* **`predict()` sparar ingen egen kopia av prediktionen.** Den returnerar en referens direkt till
  utgångslagrets utdata. Det är detta som gör att `setOutput()` på stubben från **L06** slår
  igenom på nätverkets prediktion; en egen kopia i `Shallow` hade gjort den inaktuell.
* **Referenser som medlemsvariabler.** Lagren och träningsdatan måste leva minst lika länge som
  nätverket, eftersom `Shallow` enbart refererar till dem.

---

### 4. Konstruktor och prediktion
Implementera följande i `source/ml/neural_network/shallow.cpp`:

**Konstruktorn:**
* Initiera samtliga medlemsvariabler enligt tabellen i avsnitt 3.
* Skriv därefter ut ett felmeddelande och anropa `std::terminate()` om något av följande gäller:
    * `myTrainSetCount == 0`, dvs. det minsta av `trainInput.size()` och `trainOutput.size()` är
      0. Utan en enda fullständig träningsuppsättning finns det ingenting att träna på.
    * `myHiddenLayer.nodeCount() != myOutputLayer.weightCount()`, dvs. lagren är felkopplade.
      Utgångslagret måste ha en vikt per nod i det dolda lagret, eftersom det dolda lagrets
      utdata utgör utgångslagrets indata.
* Som i **L02**, **L04** samt **L06** är konstruktorn den enda plats som avslutar programmet,
  eftersom den inte kan returnera någon felkod till anroparen.

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
Testprogrammet i `source/main.cpp` är redan skrivet. Det genomför följande:
1. Skapar två stubbar: det dolda lagret med 3 noder och 2 vikter per nod, samt utgångslagret med
   1 nod och 3 vikter per nod. Antalet vikter i utgångslagret matchar antalet noder i det dolda
   lagret, vilket är det som kopplar ihop dem.
2. Skapar ett `ml::neural_network::Shallow`-nätverk av de två lagren, med ett 2-bitars
   XOR-mönster som träningsdata.
3. Predikterar för varje träningsuppsättnings indata, tränar nätverket i 100 epoker med
   lärhastigheten `0.01`, och predikterar sedan igen.

Kompilera och testkör programmet via kommandot `make`. Eftersom `train()` fortfarande är en
platshållare som returnerar `false` avbryts programmet efter de första prediktionerna:

```
--------------------------------------------------------------------------------
Predictions before training:
Input: 0 0, predicted output: 0.5
Input: 0 1, predicted output: 0.5
Input: 1 0, predicted output: 0.5
Input: 1 1, predicted output: 0.5
--------------------------------------------------------------------------------
Training failed!
```

Prediktionen är alltid 0.5, eftersom dense-lagren är stubbar. Felmeddelandet är väntat i detta
läge; det försvinner när `train()` implementeras i avsnitt 6.

---

### 6. Träningsmetod
Ersätt den tillfälliga versionen av `train()` i `source/ml/neural_network/shallow.cpp` med en 
fullständig implementation. Se [bilaga A](./a_training_loop.md) för en genomgång av träningsloopens 
struktur.

**Indatakontroll:**
* Returnera `false` om `epochCount == 0` eller `learningRate <= 0.0 || learningRate >= 1.0`.
* Träningsdatan behöver inte kontrolleras här. Konstruktorn har redan garanterat minst en
  fullständig träningsuppsättning samt att lagren är rätt kopplade (se avsnitt 4).
* Som i **L02** och **L04** rapporterar `train()` ogiltiga argument via sitt returvärde. Endast
  konstruktorn anropar `std::terminate()`, eftersom den inte kan returnera någon felkod till
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
Kompilera och testkör programmet igen via kommandot `make`. Nu ska hela programmet gå igenom:

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

Notera följande i utskriften:
* Prediktionerna är identiska före och efter träningen. Detta är förväntat, då dense-lagren
  fortfarande är stubbar och tränar därmed inte på riktigt. En skarp implementation av `Dense` läggs 
  till under **L08–L09**.
* Att träningsloopen verkligen genomför en feedforward per träningsuppsättning och epok kontrolleras 
  av testsviten (se avsnitt 8), inte av testprogrammet.

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

Konstruktorns anrop till `std::terminate()` testas inte av testsviten, eftersom det avslutar hela
testprogrammet. Kontrollera dem för hand: skapa ett nätverk utan träningsdata, samt ett vars lager
inte matchar varandra.

---

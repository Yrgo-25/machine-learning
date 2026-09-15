# Bilaga B - Övningsuppgift: Enkelt neuralt nätverk i C++ (del II)
Ni ska bygga vidare på er `ml`-kodbas från **L06** med ett interface samt en klass för ett enkelt 
neuralt nätverk innehållande ett dolt lager samt ett utgångslager. Lagren utgörs av stubbklassen 
`ml::dense_layer::Stub` från **L06** tills en skarp dense-lagerimplementation finns (se 
**L08–L09**).

---

### 1. Katalogstruktur
Bygg ut katalogstrukturen i er befintliga `ml`-kodbas enligt nedan:

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
I headerfilen `ml/neural_network/interface.h`, lägg till namnrymden `ml::neural_network`. 
I denna namnrymd, implementera ett interface döpt `Interface`:

* **`~Interface()`:** Ska sättas till `default` samt markeras `virtual` och `noexcept`.
* **`predict(input)`:** Rent virtuell. `input`: skrivskyddad flyttalsvektor med indatan som 
  prediktionen ska baseras på. Returnerar en referens till en flyttalsvektor med det predikterade 
  värdet. Ska markeras `noexcept` (**inte** `const`, eftersom lagrens output uppdateras vid varje 
  prediktion).

---

### 3. Klassen Shallow - deklaration
I headerfilen `ml/neural_network/shallow.h`, lägg till namnrymden `ml::neural_network`. Utgå från 
interfacet och gör om det till en underklass:
1. Kopiera in innehållet från `interface.h`, inklusive `#pragma once` samt namnrymden.
2. Inkludera `ml/neural_network/interface.h`, så att basklassen är känd.
3. Döp om klassen till `Shallow` och låt den ärva `Interface` via publikt arv. Klassen ska markeras 
   `final`.
4. Ta bort `virtual` samt `= 0` från metoderna; markera dem `override` i stället.

Efter omvandlingen ska klassen inneha följande publika metoder:
* **`~Shallow()`:** Ska markeras `default`, `noexcept` samt `override`.
* **`predict()`:** Överlagring av motsvarande metod i interfacet. Ska markeras `noexcept` samt 
  `override`.

---

### 4. Privata medlemsvariabler
Lägg till följande privata medlemsvariabler i `Shallow`:
* **`myHiddenLayer`:** Referens till nätverkets dolda lager, erhålles via konstruktorn.
* **`myOutputLayer`:** Referens till nätverkets utgångslager, erhålles via konstruktorn.
* **`myTrainInput`:** Referens till träningsdatans indata, erhålles via konstruktorn.
* **`myTrainOutput`:** Referens till träningsdatans utdata, erhålles via konstruktorn.
* **`myTrainSetCount`:** Konstant osignerat heltal som anger antalet fullständiga 
  träningsuppsättningar (dvs. det minsta av `myTrainInput.size()` och `myTrainOutput.size()`).

Medlemsvariablerna läggs till före konstruktorn, så att ni vet exakt vad konstruktorn ska initiera.

---

### 5. Konstruktor - deklaration
Lägg till klassens enda implementerade konstruktor som publik metod:
* **`Shallow()`:** Tar emot `hiddenLayer` samt `outputLayer` (nätverkets dolda lager respektive 
  utgångslager, `ml::dense_layer::Interface&`), samt `trainInput` och `trainOutput` 
  (skrivskyddade, tvådimensionella flyttalsvektorer med träningsdatans in- och utdata). Ska 
  markeras `explicit` samt `noexcept`.

Konstruktorn deklareras enbart här; den implementeras i avsnitt 8.

---

### 6. Borttagna konstruktorer och operatorer
Radera klassens default-konstruktor, kopierings- och förflyttningskonstruktorer samt tillhörande 
operatorer.

---

### 7. Övriga metoder - deklaration
Lägg till följande publika metod i `Shallow`:
* **`train(epochCount, learningRate = 0.01)`:** Tränar nätverket (implementeras i avsnitt 10). 
  `epochCount`: antal epoker att träna (osignerat heltal). `learningRate`: lärhastighet 
  (flyttal). Returnerar `true` om träning genomfördes, annars `false`. Ska markeras `noexcept`.

Ni får gärna lägga till fler (privata) metoder vid behov.

---

### 8. Konstruktor och prediktion
Implementera följande i `source/neural_network/shallow.cpp`:

**Konstruktorn:**
* Initiera samtliga medlemsvariabler enligt beskrivningen ovan.

**Metoden `predict()`:**
* Genomför feedforward genom hela nätverket:
    1. Anropa `myHiddenLayer.feedforward(input)` med given indata.
    2. Anropa `myOutputLayer.feedforward(myHiddenLayer.output())` med det dolda lagrets output 
       som indata.
* Returnera `myOutputLayer.output()`. Detta är en referens till utgångslagrets utdata, så ingen 
  egen lagringsvariabel för prediktionen behövs i `Shallow`.

**Metoden `train()` (tillfällig):**
* Ge `train()` en tillfällig funktionskropp bestående av `return false;`, så att filen är komplett 
  och går att kompilera. Den fullständiga implementationen genomförs i avsnitt 10.
* `false` i stället för `true`, så att en metod ni glömmer att färdigställa rapporterar att den 
  misslyckades i stället för att tyst påstå att allt gick bra.

---

### 9. Kontrollpunkt: prediktion
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

### 10. Träningsmetod
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

### 11. Kompilering och test
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

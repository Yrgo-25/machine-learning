# Bilaga B - Övningsuppgift: Enkelt neuralt nätverk i C++ (del I)
Ni ska bygga vidare på er `ml`-kodbas med ett interface samt en klass för ett enkelt neuralt 
nätverk innehållande ett dolt lager samt ett utgångslager.

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

### 2. Dense-lagrets interface och stubbklass
Det dolda lagret samt utgångslagret representeras av interfacet `ml::dense_layer::Interface`. 
En skarp implementation skapas först under **L08–L09**. Fram tills dess implementerar ni en enkel 
stubbklass `ml::dense_layer::Stub` som placeholder.

**Interfacet (`ml/dense_layer/interface.h`):**
I namnrymden `ml::dense_layer`, implementera ett interface döpt `Interface`. Samtliga metoder 
(förutom destruktorn) ska deklareras som rent virtuella (`= 0`).

* **`~Interface()`:** Ska sättas till `default` samt markeras `virtual` och `noexcept`.

Getters, samtliga `const`, `noexcept` samt `[[nodiscard]]`:

| Metod | Returnerar |
|---|---|
| `nodeCount()` | Antal noder i lagret (`std::size_t`). |
| `weightCount()` | Antal vikter per nod (`std::size_t`). |
| `output()` | Referens till lagrets utdata (skrivskyddad flyttalsvektor). |
| `error()` | Referens till lagrets fel (skrivskyddad flyttalsvektor). |
| `weights()` | Referens till lagrets vikter (skrivskyddad, tvådimensionell flyttalsvektor). |

Beräkningsmetoder, samtliga `noexcept` och med returtypen `bool`. Var och en returnerar `false` 
vid ogiltig indata (felaktig dimension eller ogiltig lärhastighet) och `true` annars, så att 
anroparen själv kan avgöra vad som ska göras åt felet. Som i **L02** och **L04** är 
`std::terminate()` reserverad för konstruktorn, som inte kan returnera någon felkod till 
anroparen:
* **`feedforward(input)`:** Genomför feedforward.
    * `input`: skrivskyddad flyttalsvektor med indata.
* **`backpropagate(output)`** (utgångslager): Beräknar fel utifrån referensvärden.
    * `output`: skrivskyddad flyttalsvektor med referensvärden.
* **`backpropagate(nextLayer)`** (dolt lager): Beräknar fel utifrån nästa lager.
    * `nextLayer`: referens till nästa lager (`const Interface&`).
* **`optimize(input, learningRate)`:** Uppdaterar bias och vikter.
    * `input`: skrivskyddad flyttalsvektor.
    * `learningRate`: flyttal.

**Stubbklassen (`ml/dense_layer/stub.h`):**
I namnrymden `ml::dense_layer`, implementera en underklass döpt `Stub` som ärver `Interface` via 
publikt arv. Klassen ska markeras `final`. Stubben genomför ingen riktig beräkning; den finns 
enbart för att annan kod ska gå att kompilera, testköra och enhetstesta mot ett riktigt 
`dense_layer::Interface` innan en skarp `Dense`-implementation finns (se **L08–L09**). Nätverket 
ni bygger i avsnitt 4-8 nedan testas i sin helhet mot denna stubb, så det är värt att få den rätt.

Lägg till privata medlemsvariabler för utdatan, felet, vikterna samt feedforward-räknaren.

Antalet noder och antalet vikter per nod behöver inga egna medlemsvariabler. Båda går att läsa ur 
de vektorer ni redan har: `nodeCount()` är utdatavektorns storlek, och `weightCount()` är 
viktmatrisens bredd.

Bias behöver ingen alls. Den ingår inte i `Interface`, och denna stubb optimerar aldrig 
någonting, så det skulle inte finnas något att lägga i den och inget sätt att läsa ut den igen.

**OBS!** Att utelämna den är en förenkling som bara gäller stubben. Det skarpa `Dense`-lagret ni 
skriver i **L08–L09** behöver en biasvektor: den är en träningsbar parameter som justeras 
tillsammans med vikterna vid varje anrop till `optimize()`.

Klassen ska inneha följande publika metoder:
* **`Stub()`:**
    * Klassens enda implementerade konstruktor.
    * Ska ha följande ingående argument:
        * `nodeCount`: antal noder i lagret (osignerat heltal).
        * `weightCount`: antal vikter per nod (osignerat heltal).
        * `outputValue`: värdet som varje element i utdatavektorn sätts till (flyttal). 
          Defaultvärde: `0.5`.
    * Initierar varje element i utdatavektorn till `outputValue`, samt fel- och viktvektorerna 
      till nollor.
    * Ska skriva ut ett felmeddelande och anropa `std::terminate()` om `nodeCount` eller 
      `weightCount` är 0:
        * Som i **L02** och **L04** är konstruktorn den enda plats som avslutar programmet, 
          eftersom den inte kan returnera någon felkod.
        * Det garanterar också minst en nod och en vikt, vilket är det som gör det säkert för 
          `weightCount()` att läsa bredden på viktmatrisens första rad.
    * `outputValue` är ett argument i stället för en hårdkodad konstant för att två stubbar ska 
      gå att skilja åt. Ett nätverk vars samtliga lager rapporterar samma utdata kan inte visa 
      om en prediktion kom från utgångslagret eller från det dolda lagret; ger man varje lager 
      sitt eget värde blir det synligt.
    * Ska markeras `explicit` samt `noexcept`.
* **`~Stub()`:**
    * Destruktor som överlagrar interfacets destruktor.
    * Ska markeras `default`, `noexcept` samt `override`.
* **Getters** (`nodeCount()`, `weightCount()`, `output()`, `error()`, `weights()`):
    * Överlagrar motsvarande metoder i interfacet.
    * Ska markeras `override` (behåller interfacets `const` och `noexcept`, men **inte** 
      `[[nodiscard]]`).
* **`feedforward()`**, båda varianterna av **`backpropagate()`** samt **`optimize()`**:
    * Genomför endast range-checkar:
        * Returnera `false` när dimensionerna inte stämmer, eller, för `optimize()`, när 
          lärhastigheten ligger utanför `(0.0, 1.0)`.
        * Returnera `true` annars.
    * Beräknar avsiktligt ingenting:
        * Utdatan förblir `outputValue` oavsett vad som matas in.
        * Felet förblir noll.
    * Ska markeras `override` samt `noexcept`.
* **`setOutput()`:**
    * Sätter varje element i utdatavektorn till det angivna värdet.
    * Ska ha ett enda ingående argument:
        * `outputValue`: värdet som varje element i utdatavektorn ska sättas till (flyttal).
    * Returnerar ingenting, och ska markeras `noexcept`.
    * Ingår **inte** i `Interface`. Den finns enbart på stubben.
    * Det är denna metod som gör det möjligt att styra utdatan för ett helt nätverk i ett test:
        * Nätverket lagrar sina lager som referenser, så ett anrop till `setOutput()` på det 
          lager nätverket byggdes med ändrar vad nätverket predikterar.
        * Det avslöjar om nätverket läser sitt utgångslager direkt eller har sparat en egen 
          kopia av utdatan. Just den kopian är vad noteringen i avsnitt 9 nedan är till för att 
          förhindra.
* **`feedforwardCount()`** samt **`clearFeedforwardCount()`:**
    * `feedforwardCount()` returnerar antalet gånger `feedforward()` har anropats på detta 
      lager, och räknar **varje** anrop, inte bara de som klarade range-checken:
        * Öka räknaren (`myFeedforwardCount`) allra först i `feedforward()`, innan indatans 
          storlek kontrolleras, så att ett avvisat anrop höjer den precis som ett godkänt.
        * Räknaren mäter hur ofta lagret *ombads* göra feedforward, inte hur ofta det gick med 
          på det.
        * Ska markeras `[[nodiscard]]`, `const` samt `noexcept`.
    * `clearFeedforwardCount()` nollställer räknaren.
        * Returnerar ingenting, och ska markeras `noexcept`.
    * Ingen av dem ingår i `Interface`.
    * `train()` i **L07** genomför en feedforward per träningsuppsättning och epok, så det är 
      detta som gör träningsloopen möjlig att fastställa i ett test:
        * Inget annat kan göra det: en loop som kör ett enda pass i stället för samtliga epoker 
          stämmer fortfarande dimensionsmässigt och returnerar fortfarande `true`, så utan en 
          räknare går den inte att skilja från en korrekt loop.

För denna klass ska default-konstruktorn samt copy- och move-konstruktorerna (och tillhörande 
operatorer) raderas.

---

### 3. Interface för neurala nätverk
I headerfilen `ml/neural_network/interface.h`, lägg till en namnrymd döpt `ml::neural_network`. 
I denna namnrymd, implementera ett interface döpt `Interface`:

* **`~Interface()`:** Ska sättas till `default` samt markeras `virtual` och `noexcept`.
* **`predict(input)`:** Rent virtuell. `input`: skrivskyddad flyttalsvektor med indatan som 
  prediktionen ska baseras på. Returnerar en referens till en flyttalsvektor med det predikterade 
  värdet. Ska markeras `noexcept` (**inte** `const`, eftersom lagrens output uppdateras vid varje 
  prediktion).

---

### 4. Klassen Shallow - deklaration
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

### 5. Privata medlemsvariabler
Lägg till följande privata medlemsvariabler i `Shallow`:
* **`myHiddenLayer`:** Referens till nätverkets dolda lager, erhålles via konstruktorn.
* **`myOutputLayer`:** Referens till nätverkets utgångslager, erhålles via konstruktorn.
* **`myTrainInput`:** Referens till träningsdatans indata, erhålles via konstruktorn.
* **`myTrainOutput`:** Referens till träningsdatans utdata, erhålles via konstruktorn.
* **`myTrainSetCount`:** Konstant osignerat heltal som anger antalet fullständiga träningsuppsättningar (dvs. det minsta av `myTrainInput.size()` och `myTrainOutput.size()`).

Medlemsvariablerna läggs till före konstruktorn, så att ni vet exakt vad konstruktorn ska initiera.

---

### 6. Konstruktor - deklaration
Lägg till klassens enda implementerade konstruktor som publik metod:
* **`Shallow()`:** Tar emot `hiddenLayer` samt `outputLayer` (nätverkets dolda lager respektive 
  utgångslager, `ml::dense_layer::Interface&`), samt `trainInput` och `trainOutput` 
  (skrivskyddade, tvådimensionella flyttalsvektorer med träningsdatans in- och utdata). Ska 
  markeras `explicit` samt `noexcept`.

Konstruktorn deklareras enbart här; den implementeras i uppgift 9.

---

### 7. Borttagna konstruktorer och operatorer
Radera klassens default-konstruktor, kopierings- och förflyttningskonstruktorer samt tillhörande operatorer.

---

### 8. Övriga metoder - deklaration
Lägg till följande publika metod i `Shallow`:
* **`train(epochCount, learningRate = 0.01)`:** Tränar nätverket (implementeras i sin helhet i 
  **L07**). `epochCount`: antal epoker att träna (osignerat heltal). `learningRate`: lärhastighet 
  (flyttal). Returnerar `true` om träning genomfördes, annars `false`. Ska markeras `noexcept`.

Ni får gärna lägga till fler (privata) metoder vid behov.

---

### 9. Konstruktor och prediktion
Implementera följande i `source/neural_network/shallow.cpp`:

**Konstruktorn:**
* Initiera samtliga medlemsvariabler enligt beskrivningen ovan.

**Metoden `predict()`:**
* Genomför feedforward genom hela nätverket:
    1. Anropa `myHiddenLayer.feedforward(input)` med given indata.
    2. Anropa `myOutputLayer.feedforward(myHiddenLayer.output())` med det dolda lagrets output som indata.
* Returnera `myOutputLayer.output()` (en referens - ingen egen lagringsvariabel behövs i `Shallow`).

---

### 10. Träningsmetod (placeholder)
Implementera en tillfällig version av `train()` i `source/neural_network/shallow.cpp` som enbart returnerar `true`. Den fullständiga implementationen (feedforward, backpropagation samt optimering för varje träningsuppsättning och epok) genomförs under **L07**.

---

### 11. Kompilering och test
Skriv en `main`-funktion i `main.cpp` som:
* Skapar en `ml::dense_layer::Stub`-instans för det dolda lagret samt en för utgångslagret (t.ex. 3 noder/2 vikter per nod respektive 1 nod/3 vikter per nod - antalet vikter i utgångslagret ska matcha antalet noder i det dolda lagret).
* Skapar en `ml::neural_network::Shallow`-instans utifrån dessa två lager samt valfri träningsdata (t.ex. ett 2-bitars XOR-mönster).
* Genomför en prediktion för varje träningsuppsättnings indata, och skriver ut indatan samt den predikterade utdatan i terminalen.

Kompilera och testkör programmet. Ni ska få följande utskrift (dense-lagren är fortfarande stubbar, så prediktionen är alltid 0.5):

```
--------------------------------------------------------------------------------
Input: 0 0, predicted output: 0.5
Input: 0 1, predicted output: 0.5
Input: 1 0, predicted output: 0.5
Input: 1 1, predicted output: 0.5
--------------------------------------------------------------------------------
```

---

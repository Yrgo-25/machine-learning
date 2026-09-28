# Bilaga B - Övningsuppgift: Dense-lager (del II)
Ni ska färdigställa klassen `Dense` från **L08** genom att implementera metoderna `feedforward()`, `backpropagate()` samt `optimize()`. Se [bilaga A](./a_math_to_code.md) för en översikt av hur 
matematiken från **L05** motsvaras av koden nedan.

---

### 1. Kom igång
Har ni inte redan hämtat testramverket under en tidigare lektion, gör det först. Kör följande
kommando en gång, i repots rotkatalog:

```bash
git submodule update --init --recursive
```

Katalogen [`exercises`](../exercises) innehåller lösningsförslaget från **L06–L08**, samt en
testsvit i `exercises/test` (se avsnitt 8):
* `include/ml/dense_layer/`: dense-lagrets interface samt stubbklassen `Stub` (**L06**).
* `include/ml/neural_network/` samt `source/ml/neural_network/`: nätverkets interface samt klassen
  `Shallow` (**L07**).
* `include/ml/dense_layer/dense.h` samt `source/ml/dense_layer/dense.cpp`: klassen `Dense`
  (**L08**), där `feedforward()`, `backpropagate()` samt `optimize()` än så länge enbart
  kontrollerar sina argument.
* `include/ml/types.h`, `source/main.cpp` samt `Makefile`.

Skriv er kod där, eller i er befintliga `ml`-kodbas. Är er egen kod från **L06–L08** inte färdig,
utgå från lösningsförslaget, så att ni kan lägga hela lektionen på de tre metoderna.

---

### 2. Hjälpfunktioner
I filen `source/ml/dense_layer/dense.cpp`, skapa en anonym namnrymd. I denna namnrymd, definiera 
följande hjälpfunktioner:
* `initRandom()`: Funktion för att initiera slumptalsgeneratorn en gång.
    * **Implementation:**
        * Lägg till en statisk lokal variabel döpt `initialized` med initialvärde `false`. 
        Eftersom variabeln är statisk behåller den sitt värde mellan funktionsanrop, vilket gör att 
        vi bara initierar slumptalsgeneratorn en gång.
        * Om `initialized == true`, avsluta funktionen tidigt (med `return`).
        * Initiera slumptalsgeneratorn med aktuell tid genom att anropa `std::srand(std::time(nullptr))`. För att åstadkomma detta, inkludera `<cstdlib>` samt `<ctime>`.
        * Efter initieringen, sätt `initialized = true` så att initiering inte sker nästa gång funktionen anropas.
    * Ska markeras `noexcept`.

* `randomStartVal()`: Funktion för att generera och returnera ett slumptal i intervallet `[-1.0, 1.0]`, slutet i båda ändar.
    * **Implementation:**
        * Generera först ett slumptal i intervallet `[0.0, 1.0]` genom att kalla på `std::rand()`, som genererar ett slumptal i intervallet `[0, RAND_MAX]`, och dividera med `RAND_MAX`.
        * En av operatorerna måste omvandlas till ett flyttal för att inte heltalsdivision ska ske, exempelvis `static_cast<double>(RAND_MAX)`.
        * Skala sedan om talet till intervallet `[-1.0, 1.0]` genom att multiplicera med `2.0` och subtrahera `1.0`. Ett utfall på `0` ger då `-1.0`, och ett utfall på `RAND_MAX` ger `1.0`.
    * Ska markeras `noexcept`.
    * **OBS!** Startvärdena måste kunna bli negativa. Startar samtliga vikter och bias positiva, ger
      insignalen `{1, 1}` alltid den största viktade summan i varje ReLU-nod, och nätverket kan då
      inte lära sig XOR-mönstret, oavsett antal noder, epoker eller lärhastighet.

* `actFuncOutput()`: Funktion för att beräkna och returnera utdata (flyttal) ur en given aktiveringsfunktion.
    * **Tar emot:**
        * `actFunc`: Aktiveringsfunktionen som ska användas (av typen `ActFunc`).
        * `input`: Indatavärde till aktiveringsfunktionen (flyttal).
    * **Implementation:**
        * Använd en switch-sats för att beräkna utdatan beroende på angiven aktiveringsfunktion:
            * `ActFunc::Relu`: Returnera `input` om `input > 0.0`, annars `0.0`.
            * `ActFunc::Tanh`: Returnera `std::tanh(input)` (kräver `#include <cmath>`).
            * `ActFunc::None`: Returnera `input` oförändrad.
            * Default-fall: Skriv ut felmeddelandet `"Invalid activation function!"` och returnera `0.0`.
    * Ska markeras `noexcept`.

* `actFuncDelta()`: Funktion för att beräkna och returnera derivatan (flyttal) av en given aktiveringsfunktion.
    * **Tar emot:**
        * `actFunc`: Aktiveringsfunktionen som ska användas (av typen `ActFunc`).
        * `input`: Indatavärde till aktiveringsfunktionen (flyttal).
    * **Implementation:**
        * Använd en switch-sats för att beräkna derivatan beroende på angiven aktiveringsfunktion:
            * `ActFunc::Relu`: Returnera `1.0` om `input > 0.0`, annars `0.0`.
            * `ActFunc::Tanh`: Beräkna `const auto tanhOutput = std::tanh(input)` och returnera `1.0 - tanhOutput * tanhOutput`.
            * `ActFunc::None`: Returnera `1.0`.
            * Default-fall: Skriv ut felmeddelandet `"Invalid activation function!"` och returnera `0.0`.
    * Ska markeras `noexcept`.

---

### 3. Randomisering av bias och vikter
Randomisera samtliga biasvärden och vikter:
* I konstruktorn, anropa först `initRandom()` för att initiera slumptalsgeneratorn.
* Iterera genom samtliga noder i lagret med en for-loop: `for (std::size_t i{}; i < nodeCount; ++i)`.
* För varje nod `i`:
    * Tilldela ett slumptal till dess biasvärde genom att anropa `randomStartVal()`: `myBias[i] = randomStartVal()`.
    * Iterera genom nodens vikter med en nästlad for-loop: `for (std::size_t j{}; j < weightCount; ++j)`.
    * För varje vikt `j`:
        * Tilldela ett slumptal till vikten genom att anropa `randomStartVal()`: `myWeights[i][j] = randomStartVal()`.

---

### 4. Metoden `feedforward()`
**Indatakontroll:**
* Behåll kontrollen från **L08** (`input.size() == weightCount()`) oförändrad, och lägg till
  beräkningen nedan efter den.

**Beräkning för varje nod:**
* Iterera genom samtliga noder i lagret med en for-loop: `for (std::size_t i{}; i < nodeCount(); ++i)`.
* För varje nod `i`, beräkna den viktade summan:
    1. Starta med nodens bias-värde: `auto sum{myBias[i]}`.
    2. Lägg till varje vikt multiplicerat med motsvarande input: `for (std::size_t j{}; j < weightCount(); ++j)` där `sum += myWeights[i][j] * input[j]`.
* Spara den viktade summan innan aktiveringsfunktionen appliceras: `myPreActOutput[i] = sum`.
  Detta värde behövs av `backpropagate()` nedan för att beräkna aktiveringsfunktionens derivata korrekt.
* Applicera aktiveringsfunktionen på summan: `myOutput[i] = actFuncOutput(myActFunc, sum)`.

**Returvärde:** `true` när samtliga noder har beräknats.

---

### 5. Metoden `backpropagate()` (utgångslager)
Implementera `backpropagate()` för utgångslager (med referensvärden):

**Indatakontroll:**
* Behåll kontrollen från **L08** (`reference.size() == nodeCount()`) oförändrad, och lägg till
  felberäkningen nedan efter den.

**Felberäkning för varje nod:**
* Iterera genom samtliga noder i lagret: `for (std::size_t i{}; i < nodeCount(); ++i)`.
* För varje nod `i`:
    1. Beräkna det råa felet: `const auto err{reference[i] - myOutput[i]}`.
    2. Beräkna gradientfelet: `myError[i] = err * actFuncDelta(myActFunc, myPreActOutput[i])`.
       **OBS!** Använd `myPreActOutput[i]` (den viktade summan innan aktiveringsfunktionen
       applicerades i `feedforward()`), inte `myOutput[i]`. `actFuncDelta()` förväntar sig
       aktiveringsfunktionens *indata*, inte dess utdata - annars blir derivatan felaktig för
       `ActFunc::Tanh` (fungerar av en slump för `ActFunc::Relu`).

**Returvärde:** `true` när samtliga noders fel har beräknats.

---

### 6. Metoden `backpropagate()` (dolt lager)
Implementera `backpropagate()` för dolda lager (med fel och vikter från nästa lager):

**Indatakontroll:**
* Behåll kontrollen från **L08** (`nextLayer.weightCount() == nodeCount()`) oförändrad, och lägg
  till felberäkningen nedan efter den.

**Felberäkning för varje nod:**
* Iterera genom samtliga noder i detta lager: `for (std::size_t i{}; i < nodeCount(); ++i)`.
* För varje nod `i`:
    1. Initiera variabel som lagrar det beräknade råa felet: `double err{}`.
    2. Summera samtliga fel från nästa lager: `for (std::size_t j{}; j < nextLayer.nodeCount(); ++j)`
        * `err += nextLayer.error()[j] * nextLayer.weights()[j][i]`.
    3. Beräkna gradientfelet: `myError[i] = err * actFuncDelta(myActFunc, myPreActOutput[i])`
       (se OBS-rutan i föregående avsnitt om varför `myPreActOutput[i]` används i stället
       för `myOutput[i]`).

**Returvärde:** `true` när samtliga noders fel har beräknats.

---

### 7. Metoden `optimize()`
**Indatakontroll:**
* Behåll kontrollerna från **L08** av lärhastigheten och inputstorleken oförändrade, och lägg till
  parameteruppdateringen nedan efter dem.

**Parameteruppdatering för varje nod:**
* Iterera genom samtliga noder i lagret: `for (std::size_t i{}; i < nodeCount(); ++i)`.
* För varje nod `i`:
    1. Uppdatera bias: `myBias[i] += myError[i] * learningRate`.
    2. Uppdatera alla vikter: `for (std::size_t j{}; j < weightCount(); ++j)`
        * `myWeights[i][j] += myError[i] * learningRate * input[j]`

**Returvärde:** `true` när samtliga noders bias och vikter har uppdaterats.

---

### 8. Enhetstester
Kontrollera er implementation mot testsviten i `exercises/test`. Se testsvitens
[README](../exercises/test/README.md) för detaljer. Testsviten innehåller även testerna från
**L06–L08**, så det räcker att köra denna.

1. Bygg och kör testsviten via kommandot `make` i katalogen `exercises/test`. Testramverket måste
   ha hämtats först, se avsnitt 1.
    * Skriver ni er kod i er egen `ml`-kodbas i stället för i `exercises`, ange sökvägen till den
      via `make ML_DIR=<sökväg till er ml-katalog>`.
2. Åtgärda eventuella fel och kör testsviten igen, tills samtliga testfall går igenom.

Testsviten kompilerar inte förrän `ml/dense_layer/dense.h` samt `source/ml/dense_layer/dense.cpp`
finns och deklarerar samtliga metoder som testerna anropar. Läs det första kompileringsfelet; det
anger oftast vilken metod som saknas eller har fel signatur.

Testerna jämför mot värden som räknas fram ur lagrets egna vikter, eftersom startvärdena slumpas.
Testfallet `NetworkLearnsXorPattern` tränar slutligen ett helt nätverk av två dense-lager på
XOR-mönstret och kontrollerar att det faktiskt lär sig.

Felmeddelanden som `Input dimension mismatch` och `Invalid learning rate` skrivs ut av de testfall
som medvetet anropar lagret med felaktiga argument. De betyder alltså inte att något test har
misslyckats.

---

### 9. Experimentera med nätverkets parametrar
När samtliga testfall går igenom, kör demoprogrammet i `exercises/source/main.cpp` via kommandot
`make` i katalogen `exercises`. Programmet tränar ett nätverk på XOR-mönstret och skriver ut
prediktionerna före och efter träning. Med startvärdena (3 dolda noder, lärhastighet `0.01` och
100 epoker) lär sig nätverket sannolikt inte mönstret.

Justera därför följande tre parametrar, en i taget, och studera hur prediktionerna påverkas:
* **Lärhastigheten** `learningRate` i `trainAndTest()`, exempelvis `0.001`, `0.01`, `0.05`, `0.1`
  och `0.5`.
* **Antalet epoker** `epochCount` i `trainAndTest()`, exempelvis `100`, `1000`, `10000` och
  `100000`.
* **Antalet dolda noder** `hiddenCount` i `main()`, exempelvis `3`, `6` och `10`.

Kör programmet flera gånger för varje inställning. Startvärdena slumpas, så samma inställning kan
lyckas vid en körning och misslyckas vid nästa. Notera att slumptalsgeneratorn seedas med aktuell tid
i sekunder; vänta därför minst en sekund mellan körningarna.

Fundera på följande frågor:
1. Hur hänger lärhastigheten och antalet epoker ihop? Vad händer om ni halverar lärhastigheten,
   men behåller antalet epoker?
2. Vad händer när lärhastigheten blir för hög? Hjälper det då att öka antalet epoker?
3. Hur påverkar antalet dolda noder hur ofta träningen lyckas?

**Förslag:** Använd till sist 10 dolda noder, lärhastigheten `0.05` och 10 000 epoker. Med dessa
inställningar bör nätverket lära sig XOR-mönstret vid de allra flesta körningarna, med
prediktioner nära `0` respektive `1`.

---

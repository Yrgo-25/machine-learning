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

Katalogen [`exercises`](../exercises) innehåller koden från **L06–L08**, samt en testsvit i
`exercises/test` (se avsnitt 8). Skriv er kod där, eller i er befintliga `ml`-kodbas.

---

### 2. Hjälpfunktioner
I filen `source/dense_layer/dense.cpp`, skapa en anonym namnrymd. I denna namnrymd, definiera 
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

* `randomStartVal()`: Funktion för att generera och returnera ett slumptal i intervallet `[0.0, 1.0]`, slutet i båda ändar.
    * **Implementation:**
        * Generera ett slumptal inom intervallet (0.0, 1.0) genom att kalla på `std::rand()`, som genererar ett slumptal mellan (0, RAND_MAX), och dividera med RAND_MAX.
        * En av operatorerna måste omvandlas till ett flyttal för att inte heltalsdivision ska ske, exempelvis `static_cast<double>(RAND_MAX)`.
    * Ska markeras `noexcept`.

* `actFuncOutput()`: Funktion för att beräkna och returnera utdata (flyttal) ur en given aktiveringsfunktion.
    * **Tar emot:**
        * `actFunc`: Aktiveringsfunktionen som ska användas (av typen `ActFunc`).
        * `input`: Indatavärde till aktiveringsfunktionen (flyttal).
    * **Implementation:**
        * Använd en switch-sats för att beräkna utdatan beroende på angiven aktiveringsfunktion:
            * `ActFunc::Relu`: Returnera `input` om `input > 0.0`, annars `0.0`.
            * `ActFunc::Tanh`: Returnera `std::tanh(input)` (kräver `#include <cmath>`).
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
* Kontrollera att dimensionerna på given input matchar antalet vikter per nod i lagret (`input.size() == weightCount()`).
* Om dimensionerna inte matchar: skriv ut felmeddelandet `"Input dimension mismatch: expected X, actual: Y!"` och returnera `false` utan att beräkna någonting.

**Beräkning för varje nod:**
* Iterera genom samtliga noder i lagret med en for-loop: `for (std::size_t i{}; i < nodeCount(); ++i)`.
* För varje nod `i`, beräkna den viktade summan:
    1. Starta med nodens bias-värde: `auto sum{myBias[i]}`.
    2. Lägg till varje vikt multiplicerat med motsvarande input: `for (std::size_t j{}; j < weightCount(); ++j)` där `sum += myWeights[i][j] * input[j]`.
* Spara den viktade summan innan aktiveringsfunktionen appliceras: `myPreActivationOutput[i] = sum`.
  Detta värde behövs av `backpropagate()` nedan för att beräkna aktiveringsfunktionens derivata korrekt.
* Applicera aktiveringsfunktionen på summan: `myOutput[i] = actFuncOutput(myActFunc, sum)`.

**Returvärde:** `true` när samtliga noder har beräknats.

---

### 5. Metoden `backpropagate()` (utgångslager)
Implementera `backpropagate()` för utgångslager (med referensvärden):

**Indatakontroll:**
* Kontrollera att dimensionerna på referensvektorns storlek matchar antalet noder (`reference.size() == nodeCount()`).
* Om dimensionerna inte matchar: skriv ut felmeddelandet `"Output dimension mismatch: expected X, actual: Y!"` och returnera `false` utan att beräkna någonting.

**Felberäkning för varje nod:**
* Iterera genom samtliga noder i lagret: `for (std::size_t i{}; i < nodeCount(); ++i)`.
* För varje nod `i`:
    1. Beräkna det råa felet: `const auto err{reference[i] - myOutput[i]}`.
    2. Beräkna gradientfelet: `myError[i] = err * actFuncDelta(myActFunc, myPreActivationOutput[i])`.
       **OBS!** Använd `myPreActivationOutput[i]` (den viktade summan innan aktiveringsfunktionen
       applicerades i `feedforward()`), inte `myOutput[i]`. `actFuncDelta()` förväntar sig
       aktiveringsfunktionens *indata*, inte dess utdata - annars blir derivatan felaktig för
       `ActFunc::Tanh` (fungerar av en slump för `ActFunc::Relu`).

**Returvärde:** `true` när samtliga noders fel har beräknats.

---

### 6. Metoden `backpropagate()` (dolt lager)
Implementera `backpropagate()` för dolda lager (med fel och vikter från nästa lager):

**Indatakontroll:**
* Kontrollera att nästa lagers viktantal matchar detta lagers nodantal (`nextLayer.weightCount() == nodeCount()`).
* Om dimensionerna inte matchar: skriv ut felmeddelandet `"Layer dimension mismatch: expected X, actual: Y!"` och returnera `false` utan att beräkna någonting.

**Felberäkning för varje nod:**
* Iterera genom samtliga noder i detta lager: `for (std::size_t i{}; i < nodeCount(); ++i)`.
* För varje nod `i`:
    1. Initiera variabel som lagrar det beräknade råa felet: `double err{}`.
    2. Summera samtliga fel från nästa lager: `for (std::size_t j{}; j < nextLayer.nodeCount(); ++j)`
        * `err += nextLayer.error()[j] * nextLayer.weights()[j][i]`.
    3. Beräkna gradientfelet: `myError[i] = err * actFuncDelta(myActFunc, myPreActivationOutput[i])`
       (se OBS-rutan i föregående avsnitt om varför `myPreActivationOutput[i]` används i stället
       för `myOutput[i]`).

**Returvärde:** `true` när samtliga noders fel har beräknats.

---

### 7. Metoden `optimize()`
**Indatakontroll:**
* Kontrollera att lärhastigheten ligger inom intervallet `(0.0, 1.0)`, samma intervall som stubben kontrollerade i **L06**.
* Om lärhastigheten är ogiltig: skriv ut felmeddelandet `"Invalid learning rate X!"` och returnera `false` utan att uppdatera någonting.
* Kontrollera att inputstorleken matchar antalet vikter per nod (`input.size() == weightCount()`).
* Om dimensionerna inte matchar: skriv ut felmeddelandet `"Input dimension mismatch: expected X, actual: Y!"` och returnera `false` utan att uppdatera någonting.

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

Testsviten kompilerar inte förrän `ml/dense_layer/dense.h` samt `source/dense_layer/dense.cpp`
finns och deklarerar samtliga metoder som testerna anropar. Läs det första kompileringsfelet; det
anger oftast vilken metod som saknas eller har fel signatur.

Testerna jämför mot värden som räknas fram ur lagrets egna vikter, eftersom startvärdena slumpas.
Testfallet `NetworkLearnsXorPattern` tränar slutligen ett helt nätverk av två dense-lager på
XOR-mönstret och kontrollerar att det faktiskt lär sig.

Felmeddelanden som `Input dimension mismatch` och `Invalid learning rate` skrivs ut av de testfall
som medvetet anropar lagret med felaktiga argument. De betyder alltså inte att något test har
misslyckats.

---

# Bilaga B - Övningsuppgifter
Ni ska implementera ett interface samt en tillhörande klass för en linjär regressionsmodell i C++.

---

### 1. Katalogstruktur
Skapa en ny katalog döpt `ml` med följande katalogstruktur:

```
ml/
├── include/
│   └── ml/
│       ├── lin_reg/
│       │   ├── fixed.h
│       │   └── interface.h
│       └── types.h
├── source/
│   ├── ml/
│   │   └── lin_reg/
│   │       └── fixed.cpp
│   └── main.cpp
└── Makefile
```

--- 

### 2. Makefile
Skriv en makefil som kompilerar och bygger ett program med ovanstående filstruktur via kommandot `make`.

Testkör programmet genom att lägga till en tom main-funktion i `main.cpp`.

---

### 3. main.cpp
Klistra in innehållet från [main.cpp](#maincpp) längst ned i detta dokument i motsvarande fil.

---

### 4. Matrix-alias
I headerfilen `ml/types.h`, lägg till en namnrymd döpt `ml`.

I denna namnrymd, implementera följande alias:
* `Matrix1d`: Alias för endimensionella flyttalsvektorer (`std::vector<double>`).
* `Matrix2d`: Alias för tvådimensionella flyttalsvektorer (`std::vector<Matrix1d>`).
* `Matrix3d`: Alias för tredimensionella flyttalsvektorer (`std::vector<Matrix2d>`).

---

### 5. Interface
I headerfilen `ml/lin_reg/interface.h`, lägg till en namnrymd döpt `ml::lin_reg`. 

I denna namnrymd, implementera ett interface döpt `Interface` med följande publika metoder:
* **`~Interface()`:**
    * Interfacets destruktor. 
    * Ska markeras `default` och `noexcept`.
* **`predict()`:**
    * Ska genomföra en prediktion. 
    * Ska ha ett flyttal döpt `input` som enda ingående argument.
    * Ska returnera det predikterade värdet som ett flyttal.
    * Ska vara rent virtuell (`= 0`).
    * Ska markeras `const` samt `noexcept`.

---

### 6. Underklass: Fixed
I headerfilen `ml/lin_reg/fixed.h`, lägg till namnrymden `ml::lin_reg`. Utgå från interfacet och gör om det till en underklass:
1. Kopiera in innehållet från `interface.h`, inklusive `#pragma once` samt namnrymden `ml::lin_reg`.
2. Inkludera `ml/lin_reg/interface.h`, så att basklassen är känd.
3. Döp om klassen till `Fixed` och låt den ärva `Interface` via publikt arv. Klassen ska inte kunna ärvas vidare, dvs. den ska markeras `final`.
4. Ta bort `virtual` samt `= 0` från metoderna; markera dem `override` i stället.

Klassen döps till `Fixed` eftersom den tränas med en fast (*fixed*) lärhastighet. En variant med adaptiv lärhastighet, `Adaptive`, kommer ni att implementera i grupp under **P03**.

Efter omvandlingen ska klassen inneha följande publika metoder:
* **`~Fixed()`:** 
    * Destruktor som överlagrar interfacets destruktor.
    * Ska markeras `default`, `noexcept` samt `override`.
* **`predict()`:** 
    * Överlagring av motsvarande metod i interfacet.
    * Ska markeras `const`, `noexcept` samt `override`.

---

### 7. Privata medlemsvariabler
Lägg till följande privata medlemsvariabler i `Fixed`:
* **`myTrainInput`:** 
    * Referens till träningsdatans indatapunkter.
    * Initieras via konstruktorn.
    * Ska markeras `const`.
* **`myTrainOutput`:** 
    * Referens till träningsdatans utdatapunkter.
    * Initieras via konstruktorn.
    * Ska markeras `const`.
* **`myBias`:** 
    * Modellens biasvärde (m-värdet i `y = kx + m`).
    * Ska vara av flyttalstyp.
* **`myWeight`:** 
    * Modellens viktvärde (k-värdet i `y = kx + m`).
    * Ska vara av flyttalstyp.
* **`mySetCount`:** 
    * Antalet fullständiga träningsuppsättningar, dvs. `std::min(myTrainInput.size(), myTrainOutput.size())` (`std::min` finns i `<algorithm>`).
    * Träningsdata som innehåller fler indata än utdata, eller tvärtom, har inga fullständiga uppsättningar bortom den kortaste av de två vektorerna, så överskottsvärdena är oanvändbara.
    * Initieras i konstruktorn.
    * Ska vara ett osignerat heltal (`std::size_t`, från `<cstddef>`) och markeras `const`.

Medlemsvariablerna läggs till före konstruktorn, så att ni vet exakt vad konstruktorn ska initiera.

---

### 8. Konstruktor - deklaration
Lägg till klassens enda implementerade konstruktor som publik metod:
* **`Fixed()`:** 
    * Ska ha följande ingående argument:
        * `trainInput`: Referens till en skrivskyddad vektor med flyttal (träningsdata, indata).
        * `trainOutput`: Referens till en skrivskyddad vektor med flyttal (träningsdata, utdata).
    * Ska markeras `explicit` samt `noexcept`.

Konstruktorn deklareras enbart här; den implementeras i uppgift 11.

---

### 9. Borttagna konstruktorer och operatorer
Radera klassens defaultkonstruktor, kopierings- och förflyttningskonstruktorer samt tillhörande operatorer.

---

### 10. Övriga metoder - deklaration
Lägg till följande publika metod i `Fixed`:
* **`train()`:** 
    * Tränar modellen med ingående argument:
        * `epochCount`: Antal epoker att träna (osignerat heltal).
        * `learningRate`: Lärhastighet som flyttal. Defaultvärde: `0.01` (1 %).
    * Returnerar `false` om `epochCount` är 0, eller om `learningRate` ligger utanför intervallet `(0.0, 1.0)`. Annars returneras `true` efter genomförd träning.
    * Ska markeras `noexcept`.

Lägg också till följande privata metod:
* **`optimize()`:**
    * Ingående argument:
        * `input`: Indatavärde (`x`) som ett flyttal.
        * `output`: Referensvärde (`y_ref`) som ett flyttal.
        * `learningRate`: Lärhastigheten (`LR`) som ett flyttal.
    * Returnerar ingenting.
    * Ska markeras `noexcept`.

---

### 11. Konstruktor - implementation
Implementera konstruktorn i `source/ml/lin_reg/fixed.cpp`:
* Initiera alla medlemsvariabler. 
* Sätt `mySetCount` till antalet fullständiga träningsuppsättningar, dvs. det minsta av de två vektorernas storlekar.
* `myBias` och `myWeight` ska båda initieras till `0.0`, så att modellen predikterar `0` för varje indata tills den har tränats.
* Skriv ut ett felmeddelande till `stderr` och anropa `std::terminate()` (från `<exception>`) om `mySetCount` är 0. En modell utan träningsdata kan aldrig tränas, och eftersom konstruktorn inte kan returnera någon felkod finns det inget sätt att rapportera problemet till anroparen.

---

### 12. Prediktion
Implementera metoden `predict()` i `source/ml/lin_reg/fixed.cpp`.

Returnera en prediktion enligt formeln:

$$y = kx + m$$

där:
* $y$ är prediktionen som returneras.
* $k$ är modellens viktvärde (`myWeight`).
* $x$ är indatan (`input`).
* $m$ är modellens biasvärde (`myBias`).

---

### 13. Träning
Implementera metoden `train()` i `source/ml/lin_reg/fixed.cpp`:
* Returnera `false` direkt om `epochCount` är 0, eller om `learningRate` ligger utanför intervallet `(0.0, 1.0)`:
    * En lärhastighet på `0.0` eller mindre kan aldrig förbättra modellen.
    * En lärhastighet på `1.0` eller mer korrigerar med minst hela felet i varje steg, vilket får träningen att oscillera eller divergera i stället för att konvergera.
* Till skillnad från konstruktorn kan `train()` rapportera ogiltiga argument via sitt returvärde, så här anropas inte `std::terminate()`.
* Avsaknad av träningsdata hanteras redan av konstruktorn, så `train()` behöver inte kontrollera det.
* Träna modellen givet antal epoker.
* Under varje epok, iterera genom alla `mySetCount` träningsuppsättningar. Genomför optimering genom att anropa metoden `optimize()` med aktuell träningsuppsättning (erhålls från vektorerna, förslagsvis via index).
* Returnera `true` efter genomförd träning.

---

### 14. Optimering
Implementera metoden `optimize()` i `source/ml/lin_reg/fixed.cpp`:
* Om `input == 0`: sätt `myBias = output` direkt och returnera, utan att röra `myWeight`. Eftersom `y = k * 0 + m = m` *är* referensvärdet biasvärdet, och inget värde på `k` påverkar en prediktion gjord med indatan noll.
* Annars:
    * Beräkna en prediktion med given indata: `prediction = myWeight * input + myBias`.
    * Beräkna felet som differensen mellan referensvärdet och prediktionen: `error = output - prediction`.
    * Justera modellens parametrar utefter felet:
        * `myBias   = myBias   + error * learningRate`
        * `myWeight = myWeight + error * learningRate * input`

---

### 15. Kompilering och test
Kompilera programmet och se till att du får följande utskrift:

```
--------------------------------------------------------------------------------
Input: 0, predicted output: 2
Input: 1, predicted output: 4
Input: 2, predicted output: 6
Input: 3, predicted output: 8
Input: 4, predicted output: 10
--------------------------------------------------------------------------------
```

---

## main.cpp

```cpp
/**
 * @brief Linear regression demonstration.
 */
#include <cstdint>
#include <cstdio>

#include "ml/lin_reg/fixed.h"
#include "ml/types.h"

namespace
{
/**
 * @brief Predict with the given linear regression model.
 *
 * @param[in] linReg Linear regression model to predict with.
 * @param[in] inputData Input data to predict with.
 */
void predict(const ml::lin_reg::Interface& linReg, const ml::Matrix1d& inputData) noexcept
{
    // Check the number of input sets, terminate if missing.
    if (inputData.empty())
    {
        std::printf("No input data!\n");
        return;
    }

    // Perform prediction with each input set, print the result in the terminal.
    std::printf("--------------------------------------------------------------------------------\n");
    for (const auto& input : inputData)
    {
        const auto prediction = linReg.predict(input);
       std::printf("Input: %g, predicted output: %g\n", input, prediction);
    }
    std::printf("--------------------------------------------------------------------------------\n\n");
}
} // namespace

/**
 * @brief Train and predict with a linear regression model.
 *
 * @return 0 on success, or -1 on failure.
 */
int main()
{
    constexpr std::uint32_t epochCount{1000U};
    constexpr double learningRate{0.1};

    // Create linear regression model to predict y = 2x + 2.
    const ml::Matrix1d trainInput{0.0, 1.0, 2.0, 3.0, 4.0};
    const ml::Matrix1d trainOutput{2.0, 4.0, 6.0, 8.0, 10.0};
    ml::lin_reg::Fixed linReg{trainInput, trainOutput};

    // Train the model, terminate on failure.
    if (!linReg.train(epochCount, learningRate))
    {
        std::printf("Training failed!\n");
        return -1;
    }

    // Perform prediction with all input sets.
    predict(linReg, trainInput);
    return 0;
}
```

---

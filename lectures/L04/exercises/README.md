# L04 - Lösningsförslag: Linjär regression med precisionsberäkning i C++
Linjär regressionsmodell som predikterar enligt `y = kx + m` och tränas med en fast lärhastighet, nu med beräkning av precisionen samt tidig avstängning av träningen så snart modellen är tillräckligt tränad.

Koden bygger vidare på [lösningsförslaget från **L03**](../../L03/exercises/README.md). Interfacet `ml::lin_reg::Interface` är fortsatt oförändrat; all ny funktionalitet ligger i underklassen `ml::lin_reg::Fixed`.

---

## Ändringar jämfört med L03

| Uppgift | Ändring |
|---|---|
| 1 | Ny privat metod `computePrecision()`, som beräknar precisionen utifrån medelabsolutfelet. |
| 2 | `train()` tar emot ett tröskelvärde `precisionThreshold` och avbryter träningen då precisionen når det. |
| 3 | Antalet epoker i `main.cpp` höjt till 1000; det är nu precisionen, inte epokantalet, som avgör när träningen avslutas. |

Inga nya medlemsvariabler tillkommer; precisionen beräknas ur träningsdatan och modellens nuvarande parametrar varje gång den efterfrågas, och behöver därför inte lagras.

---

## Struktur
```
exercises/
├── Makefile
├── include/
│   └── ml/
│       ├── lin_reg/
│       │   ├── fixed.h        # Klassen Fixed, deklaration.
│       │   └── interface.h    # Interfacet Interface.
│       └── types.h            # Matrisalias: Matrix1d, Matrix2d, Matrix3d, MatrixU32.
└── source/
    ├── main.cpp               # Demo: tränar en modell för y = 2x + 2.
    └── ml/
        └── lin_reg/
            └── fixed.cpp      # Klassen Fixed, implementation.
```

---

## Precision som mått
Precisionen beräknas som `1.0 - MAE`, där MAE (*mean absolute error*) är det genomsnittliga absolutfelet över samtliga träningsuppsättningar:
1. Prediktera för varje träningsuppsättning och beräkna det absoluta felet `|yref - yp|`.
2. Summera samtliga absoluta fel och dividera med antalet träningsuppsättningar; resultatet är MAE.
3. Precisionen är `1.0 - MAE`, ett värde som närmar sig 1.0 ju bättre modellen predikterar.

Måttet ska inte förväxlas med *precision* såsom begreppet används vid klassificering, där precision anger hur stor andel av de positiva prediktionerna som var korrekta. Det som beräknas här är ett passningsmått för en regressionsmodell, med två egenskaper värda att känna till:
* **Skalberoende:** Ett absolutfel på `0.1` betyder helt olika saker för utdata kring `1` respektive kring `1000`.
* **Kan bli negativt:** Ingenting hindrar att medelabsolutfelet överstiger `1.0`, vilket ger ett negativt "precisionsvärde".

Att läsa `0.999999` som "99,9999 % rätt" är alltså en bekväm förenkling snarare än en procentandel av någonting. Se [bilaga A](../appendix/a_precision.md) för en utförligare genomgång.

---

## Klassen `ml::lin_reg::Fixed`
Klassen är oförändrad från **L03** vad gäller medlemsvariabler; det som tillkommer är en privat metod samt ett nytt argument till `train()`.

### Metoder
* **`Fixed(trainIn, trainOut)`:** Oförändrad från **L03**.
* **`predict(input)`:** Returnerar `myWeight * input + myBias`.
* **`train(epochCount, learningRate, precisionThreshold)`:** Tränar modellen i högst `epochCount` epoker, och avslutar tidigt så snart precisionen når `precisionThreshold`.
* **`optimize(input, output, learningRate)`:** Privat metod som justerar `myBias` samt `myWeight` för en enskild träningsuppsättning.
* **`shuffle()`:** Privat metod som blandar `myTrainOrder` i slumpvis ordning.
* **`computePrecision()`:** Privat metod som beräknar modellens nuvarande precision. Markerad `[[nodiscard]]`, `const` samt `noexcept`; metoden läser modellens tillstånd utan att ändra det, och ett returvärde som kastas bort innebär att hela beräkningen var förgäves.

---

## Metoden `computePrecision()`
```cpp
double Fixed::computePrecision() const noexcept
{
    double sum{};

    // Iterate through all training sets.
    for (std::size_t i{}; i < myTrainOrder.size(); ++i)
    {
        const auto input      = myTrainIn[i];
        const auto output     = myTrainOut[i];
        const auto prediction = predict(input);

        // Compute the current error (absolute value, use std::abs() from <cmath>).
        const auto error = std::abs(output - prediction);
        sum += error; // sum = sum + error
    }
    // Compute average error, then return 1.0 - the average.
    // The return value will be very close to 1.0 if the model predicts well.
    const auto avgError = sum / myTrainOrder.size();
    return 1.0 - avgError;
}
```

Tre detaljer att notera:
* **Absolutbeloppet är nödvändigt.** Utan `std::abs()` skulle positiva och negativa fel ta ut varandra; en modell som predikterar 1.0 för högt i hälften av fallen och 1.0 för lågt i resten skulle då framstå som felfri. Kräver `<cmath>`.
* **Ordningen spelar ingen roll.** Metoden itererar sekventiellt via index i stället för via `myTrainOrder`, till skillnad från träningsloopen. Ett medelvärde är oberoende av summeringsordningen, och precisionen ska mätas på hela datamängden.
* **`myTrainOrder.size()` anger antalet träningsuppsättningar.** Samma resonemang som i **L03**: antalet lagras inte separat, eftersom indexvektorn redan innehåller exakt ett index per uppsättning.

---

## Tidig avstängning i `train()`
Metoden `train()` tar emot tröskelvärdet `precisionThreshold` med defaultvärdet `0.999999`, och kontrollerar det på samma sätt som lärhastigheten:

```cpp
// Check precision threshold, return false if outside range (0.0, 1.0).
if ((0.0 >= precisionThreshold) || (1.0 <= precisionThreshold)) { return false; }
```

Intervallet `(0.0, 1.0)` är öppet i båda ändar av två skäl:
* Ett tröskelvärde på `1.0` eller mer kan aldrig uppnås, eftersom precisionen är `1.0 - MAE` och medelabsolutfelet inte kan vara negativt. Träningen skulle alltid köra samtliga epoker.
* Ett tröskelvärde på `0.0` eller mindre godkänner en modell vars medelabsolutfel är `1.0` eller sämre, vilket omintetgör hela kontrollen.

Precis som i **L02** rapporterar `train()` ogiltiga argument via sitt returvärde; endast konstruktorn anropar `std::terminate()`, eftersom den inte kan returnera någon felkod till anroparen.

Själva kontrollen sker i slutet av varje epok:

```cpp
// Evaluate the precision every tenth epoch, skip the first one.
const auto evaluate = ((0U < epoch) && (0U == (epoch % evaluationInterval)));

if (evaluate)
{
    // Compute precision, stop training and print result if above the given threshold.
    const auto precision = computePrecision();

    if (precision >= precisionThreshold)
    {
        std::printf("Finished training with precision %g after %zu epochs!\n", precision, epoch);
        return true;
    }
}
```

* **`evaluationInterval`** är en namngiven konstant satt till 10. Precisionsberäkningen går igenom samtliga träningsuppsättningar, så att kontrollera var tionde epok i stället för varje epok skär bort merparten av den kostnaden över tusentals epoker. En naken `10` mitt i uttrycket hade sagt betydligt mindre om varför talet står där.
* **`0U < epoch`** hoppar över den allra första epoken. Uttrycket `epoch % evaluationInterval` är noll även då `epoch` är 0, och en enda genomgång av träningsdatan räcker sällan för att nå tröskelvärdet.
* **Villkoret bryts ut till `evaluate`.** Ett namngivet villkor läses lättare än samma uttryck inuti en `if`-sats.

---

## Träningsalgoritmen
`train()` utför följande per epok:
1. Blanda träningsordningen via `shuffle()`.
2. Iterera genom `myTrainOrder` och anropa `optimize()` för varje träningsuppsättning.
3. Var tionde epok, med undantag för den första: beräkna precisionen via `computePrecision()`.
4. Om precisionen når `precisionThreshold`: skriv ut resultatet och returnera `true` direkt.

Nås aldrig tröskelvärdet körs samtliga `epochCount` epoker, varpå `true` returneras; träningen har genomförts, om än utan att modellen blev tillräckligt bra. Argumenten var giltiga, vilket är vad returvärdet rapporterar.

---

## Bygg och kör
Bygg och kör programmet via följande kommando i denna katalog:

```bash
make
```

Övriga targets i [makefilen](./Makefile):

```bash
make build  # Bygger programmet utan att köra det.
make run    # Kör programmet utan att bygga om det.
make clean  # Tar bort den byggda binären.
```

`main.cpp` tränar en modell på `y = 2x + 2` i högst 1000 epoker med lärhastigheten 0.23, och predikterar sedan med varje indata i träningsdatan:

```
Finished training with precision 1 after 20 epochs!
----------------------------------------
Input: 0, output: 2
Input: 1, output: 4
Input: 2, output: 6
Input: 3, output: 8
Input: 4, output: 10
----------------------------------------
```

Antalet epoker varierar mellan körningarna, eftersom träningsordningen blandas slumpmässigt; med denna träningsdata nås tröskelvärdet oftast vid epok 20, ibland redan vid 10 och i enstaka fall först vid 30. Träningen avslutas alltså långt innan de 1000 epokerna är genomförda, vilket är hela poängen med precisionsberäkningen.

---

## Att notera
* Precisionen skrivs ut med `%g`, som avrundar till sex signifikanta siffror. En precision på `0.9999995` skrivs därför ut som `1`, trots att modellen inte är exakt perfekt. Använd exempelvis `%.10g` för att se hela värdet.
* Epokräknaren är nollindexerad, så utskriften anger indexet för den epok då tröskelvärdet nåddes, inte antalet genomförda epoker; dessa skiljer sig åt med ett. Skriv ut `epoch + 1U` för att få antalet.
* Eftersom precisionen bara kontrolleras var tionde epok kan modellen ha nått tröskelvärdet upp till nio epoker tidigare än utskriften anger. Det är den avvägning som görs för att slippa kontrollen varje epok.
* Tröskelvärdet `0.999999` är rimligt för denna träningsdata, där utdatan ligger i intervallet 2 till 10. För utdata i storleksordningen tusental hade samma tröskelvärde krävt en orimlig noggrannhet, eftersom måttet är skalberoende.
* `std::abs()` för flyttal deklareras i `<cmath>`. Utan den inkluderingen kan koden i värsta fall råka anropa heltalsvarianten från `<cstdlib>`, som trunkerar felet mot noll; så snart samtliga absolutfel understiger 1.0 ger beräkningen då precisionen exakt 1.0.

---

## Nästa steg
* **L05:** Introduktion till neurala nätverk: feedforward, backpropagation, gradient descent och aktiveringsfunktioner.
* **P03:** `ml::lin_reg::Adaptive`, en modell med adaptiv lärhastighet, i ett inbyggt system.

---

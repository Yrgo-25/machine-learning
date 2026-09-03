# Bilaga B - Övningsuppgifter
Ni ska bygga ut klassen `ml::lin_reg::Fixed` från L03 med precisionsberäkning.

---

### 1. Precisionsberäkning
Lägg till en privat metod döpt `computePrecision()` i klassen `Fixed`. Metoden ska beräkna precisionen för modellen givet träningsdatan:
* Beräkna medelabsolutfelet (MAE) för samtliga träningsuppsättningar:
    * För varje träningsuppsättning, beräkna det absoluta felet: `std::abs(output - predict(input))`.
    * Summera samtliga absoluta fel och dividera med antalet träningsuppsättningar.
* Returnera `1.0 - MAE` som ett flyttal.
* Ska markeras `[[nodiscard]]`, `const` samt `noexcept`.

Inkludera `<cmath>` i `fixed.cpp` för `std::abs`.

Metoden döps till `computePrecision()` snarare än `precision()`, så att namnet `precision` är ledigt för den lokala variabel i `train()` som håller det beräknade värdet. Hade metoden hetat `precision()` hade den lokala variabeln skuggat metoden, och ett efterföljande anrop hade krävt `this->precision()` för att kompilera.

---

### 2. Uppdatera träningsmetoden
Uppdatera metoden `train()` i `source/ml/lin_reg/fixed.cpp` så att träningen avslutas tidigt om precisionen når ett givet tröskelvärde:
* Lägg till ett ingående argument döpt `precisionThreshold` (ett flyttal som anger tröskelvärdet för precision). Defaultvärde: `0.999999` (99,9999 %).
* Returnera `false` om `precisionThreshold` faller utanför intervallet `(0.0, 1.0)`:
    * Ett tröskelvärde på `1.0` eller mer kan aldrig uppnås, eftersom precisionen är `1.0 - MAE` och medelabsolutfelet inte kan vara negativt. Träningen skulle alltid köra samtliga epoker.
    * Ett tröskelvärde på `0.0` eller mindre godkänner en modell vars medelabsolutfel är `1.0` eller sämre, vilket omintetgör hela kontrollen.
* Som i **L02**, `train()` rapporterar ogiltiga argument via sitt returvärde. Endast konstruktorn anropar `std::terminate()`, eftersom den inte kan returnera någon felkod till anroparen.
* Anropa `computePrecision()` var tionde epok, men hoppa över den allra första epoken:
    * Använd en namngiven konstant döpt `evaluationInterval` för intervallet i stället för en naken `10`.
    * Villkoret blir sant då epokräknaren är större än 0 och delbar med intervallet, dvs. `(0U < epoch) && (0U == (epoch % evaluationInterval))`.
    * Beräkningen går igenom samtliga träningsuppsättningar, så att kontrollera mer sällan än varje epok håller nere den kostnaden över tusentals epoker.
* Avbryt träningen och returnera `true` så snart precisionen når eller överstiger `precisionThreshold`, och skriv ut uppnådd precision samt hur många epoker det tog.

---

### 3. Kompilering och test
Kompilera och testkör programmet:
* Sätt antalet epoker i `main.cpp` till 1000, så att epokantalet inte längre är det som avgör när träningen avslutas.
* Modellen ska nu avbryta träningen så snart precisionen når tröskelvärdet, långt innan samtliga epoker är genomförda.
* Utskriften från `train()` visar uppnådd precision samt antalet epoker, och används för att verifiera att den tidiga avstängningen fungerar.

---

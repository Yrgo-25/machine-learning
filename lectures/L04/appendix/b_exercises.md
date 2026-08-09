# Bilaga B - Övningsuppgifter
Ni ska bygga ut klassen `ml::lin_reg::Fixed` från L03 med precisionsberäkning.

---

### 1. Precisionsberäkning
Lägg till en privat metod döpt `precision()` i klassen `Fixed`. Metoden ska beräkna precisionen för modellen givet träningsdatan:
* Beräkna medelabsolutfelet (MAE) för samtliga träningsuppsättningar:
    * För varje träningsuppsättning, beräkna det absoluta felet: `abs(output - predict(input))`.
    * Summera samtliga absoluta fel och dividera med antalet träningsuppsättningar.
* Returnera `1.0 - MAE` som ett flyttal.
* Ska markeras `[[nodiscard]]`, `const` samt `noexcept`.

Inkludera `<cmath>` i `fixed.cpp` för `std::abs`.

---

### 2. Uppdatera träningsmetoden
Uppdatera metoden `train()` i `source/ml/lin_reg/fixed.cpp` så att träningen avslutas tidigt om precisionen överstiger ett givet tröskelvärde:
* Lägg till ett ingående argument döpt `precisionThreshold` (ett flyttal som anger tröskelvärdet för precision). Defaultvärde: `0.999999` (99,9999 %). Observera att argumentet inte ska döpas till `precision`: inuti `train()` skuggar en parameter med det namnet metoden `precision()`, så ett anrop till den kompilerar inte längre (om inte `this->precision()` används).
* Returnera `false` om `precisionThreshold` faller utanför intervallet `(0.0, 1.0)`:
    * Ett tröskelvärde på `1.0` eller mer kan aldrig uppnås, eftersom precisionen är `1.0 - MAE` och medelabsolutfelet inte kan vara negativt. Träningen skulle alltid köra samtliga epoker.
    * Ett tröskelvärde på `0.0` eller mindre godkänner en modell vars medelabsolutfel är `1.0` eller sämre, vilket omintetgör hela kontrollen.
* Som i **L02**, `train()` rapporterar ogiltiga argument via sitt returvärde. Endast konstruktorn anropar `std::terminate()`, eftersom den inte kan returnera någon felkod till anroparen.
* Anropa `precision()` var tionde epok, med start vid den första. Beräkningen går igenom samtliga träningsuppsättningar, så att kontrollera mer sällan än varje epok håller nere den kostnaden över tusentals epoker. Använd en namngiven konstant för intervallet i stället för en naken `10`.
* Avbryt träningen och returnera `true` så snart precisionen når eller överstiger `precisionThreshold`, och skriv ut uppnådd precision samt antalet epoker det tog.
* Skriv ut uppnådd precision efter den sista epoken om tröskelvärdet aldrig nåddes, och returnera sedan `true`.

---

### 3. Kompilering och test
Kompilera och testkör programmet. Modellen ska nu avbryta träningen så snart precisionen överstiger tröskelvärdet. Skriv ut antal epoker och uppnådd precision i `train()` för att verifiera.

---

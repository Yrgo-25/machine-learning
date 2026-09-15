# L07 - Neurala nätverk (del III)

## Dagordning
* Vidare arbete med övningsuppgiften: Enkelt neuralt nätverk i C++ (del II).

---

## Mål med lektionen
* Ha skapat ett interface för neurala nätverk.
* Kunna koppla ihop komponenter via interfaces.
* Kunna skapa en enkel implementation av ett neuralt nätverk innehållande ett dolt lager samt ett utgångslager.

---

## Instruktioner

### Innan lektionen
* Läs igenom [bilaga A](./appendix/a_training_loop.md) för en genomgång av nätverkets struktur, träningsloopen samt hur feedforward, backpropagation och optimering hänger ihop i kod.

### Under lektionen
* Delta aktivt på genomgången.
* Genomför tillhörande övningsuppgift i [bilaga B](./appendix/b_exercises.md).
    * Ni får tid på er att genomföra uppgiften enskilt eller i grupp.
    * Uppgiften gås sedan igenom i helklass.

---

## Utvärdering
* Kan ni, utan att titta i koden, beskriva hur komponenterna i nätverket kopplas ihop via interfaces?
* Kan ni förklara flödet från indata till prediktion i er implementation, steg för steg?
* Varför returnerar `feedforward()` en `bool` i stället för lagrets utdata, och hur läser nätverkets övriga delar av resultatet istället?

---

## Nästa lektion
* Implementering av dense-lager i mjukvara (del I).

---

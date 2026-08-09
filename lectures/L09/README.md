# L09 - Dense-lager (del II)

## Dagordning
* Slutförande av implementering av dense-lager i mjukvara - feedforward, backpropagation och optimering.

---

## Mål med lektionen
* Ha implementerat metoderna `feedforward()`, `backpropagate()` och `optimize()` i dense-lager-klassen.
* Ha skapat en fullt fungerande dense-lager-implementation.
* Ha testat implementationen med ett komplett neuralt nätverk (det som skapades i övningsuppgiften i L06-L07).

---

## Instruktioner
* Fortsätt där vi arbetade senast; implementera de tre metoderna som lämnades tomma förra lektionen.
* Översätt de matematiska operationerna för feedforward, backpropagation och optimering som vi genomfört för hand till kod: 
    * Se [bilaga A](./appendix/a_math_to_code.md) för en översikt av hur matematiken motsvaras av koden.
    * Se [bilaga B](./appendix/b_exercises.md) för instruktioner gällande implementering av tidigarenämnda metoder.
* Testa implementationen genom att träna det neurala nätverk på befintlig träningsdata, justera vid behov antalet genomförda epoker och/eller lärhastigheten.

---

## Utvärdering
* Kan ni koppla ihop de matematiska formlerna för backpropagation med er implementation; vad representerar varje beräkningssteg i koden?
* Hur testar ni att implementationen är korrekt, och hur vet ni när nätverket är tillräckligt tränat?

---

## Nästa lektion (i januari)
* Övningstentamen: Implementering av neurala nätverk + tester i C++.

---

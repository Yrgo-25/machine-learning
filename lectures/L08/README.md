# L08 - Dense-lager (del I)

## Dagordning
* Implementering av dense-lager i mjukvara (med programkoden från övningsuppgiften i L06-L07):
    * Skapande av en skarp underklass `Dense`, som ersätter befintlig stubbklass.
    * Tillägg av ett fåtal aktiveringsfunktioner.

---

## Mål med lektionen
* Ha skapat en skarp underklass som ärver det befintliga dense-lager-interfacet.
* Ha implementerat samtliga metoder förutom `feedforward()`, `backpropagate()` samt `optimize()`; dessa kommer implementeras under nästa lektion.

---

## Instruktioner
* Använd er programkod från övningsuppgiften i L06-L07.
* Läs igenom [bilaga A](./appendix/a_architecture.md) för information om dense-lagrets arkitektur.
* Skapa en konkret implementation som ärver `ml::dense_layer::Interface`, se [bilaga B](./appendix/b_exercises.md) för detaljerade instruktioner.
* Implementera samtliga metoder förutom de ovannämnda - låt dessa ha en tom funktionskropp tills vidare.

---

## Utvärdering
* Vad är skillnaden mellan `ml::dense_layer::Interface` och er konkreta klass, och varför är denna uppdelning fördelaktig?
* Vilka aktiveringsfunktioner stödjer er implementation, och när lämpar sig respektive funktion?

---

## Nästa lektion
* Implementering av dense-lager i mjukvara (del II) - feedforward, backpropagation och optimering.

---

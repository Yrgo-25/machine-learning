# L13 - Konvolutionella neurala nätverk (del II)

## Dagordning
* Genomgång av ett komplett CNN i kod (`cnn_demo`): arkitektur och hur lagren hänger ihop.
* Översiktlig genomgång av träning för hand (bilaga A).
* Fortsatt arbete med **P03**.

---

## Mål med lektionen
* Kunna beskriva hur ett komplett CNN är uppbyggt i kod: hur conv-, maxpool-, flatten- och dense-lager hänger ihop från indata till prediktion.
* Ha en översiktlig förståelse för hur ett litet konvolutionellt neuralt nätverk tränas för hand.

---

## Instruktioner

### Innan lektionen
* Läs igenom [bilaga A](./appendix/a_cnn_by_hand.md) för en översiktlig genomgång av hur ett litet konvolutionellt neuralt nätverk tränas för hand.

### Under lektionen
* Delta aktivt på genomgången av [cnn_demo](./cnn_demo), ett komplett CNN (conv → maxpool → flatten → dense) som klassificerar 4×4-pixelbilder av siffrorna 0–3.
    * Bygg och kör med `make` i katalogen.

---

## Utvärdering
* Kan ni, utan att titta i bilagan, beskriva hur en kernel appliceras över en bild vid feedforward i ett conv-lager?
* I `cnn_demo`, hur hänger conv-, maxpool-, flatten- och dense-lagren ihop för att gå från indata till prediktion?
* Varför använder demo-CNN:ets dense-lager one-hot-kodning (en nod per siffra) i utdatan i stället för en enda utgångsnod?

---

## Nästa lektion
* Implementation av conv-lager i mjukvara.

---

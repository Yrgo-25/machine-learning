# Praktiska prov

## Information
Kursen innehåller två praktiska prov. Båda genomförs som inlämningsuppgifter på egen hand, med en
deadline, och redovisas därefter muntligt: ni visar er egen kod och förklarar den. AI-verktyg får
användas under arbetet, men koden ska skrivas av er själva, och ni ska kunna redogöra för varje
test och varje rättning ni lämnar in.

Proven täcker följande:
* Det första provet täcker:
    * Feedforward, backpropagation samt optimering (gradient descent) i dense-lager.
    * Aktiveringsfunktioner och val av dessa.
    * Träning för hand av ett litet neuralt nätverk.
    * Praktiska aspekter av träning: randomisering av träningsordning, lärhastighet.
* Det andra provet täcker:
    * Feedforward, backpropagation samt optimering i konvolutionella lager (Conv), maxpooling-lager samt flatten-lager.
    * Varför CNN-nätverk används vid bildklassificering, samt kernels och pooling-lager.
    * Träning för hand av ett litet konvolutionellt neuralt nätverk.
    * Enhets- och komponenttester av CNN-lager med testramverket `yrgo::test`.

Båda proven är gemensamma med kursen **Mjuk- och hårdvarutestning** och betygssätts separat per
kurs. Lektionerna i den här kursen (se [föreläsningar](../lectures/README.md)) utgör förberedelse
inför dem; själva uppgifterna genomförs på egen hand mellan lektionstillfällena.

---

## Övningstentor
* [Övningstentamen 1](./exam1/practice_exam1.md) (neurala nätverk) övar samma kunskaper som det
  första praktiska provet, och i samma form: en färdig implementation innehåller sex planterade
  buggar som ska hittas med enhets- och komponenttester och rättas.
  Lösningsförslag finns [här](./exam1/practice_exam1_solution.md).
* [Hemtentamen 2](./exam2/home_exam2.md) (konvolutionella neurala nätverk) genomförs på egen hand:
  en färdig CNN-implementation innehåller åtta planterade buggar som ska hittas med enhets- och
  komponenttester och rättas.

---

# Praktiska prov

## Information
Kursen innehåller två praktiska prov:
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

Båda proven är gemensamma med kursen **Mjuk- och hårdvarutestning** och betygssätts separat per kurs. De skarpa proven genomförs under den kursens egna lektionstillfällen; motsvarande lektioner i den här kursen (se [föreläsningar](../lectures/README.md)) utgör enbart förberedelse och övning inför dem.

---

## Övningstentor
* [Övningstentamen 1](./exam1/practice_exam1.md) (neurala nätverk) övar samma kunskaper som det
  första praktiska provet, men i form av felsökning: en färdig implementation innehåller sex
  planterade buggar som ska hittas med enhets- och komponenttester och rättas.
  Lösningsförslag finns [här](./exam1/practice_exam1_solution.md).
* [Hemtentamen 2](./exam2/home_exam2.md) (konvolutionella neurala nätverk) genomförs på egen hand:
  en färdig CNN-implementation innehåller åtta planterade buggar som ska hittas med enhets- och
  komponenttester och rättas.

---

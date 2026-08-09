# L16 - Konvolutionella neurala nätverk (del V)

## Dagordning
* Implementation av flatten-lager i mjukvara.
* Fortsatt arbete med **P03**.

---

## Mål med lektionen
* Kunna implementera enkla flatten-lager i mjukvara.

---

## Instruktioner
* Öppna filen [flatten_demo.cpp](./flatten_layer/cpp/flatten_demo.cpp). 
En strukt döpt `ml::FlattenLayer` ska läggas till i denna fil att realisera enkla flatten-lager. \
Se mer information i [bilaga A](./appendix/a_flatten_layer.md).
* Som referens, se motsvarande implementation i Python [här](./flatten_layer/python/flatten_demo.py). \
Testkör programmet via kommandot `python3 flatten_demo.py` i terminalen i motsvarande katalog.
* Om ni är intresserade finns en något mer utvecklad implementation i C [här](./flatten_layer/c/source/flatten_demo.c).
    * I denna implementation demonstreras inkapsling genom att strukten `flatten_layer_t` bara implementeras i källkodsfilen 
    [flatten_layer.c](./flatten_layer/c/source/ml/flatten_layer.c). 
    * Motsvarande headerfil  [flatten_layer.h](./flatten_layer/c/include/ml/flatten_layer.h) innehåller endast funktioner 
    för att använda flatten-lager.
    * Struktens medlemsvariabler hålls därmed privata, på samma sätt som nyckelordet `private` används i C++. 
    * Här demonstreras även hur matriser kan implementeras i C via strukten [matrix_t](./flatten_layer/c/include/ml/matrix.h).

---

## Utvärdering
* Varför behövs ett flatten-lager mellan de konvolutionella/pooling-lagren och dense-lagret?
* Vad gör flatten-lagrets `backpropagate()`, och hur skiljer sig den beräkningen från `feedforward()`?

---

## Nästa lektion
* Arbete med **P03**.

---

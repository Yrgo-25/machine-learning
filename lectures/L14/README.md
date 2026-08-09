# L14 - Konvolutionella neurala nätverk (del III)

## Dagordning
* Implementation av conv-lager i mjukvara.
* Fortsatt arbete med **P03**.

---

## Mål med lektionen
* Kunna implementera enkla conv-lager i mjukvara.

---

## Instruktioner
* Öppna filen [conv_demo.cpp](./conv_layer/cpp/conv_demo.cpp). En strukt döpt `ml::ConvLayer` ska läggas till i denna
fil att realisera enkla conv-lager. \
Se mer information i [bilaga A](./appendix/a_conv_layer.md).
* Som referens, se motsvarande implementation i Python [här](./conv_layer/python/conv_demo.py). \
Testkör programmet via kommandot `python3 conv_demo.py` i terminalen i motsvarande katalog.
* Om ni är intresserade finns en något mer utvecklad implementation i C [här](./conv_layer/c/source/conv_demo.c).
    * I denna implementation demonstreras inkapsling genom att strukten `conv_layer_t` bara implementeras i källkodsfilen 
    [conv_layer.c](./conv_layer/c/source/ml/conv_layer.c). 
    * Motsvarande headerfil [conv_layer.h](./conv_layer/c/include/ml/conv_layer.h) innehåller endast funktioner 
    för att använda conv-lager.
    * Struktens medlemsvariabler hålls därmed privata, på samma sätt som nyckelordet `private` används i C++. 
    * Här demonstreras även hur matriser kan implementeras i C via strukten [matrix_t](./conv_layer/c/include/ml/matrix.h).

---

## Utvärdering
* Vilka delar av er `ml::ConvLayer`-implementation motsvarar feedforward respektive backpropagation?
* Varför använder vi samma kernel (samma vikter) över hela bilden i stället för unika vikter per position?

---

## Nästa lektion
* Implementation av maxpooling-lager i mjukvara.
* Arbete med **P03**.

---

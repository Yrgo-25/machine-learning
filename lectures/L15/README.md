# L15 - Konvolutionella neurala nätverk (del IV)

## Dagordning
* Implementation av maxpooling-lager i mjukvara.
* Fortsatt arbete med **P03**.

---

## Mål med lektionen
* Kunna implementera enkla maxpooling-lager i mjukvara.

---

## Instruktioner
* Öppna filen [max_pool_demo.cpp](./max_pool_layer/cpp/max_pool_demo.cpp). 
En strukt döpt `ml::MaxPoolLayer` ska läggas till i denna fil att realisera enkla maxpooling-lager. \
Se mer information i [bilaga A](./appendix/a_max_pool_layer.md).
* Som referens, se motsvarande implementation i Python [här](./max_pool_layer/python/max_pool_demo.py). \
Testkör programmet via kommandot `python3 max_pool_demo.py` i terminalen i motsvarande katalog.
* Om ni är intresserade finns en något mer utvecklad implementation i C [här](./max_pool_layer/c/source/max_pool_demo.c).
    * I denna implementation demonstreras inkapsling genom att strukten `max_pool_layer_t` bara implementeras i källkodsfilen 
    [max_pool_layer.c](./max_pool_layer/c/source/ml/max_pool_layer.c). 
    * Motsvarande headerfil  [max_pool_layer.h](./max_pool_layer/c/include/ml/max_pool_layer.h) innehåller endast funktioner 
    för att använda maxpooling-lager.
    * Struktens medlemsvariabler hålls därmed privata, på samma sätt som nyckelordet `private` används i C++. 
    * Här demonstreras även hur matriser kan implementeras i C via strukten [matrix_t](./max_pool_layer/c/include/ml/matrix.h).

---

## Utvärdering
* Hur vet er `ml::MaxPoolLayer`-implementation vilka positioner gradienterna ska spridas tillbaka till vid backpropagation?
* Vilka skillnader finns mellan ett maxpooling-lager och ett dense- eller conv-lager, sett till exempelvis parametrar och beräkningar?

---

## Nästa lektion
* Implementation av flatten-lager i mjukvara.

---

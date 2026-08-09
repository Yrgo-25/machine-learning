# Bilaga A - Skapande av enkla maxpooling-lager i C++

### Uppgiftsbeskrivning
En strukt döpt `ml::MaxPoolLayer` ska läggas till i
[max_pool_demo.cpp](../max_pool_layer/cpp/max_pool_demo.cpp) för att realisera enkla
maxpooling-lager. För att hålla det så enkelt som möjligt implementerar vi en strukt och slopar
get- och set-metoder, radering av copy- och move-konstruktorer med mera.

Studera koden i funktionen `main()`. Er implementation ska skrivas så att denna kod fungerar för att
skapa och använda ett maxpooling-lager döpt `poolLayer`:

```cpp
// Create a max pooling layer: 4x4 input, 2x2 pooling regions, produces 2x2 output.
constexpr std::size_t inputSize{4U};
constexpr std::size_t poolSize{2U};
ml::MaxPoolLayer poolLayer{inputSize, poolSize};

// Example 4x4 input matrix (could represent an image or feature map).
const Matrix2d input{{2, 1, 6, 1},
                     {3, 0, 4, 6},
                     {1, 2, 4, 5},
                     {3, 4, 7, 7}};
        
// Perform feedforward (pooling).
poolLayer.feedforward(input);

// Example output gradients (same shape as pooling output, used for backpropagation demo).
const Matrix2d outputGradients{{1, 2},
                               {3, 4}};
// Perform backpropagation.
poolLayer.backpropagate(outputGradients);
```

### Kompilering samt exekvering av programmet
Som vanligt kan programmet köras genom att skriva kommandot `make` i terminalen:

```bash
make
```

När implementationen är färdig, avkommentera kompilatorflaggan `CXX_FLAGS` i
[Makefilen](../max_pool_layer/cpp/Makefile). Ändra alltså från följande:

```bash
# TODO: Uncomment this line once the implementation is finished.
#CXX_FLAGS := -std=c++17 -Wall -Werror -DMAX_POOL_LAYER_IMPLEMENTED
```

till

```bash
CXX_FLAGS := -std=c++17 -Wall -Werror -DMAX_POOL_LAYER_IMPLEMENTED
```

Ni kan då också ta bort villkorskompileringsdirektivet `MAX_POOL_LAYER_IMPLEMENTED` ur
[max_pool_demo.cpp](../max_pool_layer/cpp/max_pool_demo.cpp) om ni vill. I så fall, ändra alltså
från följande:

```cpp
/**
 * @brief Create and demonstrate a simple max pooling layer.
 * 
 * @return 0 on success, -1 on failure.
 */
int main()
{
//! @todo Remove this guard (and/or uncomment the compiler flags in the makefile) once the
//        implementation is finished.
#ifdef MAX_POOL_LAYER_IMPLEMENTED
    
    // Function content.
    return 0;

//! @todo Remove this guard (and/or uncomment the compiler flags in the makefile) once the
//        implementation is finished.
#endif /** MAX_POOL_LAYER_IMPLEMENTED */
}
```

till

```cpp
/**
 * @brief Create and demonstrate a simple max pooling layer.
 * 
 * @return 0 on success, -1 on failure.
 */
int main()
{
    // Function content.
    return 0;
}
```

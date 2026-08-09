# Bilaga A - Linjär regression i C++

Nedan visas en minimalistisk implementation av linjär regression i C++ med en strukt `LinReg` som:
* Håller modellens parametrar `bias` och `weight`.
* Predikterar med metoden `predict()`.
* Tränas med metoden `train()`.

**OBS!** Ett alias `Matrix1d` används som substitut för `std::vector<double>`.

Nedan visas den fullständiga implementationen:

```cpp
/**
 * @brief Minimalistic linear regression example.
 */
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <vector>

namespace
{
/** One-dimensional matrix. */
using Matrix1d = std::vector<double>;

/**
 * @brief Linear regression structure.
 */
struct LinReg
{
    /** Bias value. */
    double bias;

    /** Weight value. */
    double weight;

    /**
     * @brief Predict with the given linear regression model.
     * 
     * @param[in] input Input value.
     * 
     * @return The predicted value.
     */
    double predict(const double input) const noexcept { return weight * input + bias; }

    /**
     * @brief Train the given linear regression model with given training sets.
     * 
     * @param[in] trainIn Input values. Size must be greater than 0.
     * @param[in] trainOut Output values. Size must be greater than 0.
     * @param[in] epochCount Number of epochs to train. Must be greater than 0.
     * @param[in] learningRate Learning rate (default = 1 %). Must be in range (0.0, 1.0).
     * 
     * @return True if training was successful, false otherwise.
     */
    bool train(const Matrix1d& trainIn, const Matrix1d& trainOut, 
               const std::uint32_t epochCount, const double learningRate = 0.01) noexcept
    {
        // Check input parameters and training set count, terminate if invalid.
        const auto setCount = static_cast<std::uint32_t>(std::min(trainIn.size(), trainOut.size()));

        if (0U == setCount)
        {
            std::fprintf(stderr, "Invalid training data: no training sets provided!\n");
            std::terminate();
        }
        if (0U == epochCount)
        {
            std::fprintf(stderr, "Invalid epoch count: must train for at least one epoch!\n");
            std::terminate();
        }
        if ((0.0 >= learningRate) || (1.0 <= learningRate))
        {
            std::fprintf(stderr, "Invalid learning rate %g: must be in range (0.0, 1.0)!\n",
                learningRate);
            std::terminate();
        }

        // Train the model for the given number of epochs.
        for (std::uint32_t epoch{}; epoch < epochCount; ++epoch)
        {
            // Iterate through all training sets.
            for (std::uint32_t i{}; i < setCount; ++i)
            {
                const auto input  = trainIn[i];
                const auto output = trainOut[i];
                
                // Perform prediction and calculate the error.
                const auto prediction = predict(input);
                const auto error      = output - prediction;

                // Adjust the parameters in accordance with the error.
                bias   += error * learningRate;
                weight += error * learningRate * input;
            }
        }
        return true;
    }
};
} // namespace

/**
 * @brief Train and predict with a simple linear regression model.
 * 
 * @return 0 on success, -1 on failure.
 */
int main()
{
    constexpr std::uint32_t epochCount{100U};
    constexpr double learningRate{0.1};

    // Create linear regression model to predict y = 3x + 1.
    const Matrix1d trainIn{0.0, 1.0, 2.0, 3.0, 4.0};
    const Matrix1d trainOut{1.0, 4.0, 7.0, 10.0, 13.0};
    LinReg linReg{};

    // Train the model, terminate on failure.
    if (!linReg.train(trainIn, trainOut, epochCount, learningRate))
    {
        std::printf("Training failed!\n");
        return -1;
    }
    // Predict and print result on success.
    for (const auto& input : trainIn)
    {
        const auto prediction = linReg.predict(input);
        std::printf("x = %g, ypred = %g\n", input, prediction);
    }
    return 0;
}
```

Modellen ovan är avsiktligt minimalistisk. Den kan byggas ut med exempelvis:
* Specialfall för `x = 0`, där biasvärdet sätts direkt till referensvärdet (`bias = output`). Värt att veta är att detta är en genväg snarare än en bättre inlärningsregel: den skriver över en tränad parameter utifrån en enda träningsuppsättning, så med brusig data, eller med fler än en uppsättning där `x = 0`, blir det slutliga biasvärdet helt enkelt det som besöktes sist. Den vanliga uppdateringsregeln hanterar redan `x = 0` korrekt, eftersom `k` inte påverkar en prediktion gjord med indatan noll.
* Randomisering av träningsordningen för att undvika att modellen blir för bekant med träningsdatan.
* Beräkning av medelfel per epok för att följa modellens träningsförlopp.

---

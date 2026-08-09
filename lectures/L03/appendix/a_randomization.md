# Bilaga A - Linjär regression med randomisering i C++
Om modellen alltid tränas i samma ordning riskerar den att bli för anpassad till just den sekvensen; den lär sig mönstret i ordningen snarare än i datan. Genom att blanda träningsordningen varje epok exponeras modellen för datan i varierande ordning, vilket ger en mer robust träning och hindrar parametrarna från att falla in i en rytm som dikteras av ordningen. För en linjär modell finns inga lokala minima att fastna i, eftersom dess kvadratiska felyta har exakt ett minimum; den nyttan kommer först med de djupare, icke-linjära nätverken från **L05** och framåt, där samma blandning även hjälper parametrarna att ta sig ur dåliga områden av felytan.

Nedan visas en utökning av regressionsstrukturen från L02 med randomiserad träningsordning. Struktens `train()`-metod:
* Skapar en indexvektor för träningsuppsättningarna.
* Blandar indexvektorn i slumpvis ordning inför varje epok.
* Itererar genom träningsuppsättningarna i den randomiserade ordningen.

**OBS!** Utöver `Matrix1d` används även aliaset `MatrixU32` som substitut för `std::vector<std::uint32_t>`.

Nedan visas den fullständiga implementationen:

```cpp
/**
 * @brief Linear regression with randomized training order.
 */
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <exception>
#include <vector>

namespace
{
/** One-dimensional matrix. */
using Matrix1d = std::vector<double>;

/** Unsigned 32-bit integer matrix. */
using MatrixU32 = std::vector<std::uint32_t>;

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

        // Create train order matrix.
        MatrixU32 trainOrder{trainOrderMatrix(setCount)};

        // Train the model for the given number of epochs.
        for (std::uint32_t epoch{}; epoch < epochCount; ++epoch)
        {
            // Randomize training order each epoch.
            shuffle(trainOrder);

            // Iterate through all training sets in randomized order.
            for (const auto& i : trainOrder)
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

private:
     // -----------------------------------------------------------------------------
    static MatrixU32 trainOrderMatrix(const std::uint32_t setCount) noexcept
    {
        MatrixU32 order(setCount);
        for (std::uint32_t i{}; i < order.size(); ++i) { order[i] = i; }
        return order;
    }

    // -----------------------------------------------------------------------------
    static void shuffle(MatrixU32& order) noexcept
    {
        for (std::uint32_t i{}; i < order.size(); ++i)
        {
            // Swap values at index r and i.
            const auto r    = static_cast<std::uint32_t>(std::rand() % order.size());
            const auto temp = order[i];
            order[i]        = order[r];
            order[r]        = temp;
        }
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

    // Initialize the random generator.
    std::srand(std::time(nullptr));

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

Modellen ovan bygger vidare på L02 med randomiserad träningsordning. Nästa utökning är beräkning av medelfel per epok för att följa modellens träningsförlopp. Detta behandlas i L04.

---

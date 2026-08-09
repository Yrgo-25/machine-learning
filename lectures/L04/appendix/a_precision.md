# Bilaga A - Linjär regression med precisionsberäkning i C++
Utan precisionsberäkning tränas modellen ett fast antal epoker oavsett hur väl den faktiskt presterar. Genom att beräkna precisionen efter varje epok kan träningen avslutas så snart modellen är tillräckligt tränad, vilket sparar tid och undviker onödig träning.

Precisionen beräknas som `1.0 - MAE`, där MAE (*mean absolute error*) är det genomsnittliga absolutfelet över samtliga träningsuppsättningar. En precision nära `1.0` innebär att modellen predikterar mycket nära referensvärdena.

**En notis om namnet.** Detta är inte *precision* i den bemärkelse som används inom maskininlärning i övrigt, där precision anger hur stor andel av de positiva prediktionerna som var korrekta och bara är tillämpbart vid klassificering. Det som beräknas här är ett passningsmått för en regressionsmodell, med två egenskaper värda att känna till: det är skalberoende, eftersom ett absolutfel på `0.1` betyder helt olika saker för utdata kring `1` respektive kring `1000`, och det kan bli negativt, eftersom ingenting hindrar att det genomsnittliga absolutfelet överstiger `1.0`. Att läsa `0.999999` som "99.9999 % rätt" är alltså en bekväm förenkling snarare än en procentandel av någonting.

Nedan visas en utökning av regressionsstrukturen från L03 med precisionsberäkning. Den privata metoden `computePrecision()` beräknar precisionen, och `train()` avslutar träningen tidigt om precisionen överstiger ett givet tröskelvärde.

Nedan visas den fullständiga implementationen:

```cpp
/**
 * @brief Linear regression with precision calculation.
 */
#include <algorithm>
#include <cmath>
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
     * @param[in] trainIn    Input values. Size must be greater than 0.
     * @param[in] trainOut   Output values. Size must be greater than 0.
     * @param[in] epochCount Number of epochs to train. Must be greater than 0.
     * @param[in] learningRate Learning rate (default = 1 %). Must be in range (0.0, 1.0).
     * @param[in] precision  Desired precision (default = 99.9999 %). Must be in range (0.0, 1.0).
     * 
     * @return True if training was successful, false otherwise.
     */
    bool train(const Matrix1d& trainIn, const Matrix1d& trainOut, 
               const std::uint32_t epochCount, const double learningRate = 0.01,
               const double precision = 0.999999) noexcept
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
        if ((0.0 >= precision) || (1.0 <= precision))
        {
            std::fprintf(stderr, "Invalid precision %g: must be in range (0.0, 1.0)!\n",
                precision);
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

            // Calculate the precision, return true if exceeded.
            const auto currentPrecision = computePrecision(trainIn, trainOut, setCount);
            if (currentPrecision >= precision)
            { 
                const auto percent = currentPrecision * 100.0;
                std::printf("Reached %g %% precision after %u epochs!\n", percent, epoch + 1U);
                return true;
            }
        }
        return true;
    }

private:
    static MatrixU32 trainOrderMatrix(const std::uint32_t setCount) noexcept
    {
        MatrixU32 order(setCount);
        for (std::uint32_t i{}; i < order.size(); ++i) { order[i] = i; }
        return order;
    }

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

    double computePrecision(const Matrix1d& trainIn, const Matrix1d& trainOut,
                            const std::uint32_t setCount) const noexcept
    {
        double totalError{};
        for (std::uint32_t i{}; i < setCount; ++i)
        {
            const auto input      = trainIn[i];
            const auto output     = trainOut[i];
            const auto prediction = predict(input);
            const auto error      = output - prediction;

            totalError += std::abs(error);
        }
        const auto avgError = totalError / setCount;
        return 1.0 - avgError;
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

Modellen ovan bygger vidare på L03 med precisionsberäkning och tidig avstängning av träningen. Nästa steg är att tillämpa dessa tekniker i ett inbyggt system via **P03**.

---

/**
 * @file Linear regression model interface.
 */
#pragma once

// Skapa namespaces som speglar katalogstrukturen.
namespace ml::lin_reg
{
// Skapa en klass som heter Interface.
// Skapa en virtuell destruktor, markera den default.
// Skapa en metod som heter predict(), som returnerar ett flyttal och tar in 
// ett flyttal.

/**
 * @brief Linear regression model interface.
 */
class Interface
{
public:
    /**
     * @brief Destructor.
     */
    virtual ~Interface() noexcept = default;

    /**
     * @brief Predict based on the given input.
     * 
     * @param[in] input Input to predict with.
     * 
     * @return Predicted output.
     */
    [[nodiscard]] virtual double predict(double input) const noexcept = 0;
};
} // namespace ml::lin_reg

/**
 * @file Test runner entry point.
 */
#include "yrgo/test/test.h"

/**
 * @brief Run all unit tests.
 *
 * @return 0 on success, -1 on failure.
 */
int main() { return yrgo::test::runAllTests() ? 0 : -1; }

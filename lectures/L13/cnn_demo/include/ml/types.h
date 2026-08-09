/**
 * @file Generic machine learning types.
 */
#pragma once

#include <cstddef>
#include <memory>
#include <vector>

// clang-format off

/** Activation function interface. */
namespace ml::act_func { class Interface; }

/** Convolutional layer interface. */
namespace ml::conv_layer { class Interface; }

/** Dense layer interface. */
namespace ml::dense_layer { class Interface; }

/** Factory interface. */
namespace ml::factory { class Interface; }

/** Flatten layer interface. */
namespace ml::flatten_layer { class Interface; }

// clang-format on

namespace ml
{
/** Matrix types. */
using Matrix1d = std::vector<double>;
using Matrix2d = std::vector<Matrix1d>;
using Matrix3d = std::vector<Matrix2d>;

/** Pointer types. */
using ActFuncPtr      = std::unique_ptr<act_func::Interface>;
using ConvLayerPtr    = std::unique_ptr<conv_layer::Interface>;
using DenseLayerPtr   = std::unique_ptr<dense_layer::Interface>;
using FactoryPtr      = std::unique_ptr<factory::Interface>;
using FlattenLayerPtr = std::unique_ptr<flatten_layer::Interface>;

/** List types. */
using ConvLayerList  = std::vector<ConvLayerPtr>;
using DenseLayerList = std::vector<DenseLayerPtr>;
using TrainOrderList = std::vector<std::size_t>;

} // namespace ml

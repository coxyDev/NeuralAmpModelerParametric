#include "gru.h"

#include <cmath>

namespace nam
{

static float sigmoid(float x)
{
  return 1.0f / (1.0f + std::exp(-x));
}

GRUCell::GRUCell(int input_size, int hidden_size)
: _input_size(input_size)
, _hidden_size(hidden_size)
, _W_ih(3 * hidden_size, input_size)
, _W_hh(3 * hidden_size, hidden_size)
, _b_ih(3 * hidden_size)
, _b_hh(3 * hidden_size)
, _head_weight(hidden_size)
, _head_bias(0.0f)
, _hidden(Eigen::VectorXf::Zero(hidden_size))
, _gates_ih(3 * hidden_size)
, _gates_hh(3 * hidden_size)
, _input_vec(input_size)
{
  _W_ih.setZero();
  _W_hh.setZero();
  _b_ih.setZero();
  _b_hh.setZero();
  _head_weight.setZero();
}

void GRUCell::set_weights_(std::vector<float>::iterator& weights)
{
  // PyTorch GRUCell weight order:
  // weight_ih_l0: (3*hidden, input) row-major
  // weight_hh_l0: (3*hidden, hidden) row-major
  // bias_ih_l0: (3*hidden)
  // bias_hh_l0: (3*hidden)

  const int rows_3h = 3 * _hidden_size;

  // W_ih
  for (int r = 0; r < rows_3h; ++r)
    for (int c = 0; c < _input_size; ++c)
      _W_ih(r, c) = *weights++;

  // W_hh
  for (int r = 0; r < rows_3h; ++r)
    for (int c = 0; c < _hidden_size; ++c)
      _W_hh(r, c) = *weights++;

  // b_ih
  for (int i = 0; i < rows_3h; ++i)
    _b_ih(i) = *weights++;

  // b_hh
  for (int i = 0; i < rows_3h; ++i)
    _b_hh(i) = *weights++;

  // Linear head: weight (hidden_size) then bias (1)
  for (int i = 0; i < _hidden_size; ++i)
    _head_weight(i) = *weights++;
  _head_bias = *weights++;
}

float GRUCell::ProcessSample(float input)
{
  const int h = _hidden_size;

  _input_vec(0) = input; // Assuming input_size == 1

  // Compute gate pre-activations
  _gates_ih = _W_ih * _input_vec + _b_ih; // (3h)
  _gates_hh = _W_hh * _hidden + _b_hh;   // (3h)

  // Reset gate: r = sigmoid(gates_ih[0:h] + gates_hh[0:h])
  // Update gate: z = sigmoid(gates_ih[h:2h] + gates_hh[h:2h])
  // New gate: n = tanh(gates_ih[2h:3h] + r * gates_hh[2h:3h])
  Eigen::VectorXf r(h), z(h), n(h);

  for (int i = 0; i < h; ++i)
  {
    r(i) = sigmoid(_gates_ih(i) + _gates_hh(i));
    z(i) = sigmoid(_gates_ih(h + i) + _gates_hh(h + i));
    n(i) = std::tanh(_gates_ih(2 * h + i) + r(i) * _gates_hh(2 * h + i));
  }

  // Hidden state update: h' = (1 - z) * n + z * h
  for (int i = 0; i < h; ++i)
  {
    _hidden(i) = (1.0f - z(i)) * n(i) + z(i) * _hidden(i);
  }

  // Linear head
  float output = _hidden.dot(_head_weight) + _head_bias;
  return output;
}

void GRUCell::Reset()
{
  _hidden.setZero();
}

} // namespace nam

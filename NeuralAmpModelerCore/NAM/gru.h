#pragma once
// gru.h — Minimal GRU cell for nonlinear processing in parametric amp
//
// With hidden_size=1, acts as a dynamic, history-dependent nonlinearity
// capturing tube behavior (bias drift, thermal inertia, asymmetric clipping).

#include <Eigen/Dense>
#include <vector>

namespace nam
{

/// Minimal GRU cell for sample-by-sample nonlinear processing
///
/// Implements the standard GRU equations:
///   r = sigmoid(W_ir * x + b_ir + W_hr * h + b_hr)     // reset gate
///   z = sigmoid(W_iz * x + b_iz + W_hz * h + b_hz)     // update gate
///   n = tanh(W_in * x + b_in + r * (W_hn * h + b_hn))  // new gate
///   h' = (1 - z) * n + z * h                            // new hidden state
///
/// Output = head_weight * h' + head_bias
class GRUCell
{
public:
  /// \param input_size Input dimension (typically 1 for mono audio)
  /// \param hidden_size Hidden state dimension (typically 1 for nonlinearity use)
  GRUCell(int input_size = 1, int hidden_size = 1);

  /// Load weights from iterator (matches PyTorch GRUCell + Linear head export order)
  void set_weights_(std::vector<float>::iterator& weights);

  /// Process a single input sample, updating internal hidden state
  /// \return Output value
  float ProcessSample(float input);

  /// Reset hidden state to zeros
  void Reset();

  int GetInputSize() const { return _input_size; }
  int GetHiddenSize() const { return _hidden_size; }

private:
  int _input_size;
  int _hidden_size;

  // GRUCell weights: W_ih (3*hidden x input), W_hh (3*hidden x hidden)
  Eigen::MatrixXf _W_ih; // (3*hidden_size, input_size)
  Eigen::MatrixXf _W_hh; // (3*hidden_size, hidden_size)
  Eigen::VectorXf _b_ih; // (3*hidden_size)
  Eigen::VectorXf _b_hh; // (3*hidden_size)

  // Linear head: hidden -> 1
  Eigen::VectorXf _head_weight; // (hidden_size)
  float _head_bias;

  // State
  Eigen::VectorXf _hidden; // (hidden_size)

  // Scratch buffers (avoid allocation in process loop)
  Eigen::VectorXf _gates_ih;
  Eigen::VectorXf _gates_hh;
  Eigen::VectorXf _input_vec;
};

} // namespace nam
